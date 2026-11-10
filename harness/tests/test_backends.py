from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import unittest
from unittest import mock

from harness.backends import (BackendError, ContextOverflow, FakeBackend, OllamaBackend,
                              OpenAICompatBackend, RetryableError, fits, make_backend, with_retries)


class Stub:
    """A tiny HTTP server answering every POST with (status, json body)."""

    def __init__(self, status, body):
        stub = self
        self.requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                data = self.rfile.read(int(self.headers["Content-Length"]))
                stub.requests.append((self.path, dict(self.headers), json.loads(data)))
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class TestBackends(unittest.TestCase):
    def stub(self, status, body):
        s = Stub(status, body)
        self.addCleanup(s.close)
        return s

    def test_fits(self):
        self.assertTrue(fits("x" * 4000, 8192))
        self.assertFalse(fits("x" * 40000, 8192))

    def test_fake_cycles(self):
        f = FakeBackend(["a", "b"])
        self.assertEqual([f.complete("p").text for _ in range(3)], ["a", "b", "a"])
        self.assertEqual(len(f.calls), 3)

    def test_ollama_ok(self):
        s = self.stub(200, {"message": {"content": "hi"}, "prompt_eval_count": 10, "eval_count": 2})
        c = OllamaBackend("m", base_url=s.url).complete("hello")
        self.assertEqual((c.text, c.prompt_tokens, c.response_tokens), ("hi", 10, 2))
        path, _, body = s.requests[0]
        self.assertEqual(path, "/api/chat")
        self.assertEqual(body["options"]["num_ctx"], 32768)
        self.assertFalse(body["stream"])

    def test_ollama_silent_truncation(self):
        s = self.stub(200, {"message": {"content": "hi"}, "prompt_eval_count": 32768, "eval_count": 2})
        with self.assertRaises(ContextOverflow):
            OllamaBackend("m", base_url=s.url).complete("hello")

    def test_openai_ok_and_auth(self):
        s = self.stub(200, {"choices": [{"message": {"content": "yo"}}],
                            "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            c = OpenAICompatBackend("m", base_url=s.url).complete("hello")
        self.assertEqual((c.text, c.prompt_tokens, c.response_tokens), ("yo", 7, 3))
        path, headers, _ = s.requests[0]
        self.assertEqual(path, "/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer k")

    def test_openai_context_error(self):
        s = self.stub(400, {"error": {"message": "This model's maximum context length is 8192"}})
        with self.assertRaises(ContextOverflow):
            OpenAICompatBackend("m", base_url=s.url, api_key_env=None).complete("hello")

    def test_http_429_retryable(self):
        s = self.stub(429, {"error": "slow down"})
        with self.assertRaises(RetryableError):
            OllamaBackend("m", base_url=s.url).complete("hello")
        s2 = self.stub(401, {"error": "nope"})
        with self.assertRaises(BackendError) as cm:
            OllamaBackend("m", base_url=s2.url).complete("hello")
        self.assertNotIsInstance(cm.exception, RetryableError)

    def test_connection_refused_retryable(self):
        s = self.stub(200, {})
        url = s.url
        s.close()
        with self.assertRaises(RetryableError):
            OllamaBackend("m", base_url=url, timeout=2).complete("hello")

    def test_with_retries(self):
        slept, n = [], {"calls": 0}

        def flaky():
            n["calls"] += 1
            if n["calls"] < 3:
                raise RetryableError("busy")
            return 5

        self.assertEqual(with_retries(flaky, sleep=slept.append), 5)
        self.assertEqual(slept, [10.0, 20.0])

        def always():
            raise RetryableError("down")

        slept.clear()
        with self.assertRaises(RetryableError):
            with_retries(always, sleep=slept.append)
        self.assertEqual(len(slept), 5)

    def test_ollama_num_predict_and_done_reason(self):
        s = self.stub(200, {"message": {"content": "hi"}, "prompt_eval_count": 10, "eval_count": 2,
                            "done_reason": "length"})
        c = OllamaBackend("m", base_url=s.url, context_tokens=8192).complete("x" * 4000)
        self.assertEqual(c.finish_reason, "length")
        opts = s.requests[0][2]["options"]
        self.assertEqual(opts["num_predict"], 8192 - 1150)

    def test_ollama_num_predict_capped(self):
        s = self.stub(200, {"message": {"content": "hi"}, "prompt_eval_count": 10, "eval_count": 2})
        OllamaBackend("m", base_url=s.url, context_tokens=131072).complete("hello")
        self.assertEqual(s.requests[0][2]["options"]["num_predict"], 16384)

    def test_openai_null_content_and_finish(self):
        s = self.stub(200, {"choices": [{"message": {"content": None}, "finish_reason": "length"}]})
        c = OpenAICompatBackend("m", base_url=s.url, api_key_env=None).complete("hello")
        self.assertEqual((c.text, c.finish_reason), ("", "length"))

    def test_openai_malformed_is_backend_error(self):
        s = self.stub(200, {"unexpected": True})
        with self.assertRaises(BackendError):
            OpenAICompatBackend("m", base_url=s.url, api_key_env=None).complete("hello")

    def test_ollama_describe(self):
        s = self.stub(200, {"details": {"quantization_level": "Q4_K_M"}, "digest": "abc123"})
        d = OllamaBackend("m", base_url=s.url, temperature=0.7).describe()
        self.assertEqual((d["quantization"], d["temperature"]), ("Q4_K_M", 0.7))

    def test_claude_context_error_not_rate_limit(self):
        from harness.backends import ClaudeCliBackend
        fake = mock.Mock(returncode=1, stdout="", stderr="Error: prompt is too long: context limit exceeded")
        with mock.patch("harness.backends.subprocess.run", return_value=fake):
            with self.assertRaises(ContextOverflow):
                ClaudeCliBackend().complete("hi")

    def test_make_backend(self):
        self.assertEqual(make_backend("ollama", "m").model, "m")
        self.assertEqual(make_backend("openai", "gpt", base_url="http://x").name, "openai")
        self.assertEqual(make_backend("claude-cli", "sonnet").name, "claude-cli")
        with self.assertRaises(ValueError):
            make_backend("nope", "m")


if __name__ == "__main__":
    unittest.main()
