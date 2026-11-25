from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import unittest
from unittest import mock

from harness.backends import (BackendError, ContextOverflow, FakeBackend, LlamaCppBackend, OllamaBackend,
                              OpenAICompatBackend, RetryableError, fits, make_backend, with_retries)


class Stub:
    """A tiny HTTP server answering every POST with (status, json body)."""

    def __init__(self, status, body, get_body=None):
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

            def do_GET(self):
                stub.requests.append((self.path, dict(self.headers), None))
                payload = json.dumps(get_body if get_body is not None else {}).encode()
                self.send_response(200 if get_body is not None else 404)
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
    def stub(self, status, body, get_body=None):
        s = Stub(status, body, get_body)
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

    def test_ollama_think_level(self):
        s = self.stub(200, {"message": {"content": "hi", "thinking": "hmm"}, "prompt_eval_count": 1, "eval_count": 1})
        b = OllamaBackend("gpt-oss:20b", base_url=s.url, think="low")
        b.complete("hello")
        self.assertEqual(s.requests[0][2]["think"], "low")
        self.assertEqual(b.describe()["think"], "low")

    def test_ollama_no_think_by_default(self):
        s = self.stub(200, {"message": {"content": "hi"}, "prompt_eval_count": 1, "eval_count": 1})
        OllamaBackend("m", base_url=s.url).complete("hello")
        self.assertNotIn("think", s.requests[0][2])

    def test_make_backend_think(self):
        self.assertEqual(make_backend("ollama", "gpt-oss:20b", think="low").think, "low")

    PROPS = {"model_path": "models\\gpt-oss-20b-MXFP4.gguf", "build_info": "b12242-3b4d6ab3f",
             "default_generation_settings": {"n_ctx": 32768, "params": {"temperature": 1.0}}}

    def test_llamacpp_request(self):
        s = self.stub(200, {"choices": [{"message": {"content": "code", "reasoning_content": "hmm"},
                                         "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 9, "completion_tokens": 4}})
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}):
            c = LlamaCppBackend("gpt-oss-20b", base_url=s.url, context_tokens=8192, think="high").complete("x" * 4000)
        self.assertEqual((c.text, c.prompt_tokens, c.response_tokens, c.finish_reason), ("code", 9, 4, "stop"))
        path, headers, body = s.requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertNotIn("Authorization", headers)
        self.assertEqual(body["max_tokens"], 8192 - 1150)
        self.assertEqual(body["chat_template_kwargs"], {"reasoning_effort": "high"})
        self.assertNotIn("temperature", body)

    def test_llamacpp_no_think_by_default(self):
        s = self.stub(200, {"choices": [{"message": {"content": "x"}}]})
        LlamaCppBackend("m", base_url=s.url, temperature=0.5).complete("hi")
        body = s.requests[0][2]
        self.assertNotIn("chat_template_kwargs", body)
        self.assertEqual(body["temperature"], 0.5)

    def test_llamacpp_context_error(self):
        s = self.stub(400, {"error": {"type": "exceed_context_size_error",
                                      "message": "the request exceeds the available context size"}})
        with self.assertRaises(ContextOverflow):
            LlamaCppBackend("m", base_url=s.url).complete("hi")

    def test_llamacpp_describe(self):
        s = self.stub(200, {}, get_body=self.PROPS)
        d = LlamaCppBackend("m", base_url=s.url, think="low").describe()
        self.assertEqual(d, {"temperature": 1.0, "think": "low", "quantization": "MXFP4",
                             "model_digest": "gpt-oss-20b-MXFP4.gguf@b12242-3b4d6ab3f"})
        self.assertEqual(LlamaCppBackend("m", base_url=s.url, temperature=0.2).describe()["temperature"], 0.2)

    def test_llamacpp_verify(self):
        s = self.stub(200, {}, get_body=self.PROPS)
        LlamaCppBackend("m", base_url=s.url, context_tokens=32768).verify()
        with self.assertRaisesRegex(BackendError, "-c 65536"):
            LlamaCppBackend("m", base_url=s.url, context_tokens=65536).verify()

    def test_llamacpp_verify_server_down(self):
        s = self.stub(200, {})
        url = s.url
        s.close()
        with self.assertRaisesRegex(BackendError, "not usable"):
            LlamaCppBackend("m", base_url=url).verify()

    def test_make_backend_llamacpp(self):
        b = make_backend("llamacpp", "gpt-oss-20b", context_tokens=65536, think="medium")
        self.assertEqual((b.name, b.base_url, b.context_tokens, b.think),
                         ("llamacpp", "http://localhost:8080", 65536, "medium"))

    def test_make_backend(self):
        self.assertEqual(make_backend("ollama", "m").model, "m")
        self.assertEqual(make_backend("openai", "gpt", base_url="http://x").name, "openai")
        self.assertEqual(make_backend("claude-cli", "sonnet").name, "claude-cli")
        with self.assertRaises(ValueError):
            make_backend("nope", "m")


if __name__ == "__main__":
    unittest.main()
