"""Swappable LLM backends: prompt in, text out. Every call is a fresh, history-free request."""

from dataclasses import dataclass
import json
import math
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

RESERVE_OUTPUT = 4096
MAX_OUTPUT_TOKENS = 16384
DEFAULT_CONTEXT = 32768


class BackendError(Exception):
    pass


class RetryableError(BackendError):
    pass


class ContextOverflow(BackendError):
    pass


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int | None
    response_tokens: int | None
    finish_reason: str | None = None  # "length" = the reply was cut off


def estimate_tokens(text):
    return math.ceil(len(text) / 4)


def fits(prompt, context_tokens):
    return math.ceil(estimate_tokens(prompt) * 1.15) + RESERVE_OUTPUT <= context_tokens


def with_retries(fn, attempts=6, base_delay=10.0, max_delay=900.0, sleep=time.sleep):
    for k in range(attempts):
        try:
            return fn()
        except RetryableError:
            if k == attempts - 1:
                raise
            sleep(min(base_delay * 2 ** k, max_delay))


def _post_json(url, payload, headers=None, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise BackendError(f"non-JSON reply from {url}: {raw[:300]}") from e
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        if e.code == 429 or e.code >= 500:
            raise RetryableError(f"HTTP {e.code}: {body[:500]}") from e
        if e.code == 400 and "context" in body.lower():
            raise ContextOverflow(f"HTTP 400: {body[:500]}") from e
        raise BackendError(f"HTTP {e.code}: {body[:500]}") from e
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        raise RetryableError(f"cannot reach {url}: {e}") from e


class FakeBackend:
    name = "fake"

    def __init__(self, replies, model="fake", context_tokens=DEFAULT_CONTEXT):
        self.replies, self.model, self.context_tokens = replies, model, context_tokens
        self.calls = []

    def complete(self, prompt):
        self.calls.append(prompt)
        if callable(self.replies):
            text = self.replies(prompt)
        else:
            text = self.replies[(len(self.calls) - 1) % len(self.replies)]
        return text if isinstance(text, Completion) else Completion(text, None, None)

    def describe(self):
        return {}


class OllamaBackend:
    name = "ollama"

    def __init__(self, model, base_url="http://localhost:11434", context_tokens=DEFAULT_CONTEXT,
                 temperature=None, timeout=600, think=None):
        self.model, self.base_url, self.think = model, base_url.rstrip("/"), think
        self.context_tokens, self.temperature, self.timeout = context_tokens, temperature, timeout

    def describe(self):
        info = {"temperature": self.temperature, "think": self.think}
        try:
            data = _post_json(f"{self.base_url}/api/show", {"model": self.model}, timeout=30)
        except BackendError:
            return info
        info["quantization"] = (data.get("details") or {}).get("quantization_level")
        info["model_digest"] = data.get("digest") or data.get("modified_at")
        return info

    def complete(self, prompt):
        # Cap the reply so prompt + reply stays inside num_ctx (Ollama silently
        # drops prompt tokens otherwise) and a looping model stops early.
        room = self.context_tokens - math.ceil(estimate_tokens(prompt) * 1.15)
        options = {"num_ctx": self.context_tokens, "num_predict": max(256, min(MAX_OUTPUT_TOKENS, room))}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        payload = {"model": self.model, "stream": False, "options": options,
                   "messages": [{"role": "user", "content": prompt}]}
        if self.think is not None:  # reasoning effort for thinking models, e.g. gpt-oss: low|medium|high
            payload["think"] = self.think
        data = _post_json(f"{self.base_url}/api/chat", payload, timeout=self.timeout)
        used = data.get("prompt_eval_count")
        # Ollama truncates an oversized prompt silently; a full window means it did.
        if used is not None and used >= self.context_tokens:
            raise ContextOverflow(f"prompt filled the {self.context_tokens}-token window")
        try:
            text = (data["message"].get("content") or "")
        except (KeyError, AttributeError, TypeError) as e:
            raise BackendError(f"unexpected Ollama reply: {str(data)[:300]}") from e
        return Completion(text, used, data.get("eval_count"), data.get("done_reason"))


class OpenAICompatBackend:
    """OpenAI, DeepSeek (https://api.deepseek.com, DEEPSEEK_API_KEY), LM Studio, llama.cpp, vLLM."""

    name = "openai"

    def __init__(self, model, base_url, api_key_env="OPENAI_API_KEY", context_tokens=DEFAULT_CONTEXT,
                 temperature=None, timeout=600):
        self.model, self.base_url, self.api_key_env = model, base_url.rstrip("/"), api_key_env
        self.context_tokens, self.temperature, self.timeout = context_tokens, temperature, timeout

    def complete(self, prompt):
        headers = {}
        key = os.environ.get(self.api_key_env) if self.api_key_env else None
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        data = _post_json(f"{self.base_url}/chat/completions", payload, headers, self.timeout)
        try:
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, AttributeError, TypeError) as e:
            raise BackendError(f"unexpected API reply: {str(data)[:300]}") from e
        usage = data.get("usage") or {}
        return Completion(text, usage.get("prompt_tokens"), usage.get("completion_tokens"),
                          choice.get("finish_reason"))

    def describe(self):
        return {"temperature": self.temperature}


class ClaudeCliBackend:
    """`claude -p` on the user's subscription, one fresh process per call.

    Isolation flags checked against `claude --help` on 2026-11-10:
      --tools ""                 no tools: text in, text out
      --setting-sources ""       ignore user/project/local settings files
      --strict-mcp-config        no MCP servers
      --disable-slash-commands   no skills or commands
      --no-session-persistence   nothing saved, nothing resumable
    cwd is a fresh empty temp directory, so no project CLAUDE.md is found.
    `--bare` would also skip CLAUDE.md and auto-memory, but it refuses OAuth
    (subscription) logins, so it is not used. A user-level ~/.claude/CLAUDE.md
    on the run machine may still load; run under an account without one.
    """

    name = "claude-cli"

    def __init__(self, model="sonnet", context_tokens=200000, timeout=900, executable="claude"):
        self.model, self.context_tokens, self.timeout, self.executable = model, context_tokens, timeout, executable

    def complete(self, prompt):
        cmd = [self.executable, "-p", "--output-format", "json", "--model", self.model,
               "--tools", "", "--setting-sources", "", "--strict-mcp-config",
               "--disable-slash-commands", "--no-session-persistence"]
        env = {**os.environ, "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
        with tempfile.TemporaryDirectory() as cwd:
            try:
                p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", cwd=cwd, env=env, timeout=self.timeout)
            except subprocess.TimeoutExpired as e:
                raise RetryableError(f"claude -p timed out after {self.timeout}s") from e
            except FileNotFoundError as e:
                raise BackendError("claude executable not found") from e
        out = (p.stdout or "") + (p.stderr or "")
        if p.returncode != 0:
            low = out.lower()
            if "too long" in low or "context" in low:
                raise ContextOverflow(out[:500])
            if "limit" in low or "rate" in low or "overloaded" in low:
                raise RetryableError(out[:500])
            raise BackendError(out[:500])
        try:
            data = json.loads(p.stdout)
        except json.JSONDecodeError as e:
            raise BackendError(f"unparseable claude output: {p.stdout[:300]}") from e
        if data.get("is_error"):
            raise RetryableError(str(data.get("result"))[:500])
        usage = data.get("usage") or {}
        prompt_tokens = sum(usage.get(k) or 0 for k in
                            ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")) or None
        stop = "length" if data.get("stop_reason") == "max_tokens" else data.get("stop_reason")
        return Completion(data.get("result") or "", prompt_tokens, usage.get("output_tokens"), stop)

    def describe(self):
        return {}


def make_backend(name, model, base_url=None, context_tokens=None, temperature=None, think=None):
    ctx = {} if context_tokens is None else {"context_tokens": context_tokens}
    if name == "ollama":
        return OllamaBackend(model, base_url or "http://localhost:11434", temperature=temperature,
                             think=think, **ctx)
    if name == "openai":
        if not base_url:
            base_url = "https://api.openai.com/v1"
        key_env = "DEEPSEEK_API_KEY" if "deepseek" in base_url else "OPENAI_API_KEY"
        return OpenAICompatBackend(model, base_url, key_env, temperature=temperature, **ctx)
    if name == "claude-cli":
        return ClaudeCliBackend(model, **ctx)
    if name == "fake":
        return FakeBackend(["(fake backend: no reply configured)"], model, **ctx)
    raise ValueError(f"unknown backend {name!r}; choose ollama, openai, claude-cli or fake")
