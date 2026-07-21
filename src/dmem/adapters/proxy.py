"""OpenAI-compatible proxy adapter (the "thin proxy").

Sits in front of any OpenAI-compatible ``/v1/chat/completions`` endpoint and adds
memory *transparently* — the host app only changes its base URL, no code change:

    request ──► [inject memory + compact history] ──► upstream LLM
    response ◄─ [persist facts from the exchange] ◄──┘

Two token-saving mechanisms, both optional:
- **Memory injection** — relevant remembered facts / a model-switch handoff are
  added as a system message, so the client needn't resend that context itself.
- **History compaction** — when the forwarded message list is long, older turns
  are replaced with a compact summary before hitting the upstream, cutting the
  tokens actually sent.

The orchestration core (`MemoryChatProxy`) is pure and synchronous with an
injectable ``forward_fn``, so it is fully unit-testable with no network. A
minimal, framework-free ASGI app (`create_app`) binds it to HTTP; run it with any
ASGI server (e.g. ``uvicorn dmem.adapters.proxy:app``). Streaming responses are
passed through un-augmented in v1 (documented limitation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..config import Config
from ..engine import DMemEngine

_META_LAST_MODEL = "proxy_last_model:"  # + namespace


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _content_str(msg: dict) -> str:
    """Best-effort text of a message (OpenAI content may be str or parts list)."""
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):  # multimodal parts
        return " ".join(p.get("text", "") for p in c
                        if isinstance(p, dict) and p.get("type") == "text")
    return ""


@dataclass
class ProxyConfig:
    upstream_base_url: Optional[str] = None       # BYO OpenAI-compatible endpoint
    upstream_api_key: Optional[str] = None
    inject_memory: bool = True
    compact_history: bool = True
    compact_over_tokens: int = 3000               # compact when history exceeds this
    keep_last_turns: int = 6                       # turns kept verbatim after compaction
    namespace_header: str = "x-dmem-namespace"
    memory_token_budget: int = 800

    @classmethod
    def from_env(cls, env: Optional[dict] = None) -> "ProxyConfig":
        import os
        env = dict(os.environ if env is None else env)
        def b(k, d):
            v = env.get(k)
            return d if v is None else v.strip().lower() not in ("0", "false", "no")
        return cls(
            upstream_base_url=env.get("PROXY_UPSTREAM_URL") or env.get("OPENAI_BASE_URL"),
            upstream_api_key=env.get("PROXY_UPSTREAM_API_KEY") or env.get("OPENAI_API_KEY"),
            inject_memory=b("PROXY_INJECT_MEMORY", True),
            compact_history=b("PROXY_COMPACT_HISTORY", True),
            compact_over_tokens=int(env.get("PROXY_COMPACT_OVER_TOKENS", "3000")),
            keep_last_turns=int(env.get("PROXY_KEEP_LAST_TURNS", "6")),
            memory_token_budget=int(env.get("PROXY_MEMORY_TOKEN_BUDGET", "800")),
        )


class MemoryChatProxy:
    """Pure, synchronous orchestration around an OpenAI chat request."""

    def __init__(self, engine: DMemEngine, config: Optional[ProxyConfig] = None,
                 forward_fn: Optional[Callable[[dict], dict]] = None):
        self.engine = engine
        self.config = config or ProxyConfig()
        self._forward_fn = forward_fn  # injectable for tests; else httpx

    # -- public orchestration ---------------------------------------------
    def complete(self, payload: dict, namespace: str) -> dict:
        """Augment → forward → record. Returns the upstream response unchanged."""
        augmented, ctx = self.augment_request(payload, namespace)
        response = self.forward(augmented)
        try:
            self.record_exchange(payload, response, namespace)
        except Exception:
            pass  # memory writes must never break the proxied call
        return response

    # -- step 1: augment ---------------------------------------------------
    def augment_request(self, payload: dict, namespace: str) -> tuple[dict, dict]:
        messages = list(payload.get("messages", []))
        model = payload.get("model")
        switched = self._model_switched(namespace, model)

        if self.config.compact_history:
            messages = self._maybe_compact(messages, namespace)

        if self.config.inject_memory:
            sysmsg = self._memory_system_message(namespace, messages, switched)
            if sysmsg is not None:
                messages = _insert_after_system(messages, sysmsg)

        new_payload = dict(payload)
        new_payload["messages"] = messages
        return new_payload, {"model_switched": switched}

    def _memory_system_message(self, namespace: str, messages: list[dict],
                               switched: bool) -> Optional[dict]:
        query = _last_user_text(messages)
        if not query:
            return None
        if switched:
            handoff = self.engine.handoff(
                query, namespace=namespace,
                token_budget=self.config.memory_token_budget)
            if handoff.low_confidence and not handoff.included:
                return None
            body = handoff.text
            label = "You just switched models. " + \
                    "Compact context handoff from prior conversation:"
        else:
            results = self.engine.retrieve(query, namespace=namespace)
            if not results:
                return None
            lines = [f"- {r.text}" for r in results
                     if r.payload.get("fact_type") != "credential"]
            if not lines:
                return None
            body = "\n".join(lines)
            label = "Relevant remembered context (may help answer):"
        return {"role": "system", "content": f"{label}\n{body}"}

    # -- step 2: forward ---------------------------------------------------
    def forward(self, payload: dict) -> dict:
        if self._forward_fn is not None:
            return self._forward_fn(payload)
        cfg = self.config
        if not cfg.upstream_base_url:
            raise RuntimeError("No upstream configured: set PROXY_UPSTREAM_URL "
                               "(or pass forward_fn).")
        try:
            import httpx
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("The proxy forwarder needs the 'http' extra: "
                               "pip install dmem[http]") from e
        url = cfg.upstream_base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if cfg.upstream_api_key:
            headers["Authorization"] = f"Bearer {cfg.upstream_api_key}"
        resp = httpx.post(url, headers=headers, json=payload, timeout=300.0)
        resp.raise_for_status()
        return resp.json()

    # -- step 3: record ----------------------------------------------------
    def record_exchange(self, payload: dict, response: dict, namespace: str) -> None:
        # persist the newest user turn and the assistant reply as memory
        last_user = _last_user_text(payload.get("messages", []))
        if last_user:
            self.engine.ingest_message(last_user, namespace=namespace, role="user")
        for choice in response.get("choices", []):
            content = _content_str(choice.get("message", {}))
            if content:
                self.engine.ingest_message(content, namespace=namespace,
                                           role="assistant", speaker="assistant")
        model = payload.get("model")
        if model:
            self._set_last_model(namespace, model)

    # -- helpers -----------------------------------------------------------
    def _maybe_compact(self, messages: list[dict], namespace: str) -> list[dict]:
        total = sum(_approx_tokens(_content_str(m)) for m in messages)
        if total <= self.config.compact_over_tokens:
            return messages
        systems = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]
        keep = self.config.keep_last_turns
        if len(non_system) <= keep:
            return messages
        to_compact = non_system[:-keep]
        tail = non_system[-keep:]
        transcript = [{"role": m.get("role", "user"),
                       "content": _content_str(m)} for m in to_compact]
        summary = self.engine.compact(
            transcript, target_tokens=self.config.memory_token_budget,
            namespace=namespace)
        summary_msg = {"role": "system",
                       "content": f"Summary of earlier conversation:\n{summary}"}
        return systems + [summary_msg] + tail

    def _model_switched(self, namespace: str, model: Optional[str]) -> bool:
        if not model:
            return False
        prev = self._get_last_model(namespace)
        return prev is not None and prev != model

    def _get_last_model(self, namespace: str) -> Optional[str]:
        try:
            return self.engine.store.get_meta(_META_LAST_MODEL + namespace)
        except Exception:
            return None

    def _set_last_model(self, namespace: str, model: str) -> None:
        try:
            self.engine.store.set_meta(_META_LAST_MODEL + namespace, model)
        except Exception:
            pass


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return _content_str(m)
    return ""


def _insert_after_system(messages: list[dict], msg: dict) -> list[dict]:
    idx = 0
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            idx = i + 1
        else:
            break
    return messages[:idx] + [msg] + messages[idx:]


# --------------------------------------------------------------------------- #
# Minimal framework-free ASGI binding
# --------------------------------------------------------------------------- #

def create_app(engine: Optional[DMemEngine] = None,
               config: Optional[ProxyConfig] = None,
               proxy: Optional[MemoryChatProxy] = None):
    """Return a minimal ASGI app that proxies POST /v1/chat/completions.

    Namespace is resolved from the configured header, else the request's ``user``
    field, else the engine default. No framework dependency; run with any ASGI
    server (uvicorn, hypercorn, ...)."""
    import asyncio

    if proxy is None:
        eng = engine or DMemEngine(None)
        proxy = MemoryChatProxy(eng, config or ProxyConfig.from_env())
    ns_header = (config or proxy.config).namespace_header.lower()
    default_ns = proxy.engine.config.default_namespace

    async def app(scope, receive, send):
        if scope["type"] != "http":  # pragma: no cover
            return
        path = scope.get("path", "")
        method = scope.get("method", "GET")
        if method != "POST" or not path.endswith("/chat/completions"):
            await _send_json(send, 404, {"error": "not found"})
            return

        body = await _read_body(receive)
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            await _send_json(send, 400, {"error": "invalid JSON"})
            return

        headers = {k.decode().lower(): v.decode()
                   for k, v in scope.get("headers", [])}
        namespace = (headers.get(ns_header) or payload.get("user")
                     or default_ns)

        if payload.get("stream"):
            # streaming is passed through un-augmented in v1
            try:
                result = await asyncio.to_thread(proxy.forward, payload)
                await _send_json(send, 200, result)
            except Exception as e:
                await _send_json(send, 502, {"error": str(e)})
            return

        try:
            result = await asyncio.to_thread(proxy.complete, payload, namespace)
            await _send_json(send, 200, result)
        except Exception as e:
            await _send_json(send, 502, {"error": str(e)})

    return app


async def _read_body(receive) -> bytes:
    body = b""
    while True:
        event = await receive()
        body += event.get("body", b"")
        if not event.get("more_body"):
            break
    return body


async def _send_json(send, status: int, obj: dict) -> None:
    data = json.dumps(obj).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [[b"content-type", b"application/json"]]})
    await send({"type": "http.response.body", "body": data})


# Module-level app for `uvicorn dmem.adapters.proxy:app`
_app_singleton = None


def app(scope, receive, send):  # pragma: no cover - server entry
    global _app_singleton
    if _app_singleton is None:
        _app_singleton = create_app()
    return _app_singleton(scope, receive, send)
