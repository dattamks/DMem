"""OpenAI-compatible proxy adapter — offline (injected forward_fn, no network)."""

import asyncio
import json

from dmem.adapters.proxy import (MemoryChatProxy, ProxyConfig, create_app,
                                 _insert_after_system, _last_user_text)


def _canned(content="ok"):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _proxy(engine, capture=None, **cfg):
    def forward(payload):
        if capture is not None:
            capture["payload"] = payload
        return _canned("You work at Analytical Engines.")
    return MemoryChatProxy(engine, ProxyConfig(**cfg), forward_fn=forward)


# -- helpers ----------------------------------------------------------------

def test_last_user_text_picks_latest():
    msgs = [{"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"}]
    assert _last_user_text(msgs) == "second"


def test_insert_after_system_positions():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    out = _insert_after_system(msgs, {"role": "system", "content": "mem"})
    assert out[1]["content"] == "mem"  # after existing system, before user
    out2 = _insert_after_system([{"role": "user", "content": "u"}],
                                {"role": "system", "content": "mem"})
    assert out2[0]["content"] == "mem"  # front when no system


# -- augmentation -----------------------------------------------------------

def test_augment_injects_relevant_memory(engine):
    engine.ingest_message("My name is Ada. I work at Analytical Engines.",
                          namespace="u1")
    proxy = _proxy(engine)
    payload = {"model": "gpt", "messages": [
        {"role": "user", "content": "where do I work?"}]}
    aug, ctx = proxy.augment_request(payload, "u1")
    injected = [m for m in aug["messages"] if m["role"] == "system"
                and "analytical engines" in m["content"].lower()]
    assert injected, "relevant memory should be injected as a system message"
    assert ctx["model_switched"] is False


def test_augment_no_memory_no_injection(engine):
    proxy = _proxy(engine)
    payload = {"model": "gpt", "messages": [
        {"role": "user", "content": "totally unknown topic zzz"}]}
    aug, _ = proxy.augment_request(payload, "empty_ns")
    assert all(m["role"] != "system" for m in aug["messages"])


def test_model_switch_triggers_handoff(engine):
    engine.ingest_message("My name is Grace. I use Rust.", namespace="u2")
    proxy = _proxy(engine)
    p1 = {"model": "model-a", "messages": [{"role": "user", "content": "hi"}]}
    proxy.complete(p1, "u2")  # records last model = model-a

    p2 = {"model": "model-b", "messages": [
        {"role": "user", "content": "remind me who I am"}]}
    aug, ctx = proxy.augment_request(p2, "u2")
    assert ctx["model_switched"] is True
    assert any("switched models" in m.get("content", "").lower()
               for m in aug["messages"] if m["role"] == "system")


def test_credentials_not_injected(engine):
    engine.ingest_message("I use api key sk-secret-proxy-1.", namespace="u3")
    proxy = _proxy(engine)
    payload = {"model": "gpt", "messages": [
        {"role": "user", "content": "what do I use?"}]}
    aug, _ = proxy.augment_request(payload, "u3")
    blob = json.dumps(aug)
    assert "sk-secret-proxy-1" not in blob
    assert "[REDACTED]" not in blob


# -- compaction -------------------------------------------------------------

def test_long_history_is_compacted(engine):
    capture = {}
    proxy = _proxy(engine, capture=capture, compact_over_tokens=40,
                   keep_last_turns=2, inject_memory=False)
    long_msgs = []
    for i in range(12):
        long_msgs.append({"role": "user",
                          "content": f"This is a fairly long message number {i} "
                                     f"with enough words to add up token cost."})
        long_msgs.append({"role": "assistant", "content": f"Acknowledged {i}."})
    payload = {"model": "gpt", "messages": long_msgs}
    proxy.complete(payload, "u4")
    fwd = capture["payload"]["messages"]
    assert len(fwd) < len(long_msgs)  # compacted
    assert any("summary of earlier conversation" in m.get("content", "").lower()
               for m in fwd if m["role"] == "system")


def test_short_history_not_compacted(engine):
    capture = {}
    proxy = _proxy(engine, capture=capture, compact_over_tokens=100000,
                   inject_memory=False)
    msgs = [{"role": "user", "content": "hello"}]
    proxy.complete({"model": "gpt", "messages": msgs}, "u5")
    assert capture["payload"]["messages"] == msgs


# -- record + orchestration -------------------------------------------------

def test_complete_returns_response_unchanged_and_records(engine):
    proxy = _proxy(engine)
    payload = {"model": "gpt", "messages": [
        {"role": "user", "content": "My name is Bob and I use Kotlin."}]}
    resp = proxy.complete(payload, "u6")
    assert resp["choices"][0]["message"]["content"].startswith("You work at")
    # user turn was ingested and is retrievable afterward
    results = engine.retrieve("what language does the user use?", namespace="u6")
    assert any("kotlin" in r.text.lower() for r in results)


def test_record_ingests_assistant_reply(engine):
    def forward(_):
        return _canned("The user's project is named DarkMatter.")
    proxy = MemoryChatProxy(engine, ProxyConfig(inject_memory=False,
                                                compact_history=False),
                            forward_fn=forward)
    proxy.complete({"model": "gpt", "messages": [
        {"role": "user", "content": "hi"}]}, "u7")
    results = engine.retrieve("project name", namespace="u7")
    assert any("darkmatter" in r.text.lower() for r in results)


# -- ASGI binding -----------------------------------------------------------

def _call_asgi(app, method, path, body: bytes, headers: dict):
    scope = {"type": "http", "method": method, "path": path,
             "headers": [[k.encode(), v.encode()] for k, v in headers.items()]}
    sent = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(ev):
        sent.append(ev)

    asyncio.run(app(scope, receive, send))
    status = next(e["status"] for e in sent if e["type"] == "http.response.start")
    out = b"".join(e.get("body", b"") for e in sent
                   if e["type"] == "http.response.body")
    return status, json.loads(out or b"{}")


def test_asgi_proxies_chat_completions(engine):
    proxy = _proxy(engine)
    app = create_app(proxy=proxy)
    body = json.dumps({"model": "gpt", "messages": [
        {"role": "user", "content": "My name is Ada."}]}).encode()
    status, resp = _call_asgi(app, "POST", "/v1/chat/completions", body,
                              {"x-dmem-namespace": "asgi_ns"})
    assert status == 200
    assert resp["choices"][0]["message"]["content"]
    # namespace from header was used for memory writes
    assert any("ada" in r.text.lower()
               for r in engine.retrieve("name", namespace="asgi_ns"))


def test_asgi_404_for_other_paths(engine):
    app = create_app(proxy=_proxy(engine))
    status, _ = _call_asgi(app, "GET", "/healthz", b"", {})
    assert status == 404


def test_asgi_bad_json_400(engine):
    app = create_app(proxy=_proxy(engine))
    status, resp = _call_asgi(app, "POST", "/v1/chat/completions",
                              b"{not json", {})
    assert status == 400
