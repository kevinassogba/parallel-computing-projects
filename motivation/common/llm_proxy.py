"""Harness-side LLM proxy: forwards OpenAI-compatible requests to the remote LLM
and logs, per request, the timing a tool platform could observe without any help
from the provider (send time, first byte, last byte, token usage, tool calls).

Run on the harness host; point the harness's base URL at this proxy.

    python llm_proxy.py --upstream http://LLM_HOST:8000 --port 8001 \
        --log results/m1/llm_requests.jsonl --session-file /tmp/agentfn_session

The session id of each record comes from the request header X-Agentfn-Session
or, if absent, from the first line of --session-file (run scripts rewrite it
before each session; sessions run one at a time per proxy).
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
from aiohttp import web

HOP_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "accept-encoding"}


def session_id(request: web.Request, session_file: Path | None) -> str:
    sid = request.headers.get("X-Agentfn-Session")
    if sid:
        return sid
    if session_file and session_file.exists():
        return session_file.read_text().strip().splitlines()[0] if session_file.read_text().strip() else ""
    return ""


def summarize_body(body: bytes) -> dict:
    try:
        req = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return {}
    return {
        "model": req.get("model"),
        "stream": bool(req.get("stream")),
        "n_messages": len(req.get("messages", [])),
        "n_tools_defined": len(req.get("tools", []) or []),
    }


def parse_response(payload: bytes, streamed: bool) -> dict:
    """Extract usage, finish reason and tool-call names from a full or SSE body."""
    out = {"usage": None, "finish_reason": None, "tool_calls": []}
    chunks = []
    if streamed:
        for line in payload.decode("utf-8", errors="replace").splitlines():
            if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                try:
                    chunks.append(json.loads(line[5:]))
                except ValueError:
                    pass
    else:
        try:
            chunks.append(json.loads(payload))
        except ValueError:
            return out
    names = {}
    for c in chunks:
        if c.get("usage"):
            out["usage"] = c["usage"]
        for ch in c.get("choices", []) or []:
            if ch.get("finish_reason"):
                out["finish_reason"] = ch["finish_reason"]
            msg = ch.get("delta") or ch.get("message") or {}
            for tc in msg.get("tool_calls", []) or []:
                idx = tc.get("index", len(names))
                fn = (tc.get("function") or {}).get("name")
                if fn:
                    names[idx] = fn
    out["tool_calls"] = [names[k] for k in sorted(names)]
    return out


async def handle(request: web.Request) -> web.StreamResponse:
    app = request.app
    body = await request.read()
    t_send = time.time()
    url = app["upstream"] + request.rel_url.path_qs
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
    record = {"session": session_id(request, app["session_file"]), "path": request.path,
              "t_send": t_send, **summarize_body(body)}
    async with app["client"].request(request.method, url, data=body, headers=headers) as up:
        resp = web.StreamResponse(status=up.status)
        for k, v in up.headers.items():
            if k.lower() not in HOP_HEADERS:
                resp.headers[k] = v
        await resp.prepare(request)
        buf = bytearray()
        t_first = None
        async for data in up.content.iter_any():
            if t_first is None:
                t_first = time.time()
            buf.extend(data)
            await resp.write(data)
        await resp.write_eof()
    t_end = time.time()
    streamed = "text/event-stream" in up.headers.get("Content-Type", "")
    record.update({"status": up.status, "t_first_byte": t_first, "t_end": t_end,
                   "latency_s": t_end - t_send,
                   "ttfb_s": (t_first - t_send) if t_first else None,
                   **parse_response(bytes(buf), streamed)})
    async with app["lock"]:
        with open(app["log"], "a") as f:
            f.write(json.dumps(record) + "\n")
    return resp


async def make_app(args) -> web.Application:
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app["upstream"] = args.upstream.rstrip("/")
    app["log"] = Path(args.log)
    app["log"].parent.mkdir(parents=True, exist_ok=True)
    app["session_file"] = Path(args.session_file) if args.session_file else None
    app["lock"] = asyncio.Lock()
    timeout = aiohttp.ClientTimeout(total=None, sock_read=args.read_timeout)
    app["client"] = aiohttp.ClientSession(timeout=timeout, auto_decompress=False)

    async def close(app):
        await app["client"].close()

    app.on_cleanup.append(close)
    app.router.add_route("*", "/{tail:.*}", handle)
    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", required=True, help="e.g. http://llm-host:8000")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--log", required=True)
    p.add_argument("--session-file", default="")
    p.add_argument("--read-timeout", type=float, default=900.0)
    args = p.parse_args()
    web.run_app(make_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
