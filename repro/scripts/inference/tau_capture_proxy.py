"""
tau_capture_proxy.py — minimal aiohttp transparent proxy that captures
per-request DFlash speculative-decode timings from llama-server (the
fork at github.com/.../llama.cpp-dflash, which exposes
``timings.draft_n`` / ``timings.draft_n_accepted`` on every
``/v1/chat/completions`` response — see repro/04-empirical-tau-llama-benchy.md
§4.3).

We sit between an OAI-API client (e.g. ``llama-benchy``) and the
upstream llama-server, forward every request transparently, and on
``/v1/chat/completions`` parse the response body to extract the
``timings`` block. Both non-streaming ``application/json`` bodies and
streaming ``text/event-stream`` SSE bodies are supported. All other
paths (``/v1/models``, etc.) pass through with a small log line.

Output is one JSONL record per request to ``OUT_JSONL`` (default
``./empirical_tau_traffic.jsonl``).

Env vars:
    UPSTREAM_URL   default http://127.0.0.1:8080
    LISTEN_PORT    default 8081
    OUT_JSONL      default ./empirical_tau_traffic.jsonl

Run::

    python3 tau_capture_proxy.py

Then point the benchmark at ``http://127.0.0.1:8081/v1`` instead of
the upstream ``http://127.0.0.1:8080/v1``.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any, Optional

import aiohttp
from aiohttp import web

UPSTREAM_URL = os.environ.get("UPSTREAM_URL", "http://127.0.0.1:8080").rstrip("/")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8081"))
OUT_JSONL = os.environ.get("OUT_JSONL", "./empirical_tau_traffic.jsonl")

# Hop-by-hop headers we should not forward.
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "content-encoding",
}


def _filter_headers(headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


def _record(rec: dict) -> None:
    rec.setdefault("ts", time.time())
    try:
        with open(OUT_JSONL, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    except OSError as exc:  # pragma: no cover
        print(f"[tau_capture_proxy] WARN: could not write {OUT_JSONL}: {exc}",
              file=sys.stderr)


def _summarize_completion(path: str, status: int, elapsed: float,
                          obj: dict) -> dict:
    timings = obj.get("timings") or {}
    return {
        "path": path,
        "status": status,
        "elapsed_s": elapsed,
        "id": obj.get("id"),
        "prompt_n": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "predicted_n": timings.get("predicted_n"),
        "predicted_ms": timings.get("predicted_ms"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "draft_n": timings.get("draft_n", 0),
        "draft_n_accepted": timings.get("draft_n_accepted", 0),
    }


def _parse_sse_final(buf: str) -> Optional[dict]:
    """
    Walk an SSE stream buffer and return the last JSON ``data:`` payload
    that contains a ``timings`` block. llama-server emits the final
    completion stats on the last non-``[DONE]`` data event.
    """
    last_with_timings: Optional[dict] = None
    for raw_event in buf.split("\n\n"):
        for line in raw_event.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("timings"):
                last_with_timings = obj
    return last_with_timings


async def _proxy(request: web.Request) -> web.StreamResponse:
    path = request.rel_url.path_qs
    method = request.method
    upstream = f"{UPSTREAM_URL}{path}"
    headers = _filter_headers(request.headers)
    body = await request.read()
    t0 = time.time()

    is_completion = request.path.endswith("/chat/completions") or \
        request.path.endswith("/completions")

    timeout = aiohttp.ClientTimeout(total=None, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout, auto_decompress=False) as sess:
        try:
            async with sess.request(method, upstream, headers=headers,
                                    data=body, allow_redirects=False) as up:
                up_ct = up.headers.get("content-type", "").lower()
                resp_headers = _filter_headers(up.headers)
                resp = web.StreamResponse(status=up.status, headers=resp_headers)
                await resp.prepare(request)

                buf_chunks: list[bytes] = []
                is_sse = "text/event-stream" in up_ct
                async for chunk in up.content.iter_any():
                    if is_completion:
                        buf_chunks.append(chunk)
                    await resp.write(chunk)
                await resp.write_eof()

                elapsed = time.time() - t0
                if is_completion:
                    body_bytes = b"".join(buf_chunks)
                    parsed: Optional[dict] = None
                    if is_sse:
                        try:
                            parsed = _parse_sse_final(body_bytes.decode(
                                "utf-8", errors="replace"))
                        except Exception as exc:  # pragma: no cover
                            _record({"path": request.path, "status": up.status,
                                     "parse_error": f"sse: {exc!r}"})
                    else:
                        try:
                            parsed = json.loads(body_bytes.decode(
                                "utf-8", errors="replace"))
                        except json.JSONDecodeError as exc:
                            _record({"path": request.path, "status": up.status,
                                     "parse_error": repr(exc)})
                    if parsed is not None:
                        _record(_summarize_completion(
                            request.path, up.status, elapsed, parsed))
                else:
                    _record({"path": request.path, "status": up.status,
                             "elapsed_s": elapsed, "draft_n": 0,
                             "draft_n_accepted": 0,
                             "predicted_n": None, "predicted_ms": None,
                             "predicted_per_second": None,
                             "prompt_n": None, "prompt_ms": None,
                             "prompt_per_second": None, "id": None})
                return resp
        except aiohttp.ClientError as exc:
            _record({"path": request.path, "status": 502,
                     "proxy_error": repr(exc)})
            return web.Response(status=502, text=f"upstream error: {exc!r}")


def main() -> None:
    print(f"[tau_capture_proxy] upstream={UPSTREAM_URL} "
          f"listen=127.0.0.1:{LISTEN_PORT} out={OUT_JSONL}", file=sys.stderr)
    app = web.Application(client_max_size=1024 * 1024 * 64)
    app.router.add_route("*", "/{tail:.*}", _proxy)
    web.run_app(app, host="127.0.0.1", port=LISTEN_PORT, access_log=None)


if __name__ == "__main__":
    main()
