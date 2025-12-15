from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
import uvicorn


APP_NAME = "Sat.AI MCP Test Server"
APP_VERSION = "0.1.0"
PROTOCOL_VERSION = "2024-11-05"


def _sse_event(event: str, data: str) -> bytes:
    lines = (data or "").splitlines() or [""]
    payload = "".join(f"data: {line}\n" for line in lines)
    return f"event: {event}\n{payload}\n".encode("utf-8")


def _tool_text(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": str(text)}]}


def _float_arg(args: dict[str, Any], key: str) -> float:
    if key not in args:
        raise ValueError(f"Missing required argument: {key}")
    try:
        value = float(args[key])
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid float for {key}") from exc
    return value


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_km * math.asin(min(1.0, math.sqrt(a)))


TOOLS: list[dict[str, Any]] = [
    {
        "name": "echo",
        "description": "Echo back a text string.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to echo."}},
            "required": ["text"],
        },
    },
    {
        "name": "haversine_km",
        "description": "Compute great-circle distance (km) between two latitude/longitude points.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "lat1": {"type": "number", "description": "Latitude of point 1 (degrees)."},
                "lon1": {"type": "number", "description": "Longitude of point 1 (degrees)."},
                "lat2": {"type": "number", "description": "Latitude of point 2 (degrees)."},
                "lon2": {"type": "number", "description": "Longitude of point 2 (degrees)."},
            },
            "required": ["lat1", "lon1", "lat2", "lon2"],
        },
    },
    {
        "name": "server_time",
        "description": "Return the server's current local time.",
        "inputSchema": {
            "type": "object",
            "properties": {"format": {"type": "string", "description": "strftime format string (optional)."}},
            "required": [],
        },
    },
]


@dataclass
class _Session:
    id: str
    queue: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)
    closed: asyncio.Event = field(default_factory=asyncio.Event)
    created_s: float = field(default_factory=lambda: time.time())


_SESSIONS: dict[str, _Session] = {}
_SESSIONS_LOCK = asyncio.Lock()


class _JsonRpcError(RuntimeError):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = int(code)
        self.message = str(message)


def _jsonrpc_response(msg_id: Any, *, result: Any | None = None, error: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result
    return payload


def _jsonrpc_error(code: int, message: str) -> dict[str, Any]:
    return {"code": int(code), "message": str(message)}


def _tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    tool = (name or "").strip()
    args = arguments if isinstance(arguments, dict) else {}

    if tool == "echo":
        text = args.get("text")
        if not isinstance(text, str):
            raise _JsonRpcError(-32602, "echo requires {text: string}")
        return _tool_text(text)

    if tool == "haversine_km":
        try:
            lat1 = _float_arg(args, "lat1")
            lon1 = _float_arg(args, "lon1")
            lat2 = _float_arg(args, "lat2")
            lon2 = _float_arg(args, "lon2")
        except ValueError as exc:
            raise _JsonRpcError(-32602, str(exc)) from exc
        distance_km = _haversine_km(lat1, lon1, lat2, lon2)
        return _tool_text(f"Distance: {distance_km:.3f} km")

    if tool == "server_time":
        fmt = args.get("format")
        if fmt is not None and not isinstance(fmt, str):
            raise _JsonRpcError(-32602, "server_time format must be a string when provided")
        if fmt:
            return _tool_text(time.strftime(fmt, time.localtime()))
        return _tool_text(time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()))

    raise _JsonRpcError(-32601, f"Unknown tool: {tool}")


def _initialize_result() -> dict[str, Any]:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "serverInfo": {"name": APP_NAME, "version": APP_VERSION},
        "capabilities": {"tools": {"listChanged": False}},
    }


async def _handle_jsonrpc(session: _Session, message: dict[str, Any]) -> dict[str, Any] | None:
    msg_id = message.get("id")
    method = message.get("method")
    params = message.get("params")
    params_obj = params if isinstance(params, dict) else {}

    if not isinstance(method, str) or not method.strip():
        if msg_id is None:
            return None
        return _jsonrpc_response(msg_id, error=_jsonrpc_error(-32600, "Invalid Request"))

    method = method.strip()

    if msg_id is None:
        if method == "exit":
            session.closed.set()
        return None

    try:
        if method == "initialize":
            return _jsonrpc_response(msg_id, result=_initialize_result())
        if method == "tools/list":
            return _jsonrpc_response(msg_id, result={"tools": TOOLS})
        if method == "tools/call":
            tool_name = params_obj.get("name")
            arguments = params_obj.get("arguments")
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise _JsonRpcError(-32602, "tools/call requires params.name")
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                raise _JsonRpcError(-32602, "tools/call params.arguments must be an object")
            return _jsonrpc_response(msg_id, result=_tool_call(tool_name, arguments))
        if method == "shutdown":
            return _jsonrpc_response(msg_id, result={"ok": True})
        return _jsonrpc_response(msg_id, error=_jsonrpc_error(-32601, f"Method not found: {method}"))
    except _JsonRpcError as exc:
        return _jsonrpc_response(msg_id, error=_jsonrpc_error(exc.code, exc.message))
    except Exception as exc:  # noqa: BLE001
        return _jsonrpc_response(msg_id, error=_jsonrpc_error(-32000, f"Server error: {exc}"))


app = FastAPI(title=APP_NAME, version=APP_VERSION)


@app.get("/")
def root() -> dict[str, Any]:
    return {"name": APP_NAME, "version": APP_VERSION, "sse": "/sse"}


@app.get("/sse")
async def sse(request: Request):
    session = _Session(id=uuid.uuid4().hex)
    async with _SESSIONS_LOCK:
        _SESSIONS[session.id] = session

    endpoint = f"/message/{session.id}"

    async def stream() -> AsyncIterator[bytes]:
        try:
            yield _sse_event("endpoint", endpoint)
            keepalive_s = 15.0
            while not session.closed.is_set():
                if await request.is_disconnected():
                    session.closed.set()
                    return
                try:
                    response = await asyncio.wait_for(session.queue.get(), timeout=keepalive_s)
                except asyncio.TimeoutError:
                    yield b": keep-alive\n\n"
                    continue
                yield _sse_event("message", json.dumps(response, ensure_ascii=False))
        finally:
            async with _SESSIONS_LOCK:
                _SESSIONS.pop(session.id, None)

    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
    return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)


@app.post("/message/{session_id}")
async def post_message(session_id: str, request: Request) -> Response:
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON-RPC payload must be an object.")

    async with _SESSIONS_LOCK:
        session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session id.")

    response_payload = await _handle_jsonrpc(session, payload)
    if response_payload is not None:
        await session.queue.put(response_payload)

    return Response(status_code=202)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sat.AI remote MCP HTTP test server (SSE transport).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    uvicorn.run(app, host=args.host, port=int(args.port), log_level=str(args.log_level))


if __name__ == "__main__":
    main()

