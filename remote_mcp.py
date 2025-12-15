from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, TextIO
from urllib.parse import urljoin, urlparse

import requests


class RemoteMCPError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteMCPServerConfig:
    name: str
    url: str
    protocol_version: str = "2024-11-05"
    init_timeout_s: float = 30.0
    headers: Dict[str, str] | None = None


def normalize_sse_url(url: str) -> str:
    cleaned = (url or "").strip()
    if not cleaned:
        raise ValueError("url is required")
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("url must start with http:// or https://")

    path = parsed.path or ""
    if path.endswith("/sse") or path.endswith("/sse/"):
        return cleaned
    if path in {"", "/"}:
        return cleaned.rstrip("/") + "/sse"
    return cleaned


def _tool_result_to_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
            return "\n".join(chunk.strip() for chunk in chunks if chunk.strip())
        if "text" in result and isinstance(result["text"], str):
            return result["text"].strip()
    return str(result).strip()


def _extract_tools(result: Any) -> list[dict[str, Any]]:
    if result is None:
        return []
    if isinstance(result, dict):
        tools = result.get("tools")
        if isinstance(tools, list):
            return [item for item in tools if isinstance(item, dict)]
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    return []


def test_connection(
    *,
    config: RemoteMCPServerConfig,
    tools_timeout_s: float = 10.0,
    log_file: TextIO | None = None,
) -> dict[str, Any]:
    with _MCPSseClient(config=config, log_file=log_file) as client:
        tools_result = client.request("tools/list", {}, timeout_s=tools_timeout_s)
        server_info: dict[str, Any] = {}
        if isinstance(client.initialize_result, dict):
            server_info = (client.initialize_result.get("serverInfo") or {}) if isinstance(client.initialize_result, dict) else {}
        return {
            "server_info": server_info,
            "protocol_version": (client.initialize_result or {}).get("protocolVersion") if isinstance(client.initialize_result, dict) else None,
            "tools": _extract_tools(tools_result),
        }


def list_tools(
    *,
    config: RemoteMCPServerConfig,
    timeout_s: float = 10.0,
    log_file: TextIO | None = None,
) -> list[dict[str, Any]]:
    with _MCPSseClient(config=config, log_file=log_file) as client:
        result = client.request("tools/list", {}, timeout_s=timeout_s)
        return _extract_tools(result)


def call_tool(
    *,
    config: RemoteMCPServerConfig,
    name: str,
    arguments: Dict[str, Any],
    timeout_s: float | None = 60.0,
    log_file: TextIO | None = None,
) -> Any:
    with _MCPSseClient(config=config, log_file=log_file) as client:
        result = client.request("tools/call", {"name": name, "arguments": arguments}, timeout_s=timeout_s)
        return result


def call_tool_text(
    *,
    config: RemoteMCPServerConfig,
    name: str,
    arguments: Dict[str, Any],
    timeout_s: float | None = 60.0,
    log_file: TextIO | None = None,
) -> str:
    return _tool_result_to_text(call_tool(config=config, name=name, arguments=arguments, timeout_s=timeout_s, log_file=log_file))


class _MCPSseClient:
    def __init__(self, *, config: RemoteMCPServerConfig, log_file: TextIO | None) -> None:
        self._config = config
        self._sse_url = normalize_sse_url(config.url)
        self._headers = {k: v for k, v in (config.headers or {}).items() if k and v}
        self._log_file = log_file
        self._log_lock = threading.Lock()
        self._session = requests.Session()
        self._resp: requests.Response | None = None
        self._reader_thread: threading.Thread | None = None
        self._cond = threading.Condition()
        self._responses: dict[int | str, dict[str, Any]] = {}
        self._next_id = 1
        self._reader_done = False
        self._reader_error: Exception | None = None
        self._message_url: str | None = None
        self.initialize_result: dict[str, Any] | None = None

    def __enter__(self) -> "_MCPSseClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    def _log(self, kind: str, message: Any) -> None:
        if self._log_file is None:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        text = message if isinstance(message, str) else json.dumps(message, ensure_ascii=False) if isinstance(message, (dict, list)) else str(message)
        with self._log_lock:
            try:
                for line in (text.splitlines() or [""]):
                    self._log_file.write(f"{ts} [{kind}] {line}\n")
                self._log_file.flush()
            except Exception:
                return

    def start(self) -> None:
        if self._resp is not None:
            return

        headers = {"Accept": "text/event-stream"}
        headers.update(self._headers)
        try:
            self._resp = self._session.get(
                self._sse_url,
                headers=headers,
                stream=True,
                timeout=(5, None),
            )
        except requests.RequestException as exc:
            raise RemoteMCPError(f"Failed to connect to MCP SSE endpoint {self._sse_url!r}: {exc}") from exc

        if self._resp.status_code < 200 or self._resp.status_code >= 300:
            body = ""
            try:
                body = self._resp.text[:4000]
            except Exception:
                pass
            raise RemoteMCPError(f"MCP SSE endpoint returned HTTP {self._resp.status_code}: {body}")

        self._reader_thread = threading.Thread(target=self._reader_loop, name="mcp-sse-reader", daemon=True)
        self._reader_thread.start()

        self._wait_for_endpoint(timeout_s=self._config.init_timeout_s)
        self._initialize()

    def close(self) -> None:
        resp = self._resp
        self._resp = None
        try:
            try:
                self.request("shutdown", {}, timeout_s=2.0)
            except Exception:
                pass
            try:
                self.notify("exit", {})
            except Exception:
                pass
        finally:
            try:
                if resp is not None:
                    resp.close()
            except Exception:
                pass
            try:
                self._session.close()
            except Exception:
                pass

    def notify(self, method: str, params: Dict[str, Any] | None = None) -> None:
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._log("SEND", payload)
        self._send(payload)

    def request(self, method: str, params: Dict[str, Any], *, timeout_s: float | None) -> Any:
        request_id = self._next_id
        self._next_id += 1
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        self._log("SEND", payload)
        self._send(payload)
        response = self._wait_for_response(request_id, timeout_s=timeout_s)
        self._log("RECV", response)
        if "error" in response:
            error = response["error"]
            if isinstance(error, dict):
                raise RemoteMCPError(error.get("message") or str(error))
            raise RemoteMCPError(str(error))
        return response.get("result")

    def _send(self, payload: Dict[str, Any]) -> None:
        message_url = self._message_url
        if not message_url:
            raise RemoteMCPError("MCP message endpoint not established (missing 'endpoint' SSE event).")

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        headers.update(self._headers)
        try:
            resp = self._session.post(message_url, headers=headers, json=payload, timeout=(5, 30))
        except requests.RequestException as exc:
            raise RemoteMCPError(f"Failed to send MCP message to {message_url!r}: {exc}") from exc
        if resp.status_code < 200 or resp.status_code >= 300:
            body = ""
            try:
                body = resp.text[:2000]
            except Exception:
                pass
            raise RemoteMCPError(f"MCP message endpoint returned HTTP {resp.status_code}: {body}")

    def _initialize(self) -> None:
        result = self.request(
            "initialize",
            {
                "protocolVersion": self._config.protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "sat-ai-nos3-example", "version": "0.1.0"},
            },
            timeout_s=self._config.init_timeout_s,
        )
        if result is None:
            raise RemoteMCPError("MCP initialize returned no result.")
        if isinstance(result, dict):
            self.initialize_result = result
        try:
            self.notify("notifications/initialized", {})
        except Exception:
            pass

    def _wait_for_endpoint(self, *, timeout_s: float) -> None:
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        with self._cond:
            while self._message_url is None:
                if self._reader_error is not None:
                    raise RemoteMCPError(f"MCP SSE reader failed: {self._reader_error}") from self._reader_error
                if self._reader_done:
                    raise RemoteMCPError("MCP SSE stream ended before endpoint was received.")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RemoteMCPError(f"MCP SSE endpoint did not provide a message endpoint within {timeout_s}s.")
                self._cond.wait(timeout=min(0.25, remaining))

    def _wait_for_response(self, request_id: int, *, timeout_s: float | None) -> Dict[str, Any]:
        deadline = None if timeout_s is None else (time.monotonic() + timeout_s)
        with self._cond:
            while request_id not in self._responses and str(request_id) not in self._responses:
                if self._reader_error is not None:
                    raise RemoteMCPError(f"MCP SSE reader failed: {self._reader_error}") from self._reader_error
                if self._reader_done:
                    raise RemoteMCPError("MCP SSE stream ended unexpectedly.")
                if deadline is None:
                    self._cond.wait(timeout=0.25)
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RemoteMCPError(f"MCP request timed out ({timeout_s}s): {request_id}")
                    self._cond.wait(timeout=min(0.25, remaining))

            key: int | str = request_id if request_id in self._responses else str(request_id)
            return self._responses.pop(key)

    def _reader_loop(self) -> None:
        try:
            resp = self._resp
            if resp is None:
                return

            event_type: str | None = None
            data_lines: list[str] = []
            for raw in resp.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                line = raw.rstrip("\r\n")
                if not line:
                    if data_lines:
                        data = "\n".join(data_lines)
                        evt = event_type or "message"
                        self._handle_event(evt, data)
                    event_type = None
                    data_lines = []
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("event:"):
                    event_type = line.split(":", 1)[1].strip()
                    continue
                if line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].lstrip())
                    continue

            if data_lines:
                data = "\n".join(data_lines)
                evt = event_type or "message"
                self._handle_event(evt, data)
        except Exception as exc:  # noqa: BLE001
            self._log("ERROR", f"SSE reader loop failed: {exc}")
            with self._cond:
                self._reader_error = exc
                self._cond.notify_all()
        finally:
            with self._cond:
                self._reader_done = True
                self._cond.notify_all()

    def _handle_event(self, event_type: str, data: str) -> None:
        if event_type == "endpoint":
            endpoint_raw = (data or "").strip().strip('"').strip("'")
            if not endpoint_raw:
                return
            message_url = urljoin(self._sse_url, endpoint_raw)
            with self._cond:
                self._message_url = message_url
                self._cond.notify_all()
            self._log("SERVER", f"endpoint: {message_url}")
            return

        if event_type != "message":
            self._log("SERVER", f"{event_type}: {data}")
            return

        payload: Any
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            self._log("SERVER", data)
            return

        if not isinstance(payload, dict):
            self._log("RECV", payload)
            return

        msg_id = payload.get("id")
        if isinstance(msg_id, (int, str)):
            with self._cond:
                self._responses[msg_id] = payload
                self._cond.notify_all()
            return

        self._log("RECV", payload)
