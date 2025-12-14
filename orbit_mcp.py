from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Deque, Dict, Optional, TextIO, Tuple

from collections import deque


class OrbitMCPError(RuntimeError):
    pass


@dataclass(frozen=True)
class OrbitMCPConfig:
    python_executable: str
    server_script: Path
    protocol_version: str = "2024-11-05"
    init_timeout_s: float = 30.0
    env: Dict[str, str] | None = None
    framing: str = "ndjson"  # "ndjson" (FastMCP) or "lsp"


_ORBIT_SIM_LOCK = threading.Lock()
_ACTIVE_SIM_LOCK = threading.Lock()
_ACTIVE_SIM_CLIENT: "_MCPStdioClient | None" = None
_ACTIVE_SIM_LOG_PATH: Path | None = None
_ACTIVE_SIM_RUN_DIR: Path | None = None
_ACTIVE_SIM_STARTED_S: float | None = None

_REPO_ROOT = Path(__file__).resolve().parent


def default_orbit_mcp_log_dir() -> Path:
    return _REPO_ROOT / "storage" / "orbit_mcp_logs"


def default_orbit_mcp_run_dir() -> Path:
    return _REPO_ROOT / "storage" / "orbit_mcp_runs"


def new_orbit_mcp_run_dir(prefix: str, *, run_dir: Path | None = None) -> Path:
    safe_prefix = re.sub(r"[^a-zA-Z0-9_.-]+", "_", (prefix or "").strip()) or "orbit_run"
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    unique = f"{os.getpid()}_{time.time_ns()}"
    directory = run_dir or default_orbit_mcp_run_dir()
    return directory / f"{safe_prefix}_{timestamp}_{unique}"


def new_orbit_mcp_log_path(prefix: str, *, log_dir: Path | None = None) -> Path:
    safe_prefix = re.sub(r"[^a-zA-Z0-9_.-]+", "_", (prefix or "").strip()) or "orbit_mcp"
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    unique = f"{os.getpid()}_{time.time_ns()}"
    directory = log_dir or default_orbit_mcp_log_dir()
    return directory / f"{safe_prefix}_{timestamp}_{unique}.log"


def load_orbit_mcp_config() -> OrbitMCPConfig:
    base_dir = Path(__file__).resolve().parent
    default_server = base_dir / "SAT_Orbit_Sim_MCP" / "satellite_server.py"
    server_script_raw = os.getenv("SAT_ORBIT_MCP_SERVER", str(default_server))
    server_script = Path(server_script_raw).expanduser()
    if not server_script.is_absolute():
        server_script = (base_dir / server_script).resolve()

    python_executable = os.getenv("SAT_ORBIT_MCP_PYTHON", sys.executable)
    protocol_version = os.getenv("SAT_ORBIT_MCP_PROTOCOL_VERSION", "2024-11-05")
    init_timeout_s = _env_float("SAT_ORBIT_MCP_INIT_TIMEOUT_S", 30.0)
    framing = (os.getenv("SAT_ORBIT_MCP_FRAMING", "ndjson") or "ndjson").strip().lower()
    return OrbitMCPConfig(
        python_executable=python_executable,
        server_script=server_script,
        protocol_version=protocol_version,
        init_timeout_s=init_timeout_s,
        framing=framing,
    )


def is_orbit_command(message: str) -> bool:
    stripped = (message or "").strip().lower()
    return stripped.startswith("/orbit") or stripped.startswith("/visualize_orbit") or stripped.startswith("/simulate_orbit")


def parse_orbit_command(message: str) -> Optional[Tuple[float, float]]:
    """
    Parse `/orbit <lat> <lon>` commands.

    Examples:
      /orbit 40.7128 -74.0060
      /orbit 48.85, 2.35
      /orbit lat=48.85 lon=2.35
    """
    stripped = (message or "").strip()
    if not stripped:
        return None

    if not is_orbit_command(stripped):
        return None

    # Remove the command token.
    parts = stripped.split(maxsplit=1)
    args = parts[1] if len(parts) > 1 else ""

    numbers = _extract_floats(args)
    if len(numbers) < 2:
        return None

    lat, lon = numbers[0], numbers[1]
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return lat, lon


def visualize_orbit(
    latitude: float,
    longitude: float,
    *,
    config: OrbitMCPConfig | None = None,
    log_path: Path | None = None,
) -> str:
    """
    Call the SAT_Orbit_Sim_MCP `visualize_orbit` tool over MCP stdio.

    Note: the MCP server blocks until the Matplotlib window closes.
    """
    if not (-90.0 <= latitude <= 90.0):
        raise ValueError("latitude must be between -90 and 90 degrees")
    if not (-180.0 <= longitude <= 180.0):
        raise ValueError("longitude must be between -180 and 180 degrees")

    if config is None:
        config = load_orbit_mcp_config()

    if not config.server_script.exists():
        raise OrbitMCPError(f"MCP server script not found: {config.server_script}")

    acquired = _ORBIT_SIM_LOCK.acquire(blocking=False)
    if not acquired:
        raise OrbitMCPError("Orbit simulation already running; try again once it finishes.")

    try:
        return _visualize_orbit_without_lock(latitude, longitude, config=config, log_path=log_path)
    finally:
        _ORBIT_SIM_LOCK.release()


def start_visualize_orbit(
    latitude: float,
    longitude: float,
    *,
    config: OrbitMCPConfig | None = None,
    on_done: Callable[[str | None, Exception | None], None] | None = None,
    log_path: Path | None = None,
) -> threading.Thread:
    """
    Start the orbit visualization in a daemon thread and return immediately.

    This is useful for web UIs where blocking the request would freeze the app.
    """
    if not (-90.0 <= latitude <= 90.0):
        raise ValueError("latitude must be between -90 and 90 degrees")
    if not (-180.0 <= longitude <= 180.0):
        raise ValueError("longitude must be between -180 and 180 degrees")

    if config is None:
        config = load_orbit_mcp_config()

    if not config.server_script.exists():
        raise OrbitMCPError(f"MCP server script not found: {config.server_script}")

    acquired = _ORBIT_SIM_LOCK.acquire(blocking=False)
    if not acquired:
        raise OrbitMCPError("Orbit simulation already running; try again once it finishes.")

    def runner() -> None:
        try:
            result = _visualize_orbit_without_lock(latitude, longitude, config=config, log_path=log_path)
            if on_done is not None:
                on_done(result, None)
        except Exception as exc:  # noqa: BLE001 - surfaced via callback/logging
            if on_done is not None:
                on_done(None, exc)
        finally:
            _ORBIT_SIM_LOCK.release()

    thread = threading.Thread(target=runner, name="orbit-visualize", daemon=True)
    thread.start()
    return thread


def get_active_orbit_simulation() -> dict[str, Any]:
    with _ACTIVE_SIM_LOCK:
        client = _ACTIVE_SIM_CLIENT
        log_path = _ACTIVE_SIM_LOG_PATH
        run_dir = _ACTIVE_SIM_RUN_DIR
        started_s = _ACTIVE_SIM_STARTED_S

    pid: int | None = None
    running = False
    if client is not None:
        try:
            pid = client.pid
            running = client.is_running()
        except Exception:
            running = False

    return {
        "running": bool(running),
        "pid": pid,
        "log_path": str(log_path) if log_path is not None else None,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "started_s": started_s,
    }


def stop_active_orbit_simulation() -> bool:
    """
    Force-stop the currently running orbit visualization (if any).

    Returns True if a running simulation was signaled to stop.
    """
    with _ACTIVE_SIM_LOCK:
        client = _ACTIVE_SIM_CLIENT
    if client is None:
        return False
    try:
        client.abort()
        return True
    except Exception:
        return False


def _visualize_orbit_without_lock(
    latitude: float,
    longitude: float,
    *,
    config: OrbitMCPConfig,
    log_path: Path | None,
) -> str:
    run_dir: Path | None = None
    if config.env:
        raw_run_dir = config.env.get("SAT_ORBIT_RUN_DIR")
        if raw_run_dir:
            try:
                run_dir = Path(raw_run_dir).expanduser()
            except Exception:
                run_dir = None

    with _MCPStdioClient(
        [config.python_executable, str(config.server_script)],
        protocol_version=config.protocol_version,
        init_timeout_s=config.init_timeout_s,
        env=config.env,
        framing=config.framing,
        log_path=log_path,
    ) as client:
        with _ACTIVE_SIM_LOCK:
            global _ACTIVE_SIM_CLIENT, _ACTIVE_SIM_LOG_PATH, _ACTIVE_SIM_RUN_DIR, _ACTIVE_SIM_STARTED_S
            _ACTIVE_SIM_CLIENT = client
            _ACTIVE_SIM_LOG_PATH = log_path
            _ACTIVE_SIM_RUN_DIR = run_dir
            _ACTIVE_SIM_STARTED_S = time.time()
        try:
            result = client.call_tool(
                "visualize_orbit",
                {"latitude": latitude, "longitude": longitude},
                timeout_s=None,
            )
        finally:
            with _ACTIVE_SIM_LOCK:
                if _ACTIVE_SIM_CLIENT is client:
                    _ACTIVE_SIM_CLIENT = None
                    _ACTIVE_SIM_LOG_PATH = None
                    _ACTIVE_SIM_RUN_DIR = None
                    _ACTIVE_SIM_STARTED_S = None
        return _tool_result_to_text(result) or "Orbit simulation request completed."


def list_tools(
    *,
    config: OrbitMCPConfig | None = None,
    timeout_s: float = 10.0,
    log_path: Path | None = None,
) -> list[dict[str, Any]]:
    if config is None:
        config = load_orbit_mcp_config()

    with _MCPStdioClient(
        [config.python_executable, str(config.server_script)],
        protocol_version=config.protocol_version,
        init_timeout_s=config.init_timeout_s,
        env=config.env,
        framing=config.framing,
        log_path=log_path,
    ) as client:
        result = client.request("tools/list", {}, timeout_s=timeout_s)
        return _extract_tools(result)


def test_connection(
    *,
    config: OrbitMCPConfig | None = None,
    tools_timeout_s: float = 10.0,
    log_path: Path | None = None,
) -> dict[str, Any]:
    if config is None:
        config = load_orbit_mcp_config()

    with _MCPStdioClient(
        [config.python_executable, str(config.server_script)],
        protocol_version=config.protocol_version,
        init_timeout_s=config.init_timeout_s,
        env=config.env,
        framing=config.framing,
        log_path=log_path,
    ) as client:
        tools_result = client.request("tools/list", {}, timeout_s=tools_timeout_s)
        return {
            "initialize": client.initialize_result or {},
            "tools": _extract_tools(tools_result),
        }


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _extract_floats(text: str) -> list[float]:
    # Supports "48.85, 2.35" and "lat=48.85 lon=2.35" styles.
    matches = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
    values: list[float] = []
    for match in matches:
        try:
            values.append(float(match))
        except ValueError:
            continue
    return values


def _encode_lsp_message(payload: Dict[str, Any]) -> bytes:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    return header + body


def _encode_ndjson_message(payload: Dict[str, Any]) -> bytes:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return body + b"\n"


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


class _MCPStdioClient:
    def __init__(
        self,
        command: list[str],
        *,
        protocol_version: str,
        init_timeout_s: float,
        env: Dict[str, str] | None,
        framing: str,
        log_path: Path | None,
    ) -> None:
        self._command = command
        self._protocol_version = protocol_version
        self._init_timeout_s = init_timeout_s
        self._env_overrides = env or {}
        self._framing = (framing or "ndjson").strip().lower()
        self._log_path = log_path
        self._log_file: TextIO | None = None
        self._log_lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._reader_thread: threading.Thread | None = None
        self._responses: dict[int | str, dict[str, Any]] = {}
        self._cond = threading.Condition()
        self._write_lock = threading.Lock()
        self._next_id = 1
        self._reader_done = False
        self._reader_error: Exception | None = None
        self.initialize_result: dict[str, Any] | None = None
        self._recent_output: Deque[str] = deque(maxlen=200)
        self._recent_output_lock = threading.Lock()

    def __enter__(self) -> "_MCPStdioClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    def start(self) -> None:
        if self._proc is not None:
            return

        self._open_log()
        self._log("CLIENT", f"Starting MCP stdio client; framing={self._framing}")
        self._log("CLIENT", f"Command: {self._command!r}")
        self._log("CLIENT", f"Host PID: {os.getpid()}")
        self._log("CLIENT", f"Host platform: {sys.platform}")
        self._log("CLIENT", f"Host python: {sys.version.replace(os.linesep, ' ')}")
        self._log("CLIENT", f"Host cwd: {os.getcwd()}")
        if self._env_overrides:
            self._log("CLIENT", f"Env overrides: {sorted(self._env_overrides.keys())}")

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.update(self._env_overrides)

        try:
            self._proc = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            self._log("ERROR", f"Failed to start MCP server: {exc}")
            raise OrbitMCPError(f"Failed to start MCP server: {exc}") from exc

        assert self._proc.stdout is not None
        assert self._proc.stderr is not None
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(self._proc.stdout,),
            name="mcp-stdio-reader",
            daemon=True,
        )
        self._reader_thread.start()
        threading.Thread(
            target=self._stderr_loop,
            args=(self._proc.stderr,),
            name="mcp-stdio-stderr",
            daemon=True,
        ).start()

        self._initialize()

    @property
    def pid(self) -> int | None:
        proc = self._proc
        if proc is None:
            return None
        return proc.pid

    def is_running(self) -> bool:
        proc = self._proc
        if proc is None:
            return False
        return proc.poll() is None

    def abort(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            self._log("CLIENT", "Abort requested; terminating MCP server process")
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        with self._cond:
            self._cond.notify_all()

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            self._close_log()
            return

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
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        self._proc = None
        try:
            self._log("CLIENT", f"MCP server exit code: {proc.returncode}")
        except Exception:
            pass
        self._log("CLIENT", "Closed MCP stdio client")
        self._close_log()

    def call_tool(self, name: str, arguments: Dict[str, Any], *, timeout_s: float | None) -> Any:
        return self.request(
            "tools/call",
            {"name": name, "arguments": arguments},
            timeout_s=timeout_s,
        )

    def notify(self, method: str, params: Dict[str, Any] | None = None) -> None:
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        self._log("SEND", payload)
        self._send(payload)

    def request(self, method: str, params: Dict[str, Any], *, timeout_s: float | None) -> Any:
        request_id = self._next_id
        self._next_id += 1

        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }
        self._log("SEND", payload)
        self._send(payload)

        response = self._wait_for_response(request_id, timeout_s=timeout_s)
        self._log("RECV", response)
        if "error" in response:
            error = response["error"]
            if isinstance(error, dict):
                message = error.get("message")
                raise OrbitMCPError(message or str(error))
            raise OrbitMCPError(str(error))
        return response.get("result")

    def _send(self, payload: Dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise OrbitMCPError("MCP server process is not running.")
        message = _encode_ndjson_message(payload) if self._framing != "lsp" else _encode_lsp_message(payload)
        with self._write_lock:
            try:
                proc.stdin.write(message)
                proc.stdin.flush()
            except BrokenPipeError as exc:
                self._log("ERROR", f"BrokenPipeError while sending {payload.get('method')!r}")
                raise OrbitMCPError("MCP server closed the pipe unexpectedly.") from exc

    def _initialize(self) -> None:
        result = self.request(
            "initialize",
            {
                "protocolVersion": self._protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "sat-ai-nos3-example", "version": "0.1.0"},
            },
            timeout_s=self._init_timeout_s,
        )
        if result is None:
            raise OrbitMCPError("MCP initialize returned no result.")
        if isinstance(result, dict):
            self.initialize_result = result
        # MCP uses an initialized notification after initialize.
        try:
            self.notify("notifications/initialized", {})
        except Exception:
            pass

    def _wait_for_response(self, request_id: int, *, timeout_s: float | None) -> Dict[str, Any]:
        deadline = None if timeout_s is None else (time.monotonic() + timeout_s)
        proc = self._proc
        if proc is None:
            raise OrbitMCPError("MCP server process is not running.")

        with self._cond:
            while request_id not in self._responses and str(request_id) not in self._responses:
                if self._reader_error is not None:
                    raise OrbitMCPError(f"MCP reader failed: {self._reader_error}") from self._reader_error
                if self._reader_done and proc.poll() is not None:
                    detail = self._format_recent_output()
                    raise OrbitMCPError(
                        "MCP server process exited unexpectedly."
                        + (f"\n\nRecent server output:\n{detail}" if detail else "")
                    )

                if deadline is None:
                    self._cond.wait(timeout=0.25)
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        detail = self._format_recent_output()
                        raise OrbitMCPError(
                            f"MCP request timed out ({timeout_s}s): {request_id}"
                            + (f"\n\nRecent server output:\n{detail}" if detail else "")
                        )
                    self._cond.wait(timeout=min(0.25, remaining))

            key: int | str = request_id if request_id in self._responses else str(request_id)
            return self._responses.pop(key)

    def _record_output_line(self, line: str) -> None:
        cleaned = (line or "").rstrip("\r\n")
        if not cleaned:
            return
        with self._recent_output_lock:
            self._recent_output.append(cleaned)
        self._log("SERVER", cleaned)

    def _format_recent_output(self) -> str:
        with self._recent_output_lock:
            lines = list(self._recent_output)
        return "\n".join(lines).strip()

    def _open_log(self) -> None:
        if self._log_path is None:
            return
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self._log_path, "a", encoding="utf-8")
        except Exception:
            self._log_file = None

    def _close_log(self) -> None:
        with self._log_lock:
            log_file = self._log_file
            if log_file is None:
                return
            try:
                log_file.close()
            except Exception:
                pass
            self._log_file = None

    def _log(self, kind: str, payload: Any) -> None:
        if self._log_file is None:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if isinstance(payload, (dict, list)):
            try:
                text = json.dumps(payload, ensure_ascii=False)
            except Exception:
                text = str(payload)
        else:
            text = str(payload)
        lines = text.splitlines() or [""]
        with self._log_lock:
            log_file = self._log_file
            if log_file is None:
                return
            try:
                for line in lines:
                    log_file.write(f"{ts} [{kind}] {line}\n")
                log_file.flush()
            except ValueError:
                # Log file was closed concurrently; disable further logging.
                try:
                    log_file.close()
                except Exception:
                    pass
                self._log_file = None
            except Exception:
                return

    def _reader_loop(self, stdout: BinaryIO) -> None:
        try:
            for message in _iter_mcp_messages(stdout, on_output_line=self._record_output_line):
                if not isinstance(message, dict):
                    continue
                msg_id = message.get("id")
                if isinstance(msg_id, (int, str)):
                    with self._cond:
                        self._responses[msg_id] = message
                        self._cond.notify_all()
                else:
                    # Log notifications/requests that aren't responses to a client request.
                    self._log("RECV", message)
        except Exception as exc:  # noqa: BLE001 - surface reader failures to waiting requests
            self._log("ERROR", f"Reader loop failed: {exc}")
            with self._cond:
                self._reader_error = exc
                self._cond.notify_all()
        finally:
            with self._cond:
                self._reader_done = True
                self._cond.notify_all()

    def _stderr_loop(self, stderr: BinaryIO) -> None:
        try:
            while True:
                line = stderr.readline()
                if not line:
                    return
                try:
                    decoded = line.decode("utf-8", errors="replace")
                except Exception:
                    decoded = str(line)
                self._record_output_line(decoded)
        except Exception as exc:  # noqa: BLE001 - best-effort diagnostics only
            self._log("ERROR", f"Stderr loop failed: {exc}")


def _iter_mcp_messages(stdout: BinaryIO, *, on_output_line: Callable[[str], None] | None = None):
    while True:
        line = stdout.readline()
        if not line:
            return

        stripped = line.strip()
        if not stripped:
            continue

        lower = stripped.lower()
        if lower.startswith(b"content-length:"):
            length_raw = stripped.split(b":", 1)[1].strip()
            try:
                content_length = int(length_raw)
            except ValueError:
                continue

            # Consume remaining headers until the blank separator line.
            while True:
                header_line = stdout.readline()
                if not header_line:
                    return
                if header_line.strip() == b"":
                    break

            body = stdout.read(content_length)
            if not body:
                continue
            try:
                yield json.loads(body.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            continue

        # Fallback: newline-delimited JSON (ignore anything else).
        if not stripped.startswith(b"{"):
            if on_output_line is not None:
                try:
                    on_output_line(line.decode("utf-8", errors="replace"))
                except Exception:
                    pass
            continue
        try:
            yield json.loads(stripped.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            if on_output_line is not None:
                try:
                    on_output_line(line.decode("utf-8", errors="replace"))
                except Exception:
                    pass
            continue
