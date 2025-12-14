"""
FastAPI-powered web chat interface for the NOS3 documentation RAG assistant.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urlparse, urlunparse

import requests
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_sambanova import ChatSambaNova
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import uvicorn

from rag_chat import (
    DEFAULT_CEREBRAS_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_PERSIST_DIR,
    DEFAULT_SOURCE_URL,
    build_chat_chain,
    get_vector_store,
)

from orbit_mcp import (
    OrbitMCPConfig,
    OrbitMCPError,
    default_orbit_mcp_log_dir,
    default_orbit_mcp_run_dir,
    get_active_orbit_simulation,
    is_orbit_command,
    new_orbit_mcp_log_path,
    new_orbit_mcp_run_dir,
    parse_orbit_command,
    start_visualize_orbit,
    stop_active_orbit_simulation,
    test_connection,
)


_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")

_ORBIT_INTENT_KEYWORDS = (
    "orbit sim",
    "orbit simulation",
    "simulate orbit",
    "visualize orbit",
    "run orbit",
    "start orbit",
    "launch orbit",
    "show orbit",
    "plot orbit",
    "animate orbit",
    "satellite sim",
)

_ORBIT_CONTEXT_KEYWORDS = ("orbit", "satellite", "iss", "target", "latitude", "longitude", "lat", "lon")
_ORBIT_ACTION_KEYWORDS = ("run", "start", "launch", "simulate", "visualize", "show", "plot", "animate")


def _extract_lat_lon_from_text(text: str) -> tuple[float, float] | None:
    matches = _FLOAT_RE.findall(text or "")
    if len(matches) < 2:
        return None
    values: list[float] = []
    for match in matches:
        try:
            values.append(float(match))
        except ValueError:
            continue
    for idx in range(len(values) - 1):
        lat, lon = values[idx], values[idx + 1]
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            return lat, lon
    return None


def _should_attempt_orbit_nl_routing(message: str, history: list["ChatTurn"]) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return False
    coords = _extract_lat_lon_from_text(text)
    if any(keyword in text for keyword in _ORBIT_INTENT_KEYWORDS):
        return True
    if any(keyword in text for keyword in _ORBIT_ACTION_KEYWORDS) and any(keyword in text for keyword in _ORBIT_CONTEXT_KEYWORDS):
        return True
    if coords is None:
        return False
    if any(keyword in text for keyword in _ORBIT_ACTION_KEYWORDS):
        return True
    if "target" in text:
        return True
    if history:
        last = (history[-1].answer or "").lower()
        if "orbit" in last and ("latitude" in last or "longitude" in last or "/orbit" in last):
            return True
    return False


def _parse_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if "\n" in raw:
            raw = raw.split("\n", 1)[1].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    candidate = raw[start : end + 1] if start != -1 and end != -1 and end > start else raw
    try:
        payload = json.loads(candidate)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _create_router_llm(provider: str, model_name: str, api_key: str):
    provider = (provider or "").strip().lower()
    if provider == "cerebras":
        from langchain_cerebras import ChatCerebras

        try:
            return ChatCerebras(model=model_name, cerebras_api_key=api_key, temperature=0)
        except TypeError:
            return ChatCerebras(model=model_name, cerebras_api_key=api_key)
    if provider == "groq":
        try:
            return ChatGroq(model_name=model_name, groq_api_key=api_key, temperature=0)
        except TypeError:
            return ChatGroq(model_name=model_name, groq_api_key=api_key)
    if provider == "sambanova":
        try:
            return ChatSambaNova(model=model_name, api_key=api_key, temperature=0)
        except TypeError:
            return ChatSambaNova(model=model_name, api_key=api_key)
    raise ValueError(f"Unsupported provider for router: {provider}")


def _select_router_model(config: "WebConfig", selected_models: dict[str, str], provider: str) -> str | None:
    provider = provider.lower()
    selected = selected_models.get(provider)
    if selected:
        return selected
    if provider == "cerebras":
        return config.cerebras_model
    if provider == "groq":
        return (config.groq_models or [None])[0]
    if provider == "sambanova":
        return (config.sambanova_models or [None])[0]
    return None


def _route_orbit_request_with_llm(llm, message: str) -> dict[str, Any] | None:
    system = SystemMessage(
        content=(
            "You are a router for Sat.AI.\n"
            "Sat.AI has two separate capabilities:\n"
            "1) Answer questions about the NOS3 documentation.\n"
            "2) Run an orbit visualization tool called `visualize_orbit` that requires a target latitude and longitude in degrees.\n\n"
            "Decide what to do for the user's message.\n"
            "- If the user is asking to run/launch/simulate/visualize the orbit tool, extract coordinates and output ONLY valid JSON:\n"
            '  {"action":"orbit","latitude":<float>,"longitude":<float>}\n'
            "- If they want the orbit tool but coordinates are missing/unclear, output ONLY valid JSON:\n"
            '  {"action":"clarify","question":"<ask for latitude and longitude>"}\n'
            "- Otherwise output ONLY valid JSON:\n"
            '  {"action":"chat"}\n\n'
            "Rules:\n"
            "- Only output JSON. No markdown, no backticks, no extra text.\n"
            "- Latitude must be between -90 and 90. Longitude between -180 and 180.\n"
        )
    )
    human = HumanMessage(content=message)
    try:
        response = llm.invoke([system, human])
    except Exception:
        return None
    content = getattr(response, "content", None)
    if content is None:
        content = str(response)
    return _parse_json_object(str(content))


def _start_orbit_from_chat(config: "WebConfig", lat: float, lon: float) -> tuple[str, str, str]:
    run_dir = new_orbit_mcp_run_dir("orbit_mcp_run", run_dir=default_orbit_mcp_run_dir())
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    run_dir_text = _format_repo_path(run_dir)

    orbit_config = OrbitMCPConfig(
        python_executable=config.orbit_mcp_python,
        server_script=config.orbit_mcp_server,
        protocol_version=config.orbit_mcp_protocol_version,
        init_timeout_s=config.orbit_mcp_init_timeout_s,
        framing=config.orbit_mcp_framing,
        env={
            "SAT_ORBIT_RUN_DIR": str(run_dir),
            "SAT_ORBIT_ROLE": "server",
        },
    )

    done_event = threading.Event()
    outcome: Dict[str, Any] = {"result": None, "error": None}
    log_path = new_orbit_mcp_log_path("orbit_mcp_orbit", log_dir=default_orbit_mcp_log_dir())
    log_path_text = _format_repo_path(log_path)

    def on_done(result: str | None, err: Exception | None) -> None:
        outcome["result"] = result
        outcome["error"] = err
        done_event.set()
        print(
            f"[orbit] {'error: ' + str(err) if err else 'completed: ' + (result or '')} (log={log_path_text} run={run_dir_text})",
            file=sys.stderr,
            flush=True,
        )

    start_visualize_orbit(
        lat,
        lon,
        config=orbit_config,
        on_done=on_done,
        log_path=log_path,
    )

    if done_event.wait(timeout=0.75):
        err = outcome.get("error")
        result = outcome.get("result")
        if err is not None:
            answer = f"Orbit visualization failed: {err}\nRun: `{run_dir_text}`\nLog: `{log_path_text}`"
        else:
            answer = ((result or "").strip() or "Orbit visualization finished.") + f"\nRun: `{run_dir_text}`\nLog: `{log_path_text}`"
    else:
        answer = (
            f"Orbit visualization started for ({lat}, {lon}). "
            "A Matplotlib window should open on the server machine; close it to finish."
            f"\nRun: `{run_dir_text}`"
            f"\nLog: `{log_path_text}`"
        )

    return answer, run_dir_text, log_path_text

PROVIDER_LABELS: Dict[str, str] = {
    "cerebras": "Cerebras",
    "groq": "Groq",
    "sambanova": "SambaNova",
}

DEFAULT_MODEL_CANDIDATES: Dict[str, List[str]] = {
    "cerebras": ["gpt-oss-120b"],
    "groq": ["openai/gpt-oss-120b"],
    "sambanova": ["gpt-oss-120b"],
}

PROVIDER_MODEL_ENDPOINTS: Dict[str, str] = {
    "cerebras": os.getenv("SAT_WEB_CEREBRAS_MODELS_URL", "https://api.cerebras.ai/v1/models"),
    "groq": os.getenv("SAT_WEB_GROQ_MODELS_URL", "https://api.groq.com/openai/v1/models"),
    "sambanova": os.getenv("SAT_WEB_SAMBANOVA_MODELS_URL", "https://api.sambanova.ai/v1/models"),
}


@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    persist_dir: Path = DEFAULT_PERSIST_DIR
    source_url: str = DEFAULT_SOURCE_URL
    max_depth: int = 2
    chunk_size: int = 1000
    chunk_overlap: int = 150
    rebuild: bool = False
    ollama_url: str = DEFAULT_OLLAMA_URL
    embedding_model: str = DEFAULT_EMBED_MODEL
    cerebras_model: str = DEFAULT_CEREBRAS_MODEL
    cerebras_api_key: str | None = None
    cerebras_models: List[str] = field(default_factory=lambda: DEFAULT_MODEL_CANDIDATES["cerebras"][:])
    groq_api_key: str | None = None
    sambanova_api_key: str | None = None
    groq_models: List[str] = field(default_factory=lambda: DEFAULT_MODEL_CANDIDATES["groq"][:])
    sambanova_models: List[str] = field(default_factory=lambda: DEFAULT_MODEL_CANDIDATES["sambanova"][:])
    orbit_mcp_python: str = field(default_factory=lambda: sys.executable)
    orbit_mcp_server: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "SAT_Orbit_Sim_MCP" / "satellite_server.py"
    )
    orbit_mcp_protocol_version: str = "2024-11-05"
    orbit_mcp_init_timeout_s: float = 30.0
    orbit_mcp_framing: str = "ndjson"


_REPO_ROOT = Path(__file__).resolve().parent


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    return path


def _format_repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except Exception:
        return str(path)


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sat.AI</title>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "Segoe UI", Tahoma, sans-serif; background: #0b1726; color: #f4f6fb; }
    a { color: #6fb1ff; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .container { max-width: 960px; margin: 0 auto; padding: 32px 24px; display: flex; flex-direction: column; gap: 16px; min-height: 100vh; }
    .header { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 16px; align-items: flex-start; }
    .title { margin: 0; font-size: 2rem; letter-spacing: 0.02em; }
    .subtitle { margin: 4px 0 0; color: #a5b6d0; max-width: 640px; }
    .control-bar { display: flex; gap: 12px; flex-wrap: wrap; }
    .button-row { display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }
    .button, .chat-form button { padding: 12px 20px; border-radius: 999px; border: none; background: #3d7eff; color: #fff; font-weight: 600; cursor: pointer; transition: background 0.2s ease; }
    .button.secondary { background: transparent; border: 1px solid #34527a; color: #c7d5eb; }
    .button:hover:not(:disabled), .chat-form button:hover:not(:disabled) { background: #2667d4; }
    .button.secondary:hover:not(:disabled) { background: rgba(61, 126, 255, 0.16); }
    .button:disabled, .chat-form button:disabled { background: #2a4f7f; cursor: not-allowed; }
    .settings-panel { background: rgba(16, 31, 50, 0.8); border: 1px solid #233654; border-radius: 16px; padding: 20px 24px; display: flex; flex-direction: column; gap: 16px; }
    .settings-panel.hidden { display: none; }
    .settings-header { display: flex; flex-direction: column; gap: 6px; }
    .settings-header h2 { margin: 0; font-size: 1.1rem; letter-spacing: 0.05em; text-transform: uppercase; color: #9ebcff; }
    .settings-header p { margin: 0; color: #8aa3c6; font-size: 0.9rem; }
    .settings-status { min-height: 1.2rem; font-size: 0.85rem; color: #9ebcff; }
    .settings-status.error { color: #ff7586; }
    .settings-grid { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
    .settings-card { background: rgba(12, 24, 40, 0.85); border: 1px solid #1f2f4a; border-radius: 14px; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
    .settings-card-header { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
    .settings-card-title { font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #b5c9ea; font-size: 0.85rem; }
    .settings-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; color: #8aa3c6; }
    .input-row { display: flex; gap: 10px; align-items: center; }
    .input-row input[type="password"] { flex: 1; padding: 10px 14px; border-radius: 999px; border: 1px solid #274268; background: #0f1e32; color: #f4f6fb; font-size: 0.95rem; }
    .input-row input[type="password"]:focus { outline: none; border-color: #3d7eff; box-shadow: 0 0 0 3px rgba(61, 126, 255, 0.2); }
    .settings-remember { display: flex; gap: 6px; align-items: center; color: #8aa3c6; font-size: 0.8rem; }
    .settings-remember input { accent-color: #3d7eff; }
    .model-select { width: 100%; padding: 10px 14px; border-radius: 10px; border: 1px solid #274268; background: #0f1e32; color: #f4f6fb; font-size: 0.95rem; }
    .model-select:focus { outline: none; border-color: #3d7eff; box-shadow: 0 0 0 3px rgba(61, 126, 255, 0.2); }
    .chat-log { flex: 1; min-height: 60vh; background: rgba(16, 31, 50, 0.85); border: 1px solid #233654; border-radius: 16px; padding: 24px; overflow-y: auto; box-shadow: inset 0 0 12px rgba(0, 0, 0, 0.2); }
    .message { margin-bottom: 18px; padding: 14px 18px; border-radius: 12px; line-height: 1.6; }
    .message .role { font-weight: 600; margin-bottom: 6px; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.06em; color: #8aa3c6; }
    .message.user { background: #1f3454; }
    .message.user .role { color: #6fb1ff; }
    .message.assistant { background: #16263d; }
    .message.assistant .role { color: #a0c6ff; }
    .message .content p { margin: 0 0 10px; }
    .message .content code { background: rgba(61, 126, 255, 0.16); padding: 2px 6px; border-radius: 6px; font-family: "Fira Code", Consolas, monospace; font-size: 0.9rem; }
    .message .content pre { background: #0f1f33; color: #f4f6fb; padding: 14px 16px; border-radius: 10px; overflow-x: auto; border: 1px solid #233654; }
    .message .content pre code { background: none; padding: 0; }
    .message .content table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.95rem; }
    .message .content th, .message .content td { border: 1px solid #2d405f; padding: 10px 12px; text-align: left; }
    .message .content th { background: #1d2f48; color: #c7d5eb; }
    .message .content ul, .message .content ol { margin: 0 0 12px 20px; padding-left: 12px; }
    .message .content blockquote { border-left: 4px solid rgba(61, 126, 255, 0.5); padding-left: 12px; color: #c7d5eb; margin: 0 0 12px; }
    .provider-tag { display: inline-flex; align-items: center; gap: 6px; margin-top: 6px; padding: 4px 10px; border-radius: 999px; background: rgba(61, 126, 255, 0.12); color: #9ebcff; font-size: 0.75rem; letter-spacing: 0.04em; text-transform: uppercase; }
    .provider-tag::before { content: "↺"; font-size: 0.8rem; }
    .chat-form { display: flex; gap: 12px; }
    .chat-form input { flex: 1; padding: 14px 18px; border-radius: 999px; border: 1px solid #274268; background: #0f1e32; color: #f4f6fb; font-size: 1rem; }
    .chat-form input:focus { outline: none; border-color: #3d7eff; box-shadow: 0 0 0 3px rgba(61, 126, 255, 0.2); }
    .sources { background: #111e33; border-left: 4px solid #3d7eff; padding: 12px 16px; margin: 8px 0 16px; border-radius: 8px; color: #c7d5eb; }
    .sources .label { font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; color: #8aa3c6; }
    .sources ul { margin: 0; padding-left: 20px; }
    .mcp-tools { font-size: 0.85rem; color: #c7d5eb; line-height: 1.4; }
    .mcp-tools ul { margin: 0; padding-left: 18px; }
    .mcp-tools li { margin: 4px 0; }
    .mcp-log-header { margin-top: 10px; }
    .mcp-log-path { font-size: 0.8rem; color: #8aa3c6; word-break: break-all; }
    .mcp-log-list { background: rgba(15, 31, 51, 0.6); border: 1px solid #233654; border-radius: 12px; padding: 10px 12px; max-height: 170px; overflow-y: auto; }
    .mcp-log-item { width: 100%; padding: 8px 10px; border-radius: 10px; border: 1px solid rgba(61, 126, 255, 0.25); background: rgba(61, 126, 255, 0.06); color: #c7d5eb; text-align: left; cursor: pointer; margin: 6px 0; font-size: 0.85rem; }
    .mcp-log-item:hover { background: rgba(61, 126, 255, 0.14); }
    .mcp-log-content { background: #0f1f33; border: 1px solid #233654; border-radius: 12px; padding: 12px; max-height: 280px; overflow: auto; white-space: pre-wrap; font-family: "Fira Code", Consolas, monospace; font-size: 0.8rem; color: #f4f6fb; }
    @media (max-width: 640px) {
      .container { padding: 24px 16px; }
      .header { flex-direction: column; align-items: stretch; }
      .settings-panel { padding: 16px; }
      .chat-log { min-height: 50vh; padding: 18px; }
      .chat-form { flex-direction: column; }
      .chat-form input, .chat-form button { width: 100%; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <h1 class="title">Sat.AI</h1>
        <p class="subtitle">Ask questions about the NOS3 documentation, compare answers across providers, and explore sourced references.</p>
      </div>
      <div class="control-bar">
        <button type="button" id="new-chat" class="button secondary">New Chat</button>
        <button type="button" id="toggle-settings" class="button secondary">Settings</button>
      </div>
    </div>
    <section id="settings-panel" class="settings-panel hidden">
      <div class="settings-header">
        <h2>Provider Settings</h2>
        <p>Enter API keys, fetch available models, and choose which model to use when chatting.</p>
      </div>
      <div id="settings-status" class="settings-status" role="status"></div>
      <div class="settings-grid">
        <div class="settings-card" data-provider="cerebras">
          <div class="settings-card-header">
            <span class="settings-card-title">Cerebras</span>
            <button type="button" class="button fetch-models" data-provider="cerebras">Refresh Models</button>
          </div>
          <div>
            <label class="settings-label" for="cerebras-api-key">API Key</label>
            <div class="input-row">
              <input id="cerebras-api-key" type="password" placeholder="sk-..." autocomplete="off" />
              <label class="settings-remember">
                <input type="checkbox" id="remember-cerebras" />
                Remember
              </label>
            </div>
          </div>
          <div>
            <label class="settings-label" for="cerebras-model">Model</label>
            <select id="cerebras-model" class="model-select">
              <option value="">Model list unavailable</option>
            </select>
          </div>
        </div>
        <div class="settings-card" data-provider="groq">
          <div class="settings-card-header">
            <span class="settings-card-title">Groq</span>
            <button type="button" class="button fetch-models" data-provider="groq">Refresh Models</button>
          </div>
          <div>
            <label class="settings-label" for="groq-api-key">API Key</label>
            <div class="input-row">
              <input id="groq-api-key" type="password" placeholder="gsk-..." autocomplete="off" />
              <label class="settings-remember">
                <input type="checkbox" id="remember-groq" />
                Remember
              </label>
            </div>
          </div>
          <div>
            <label class="settings-label" for="groq-model">Model</label>
            <select id="groq-model" class="model-select">
              <option value="">Model list unavailable</option>
            </select>
          </div>
        </div>
        <div class="settings-card" data-provider="sambanova">
          <div class="settings-card-header">
            <span class="settings-card-title">SambaNova</span>
            <button type="button" class="button fetch-models" data-provider="sambanova">Refresh Models</button>
          </div>
          <div>
            <label class="settings-label" for="sambanova-api-key">API Key</label>
            <div class="input-row">
              <input id="sambanova-api-key" type="password" placeholder="sn-..." autocomplete="off" />
              <label class="settings-remember">
                <input type="checkbox" id="remember-sambanova" />
                Remember
              </label>
            </div>
          </div>
          <div>
            <label class="settings-label" for="sambanova-model">Model</label>
            <select id="sambanova-model" class="model-select">
              <option value="">Model list unavailable</option>
            </select>
          </div>
        </div>

        <div class="settings-card" data-provider="embeddings">
          <div class="settings-card-header">
            <span class="settings-card-title">Embeddings (Ollama)</span>
            <button type="button" class="button" id="apply-embeddings">Apply &amp; Rebuild</button>
          </div>
          <div>
            <label class="settings-label" for="ollama-url">Ollama URL</label>
            <div class="input-row">
              <input id="ollama-url" type="text" placeholder="http://localhost:11434" autocomplete="off" />
            </div>
          </div>
          <div>
            <label class="settings-label" for="embedding-model">Embedding Model</label>
            <div class="input-row">
              <input id="embedding-model" type="text" placeholder="embeddinggemma:latest" autocomplete="off" />
            </div>
          </div>
        </div>

        <div class="settings-card" data-provider="mcp-orbit">
          <div class="settings-card-header">
            <span class="settings-card-title">MCP (Orbit Sim)</span>
            <div class="button-row">
              <button type="button" class="button secondary" id="stop-mcp-orbit">Stop Orbit</button>
              <button type="button" class="button" id="test-mcp-orbit">Test &amp; List Tools</button>
            </div>
          </div>
          <div>
            <label class="settings-label" for="mcp-orbit-server">Server Script</label>
            <div class="input-row">
              <input id="mcp-orbit-server" type="text" placeholder="SAT_Orbit_Sim_MCP/satellite_server.py" autocomplete="off" />
            </div>
          </div>
          <div>
            <label class="settings-label" for="mcp-orbit-python">Python Executable</label>
            <div class="input-row">
              <input id="mcp-orbit-python" type="text" placeholder="python3" autocomplete="off" />
            </div>
          </div>
          <div>
            <label class="settings-label" for="mcp-orbit-protocol">Protocol Version</label>
            <div class="input-row">
              <input id="mcp-orbit-protocol" type="text" placeholder="2024-11-05" autocomplete="off" />
            </div>
          </div>
          <div>
            <label class="settings-label" for="mcp-orbit-framing">Message Framing</label>
            <div class="input-row">
              <select id="mcp-orbit-framing" class="model-select">
                <option value="ndjson">ndjson (FastMCP)</option>
                <option value="lsp">lsp (Content-Length)</option>
              </select>
            </div>
          </div>
          <div>
            <label class="settings-label" for="mcp-orbit-timeout">Init Timeout (seconds)</label>
            <div class="input-row">
              <input id="mcp-orbit-timeout" type="number" step="0.1" min="0" placeholder="30" autocomplete="off" />
              <button type="button" class="button secondary" id="save-mcp-orbit">Save</button>
            </div>
          </div>
          <div id="mcp-orbit-tools" class="mcp-tools"></div>
          <div id="mcp-orbit-sim-status" class="mcp-log-path"></div>
          <div class="settings-card-header mcp-log-header">
            <span class="settings-card-title">MCP Logs</span>
            <button type="button" class="button secondary" id="refresh-mcp-orbit-logs">Refresh</button>
          </div>
          <div id="mcp-orbit-log-path" class="mcp-log-path"></div>
          <div id="mcp-orbit-logs" class="mcp-log-list"></div>
          <pre id="mcp-orbit-log-content" class="mcp-log-content"></pre>
        </div>
      </div>
    </section>
    <div id="chat-log" class="chat-log"></div>
    <form id="chat-form" class="chat-form">
      <input id="message-input" type="text" placeholder="Ask a question..." autocomplete="off" />
      <button type="submit">Send</button>
    </form>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/dompurify@3.0.6/dist/purify.min.js" crossorigin="anonymous"></script>
  <script>
    (() => {
      const form = document.getElementById("chat-form");
      const input = document.getElementById("message-input");
      const log = document.getElementById("chat-log");
      const newChatButton = document.getElementById("new-chat");
      const toggleSettingsButton = document.getElementById("toggle-settings");
      const settingsPanel = document.getElementById("settings-panel");
      const settingsStatus = document.getElementById("settings-status");
      let history = [];
      let chatLoading = false;

      const PROVIDERS = [
        { id: "cerebras", label: "Cerebras", placeholder: "sk-..." },
        { id: "groq", label: "Groq", placeholder: "gsk-..." },
        { id: "sambanova", label: "SambaNova", placeholder: "sn-..." },
      ];

      const providerState = {};
      const embeddingsState = {
        urlInput: null,
        modelInput: null,
        applyButton: null,
        loading: false,
      };
      const orbitMcpState = {
        serverInput: null,
        pythonInput: null,
        protocolInput: null,
        framingSelect: null,
        timeoutInput: null,
        saveButton: null,
        testButton: null,
        stopButton: null,
        toolsBox: null,
        simStatusBox: null,
        logPathBox: null,
        logsBox: null,
        logContentBox: null,
        refreshLogsButton: null,
        loading: false,
      };

      const orbitMcpStorage = {
        server: "nos3-rag-mcp-orbit-server",
        python: "nos3-rag-mcp-orbit-python",
        protocol: "nos3-rag-mcp-orbit-protocol",
        framing: "nos3-rag-mcp-orbit-framing",
        timeout: "nos3-rag-mcp-orbit-timeout",
      };

      const storageKeys = (providerId) => ({
        key: `nos3-rag-${providerId}-key`,
        remember: `nos3-rag-remember-${providerId}`,
        model: `nos3-rag-${providerId}-model`,
      });

      const updateSettingsStatus = (message, isError = false) => {
        if (!settingsStatus) return;
        settingsStatus.textContent = message || "";
        settingsStatus.classList.toggle("error", Boolean(message && isError));
      };

      const renderMarkdown = (raw) => {
        const text = typeof raw === "string" ? raw.trim() : "";
        if (!text) return "";

        if (window.marked && window.DOMPurify) {
          marked.setOptions({ breaks: true, gfm: true, mangle: false, headerIds: false });
          const html = marked.parse(text);
          return DOMPurify.sanitize(html, { ADD_ATTR: ["target", "rel"] });
        }

        // Fallback if libs failed to load
        const NL = String.fromCharCode(10);
        return text.split(NL).join("<br>");
      };

      const appendMessage = (role, content) => {
        const wrapper = document.createElement("div");
        wrapper.className = "message " + role;

        const header = document.createElement("div");
        header.className = "role";
        header.textContent = role === "user" ? "You" : "Assistant";

        const body = document.createElement("div");
        body.className = "content";
        body.innerHTML = renderMarkdown(content);

        wrapper.appendChild(header);
        wrapper.appendChild(body);
        log.appendChild(wrapper);
        log.scrollTop = log.scrollHeight;
        return wrapper;
      };

      const appendSources = (sources) => {
        if (!sources || !sources.length) return;

        const wrapper = document.createElement("div");
        wrapper.className = "sources";

        const label = document.createElement("div");
        label.className = "label";
        label.textContent = "Sources";
        wrapper.appendChild(label);

        const list = document.createElement("ul");
        for (const source of sources) {
          const item = document.createElement("li");

          // Use startsWith instead of a regex (prevents Python escape issues)
          const isHttpUrl =
            typeof source === "string" &&
            (source.startsWith("http://") || source.startsWith("https://"));

          if (isHttpUrl) {
            const link = document.createElement("a");
            link.href = source;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.textContent = source;
            item.appendChild(link);
          } else {
            item.textContent = source;
          }

          list.appendChild(item);
        }

        wrapper.appendChild(list);
        log.appendChild(wrapper);
        log.scrollTop = log.scrollHeight;
      };

      const appendProviderTag = (node, provider, model) => {
        if (!node || !provider) return;
        const badge = document.createElement("div");
        badge.className = "provider-tag";
        badge.textContent = model ? `${provider} · ${model}` : `Served by ${provider}`;
        node.appendChild(badge);
      };

      const refreshProviderDisabled = () => {
        PROVIDERS.forEach(({ id }) => {
          const state = providerState[id];
          if (!state) return;
          const disabled = chatLoading || state.loading;
          state.keyInput.disabled = disabled;
          state.remember.disabled = disabled;
          state.modelSelect.disabled = disabled || !state.models.length;
          state.fetchButton.disabled = disabled || !state.keyInput.value.trim();
        });
        const embeddingDisabled = chatLoading || embeddingsState.loading;
        if (embeddingsState.urlInput) embeddingsState.urlInput.disabled = embeddingDisabled;
        if (embeddingsState.modelInput) embeddingsState.modelInput.disabled = embeddingDisabled;
        if (embeddingsState.applyButton) embeddingsState.applyButton.disabled = embeddingDisabled;

        const orbitDisabled = chatLoading || orbitMcpState.loading;
        if (orbitMcpState.serverInput) orbitMcpState.serverInput.disabled = orbitDisabled;
        if (orbitMcpState.pythonInput) orbitMcpState.pythonInput.disabled = orbitDisabled;
        if (orbitMcpState.protocolInput) orbitMcpState.protocolInput.disabled = orbitDisabled;
        if (orbitMcpState.framingSelect) orbitMcpState.framingSelect.disabled = orbitDisabled;
        if (orbitMcpState.timeoutInput) orbitMcpState.timeoutInput.disabled = orbitDisabled;
        if (orbitMcpState.saveButton) orbitMcpState.saveButton.disabled = orbitDisabled;
        if (orbitMcpState.testButton) orbitMcpState.testButton.disabled = orbitDisabled;
        if (orbitMcpState.stopButton) orbitMcpState.stopButton.disabled = orbitDisabled;
        if (orbitMcpState.refreshLogsButton) orbitMcpState.refreshLogsButton.disabled = orbitDisabled;
      };

      const setLoading = (loading) => {
        chatLoading = loading;
        input.disabled = loading;
        const submitButton = form.querySelector("button");
        if (submitButton) submitButton.disabled = loading;
        newChatButton.disabled = loading;
        if (toggleSettingsButton) toggleSettingsButton.disabled = loading;
        refreshProviderDisabled();
      };

      const persistKey = (providerId) => {
        const state = providerState[providerId];
        if (!state) return;
        const keys = storageKeys(providerId);
        if (state.remember.checked) {
          localStorage.setItem(keys.remember, "true");
          localStorage.setItem(keys.key, state.keyInput.value);
        } else {
          localStorage.removeItem(keys.remember);
          localStorage.removeItem(keys.key);
        }
      };

      const persistModel = (providerId) => {
        const state = providerState[providerId];
        if (!state) return;
        const keys = storageKeys(providerId);
        const value = state.modelSelect.value || "";
        if (value) {
          localStorage.setItem(keys.model, value);
        } else {
          localStorage.removeItem(keys.model);
        }
      };

      const persistOrbitMcp = () => {
        if (!orbitMcpState.serverInput || !orbitMcpState.pythonInput || !orbitMcpState.protocolInput || !orbitMcpState.framingSelect || !orbitMcpState.timeoutInput) {
          return;
        }

        const server = (orbitMcpState.serverInput.value || "").trim();
        const python = (orbitMcpState.pythonInput.value || "").trim();
        const protocol = (orbitMcpState.protocolInput.value || "").trim();
        const framing = (orbitMcpState.framingSelect.value || "").trim();
        const timeout = (orbitMcpState.timeoutInput.value || "").trim();

        if (server) localStorage.setItem(orbitMcpStorage.server, server);
        else localStorage.removeItem(orbitMcpStorage.server);

        if (python) localStorage.setItem(orbitMcpStorage.python, python);
        else localStorage.removeItem(orbitMcpStorage.python);

        if (protocol) localStorage.setItem(orbitMcpStorage.protocol, protocol);
        else localStorage.removeItem(orbitMcpStorage.protocol);

        if (framing) localStorage.setItem(orbitMcpStorage.framing, framing);
        else localStorage.removeItem(orbitMcpStorage.framing);

        if (timeout) localStorage.setItem(orbitMcpStorage.timeout, timeout);
        else localStorage.removeItem(orbitMcpStorage.timeout);
      };

      const syncStoredSettings = () => {
        PROVIDERS.forEach(({ id }) => {
          persistKey(id);
          persistModel(id);
        });
        persistOrbitMcp();
      };

      const setProviderModels = (providerId, models, selected) => {
        const state = providerState[providerId];
        if (!state) return;
        const seen = new Set();
        const unique = [];
        (models || []).forEach((model) => {
          if (!model || seen.has(model)) return;
          seen.add(model);
          unique.push(model);
        });
        state.models = unique;
        state.modelSelect.innerHTML = "";
        if (!unique.length) {
          const option = document.createElement("option");
          option.value = "";
          option.textContent = "Model list unavailable";
          option.disabled = true;
          option.selected = true;
          state.modelSelect.appendChild(option);
        } else {
          unique.forEach((model) => {
            const option = document.createElement("option");
            option.value = model;
            option.textContent = model;
            state.modelSelect.appendChild(option);
          });
          const storedModel = localStorage.getItem(storageKeys(providerId).model);
          const preferred = storedModel || selected;
          if (preferred && seen.has(preferred)) {
            state.modelSelect.value = preferred;
          } else {
            state.modelSelect.value = unique[0];
          }
          persistModel(providerId);
        }
        refreshProviderDisabled();
      };

      const handleProviderModelsResponse = (providerId, data) => {
        if (!data || !Array.isArray(data.models)) {
          updateSettingsStatus(`No models returned for ${providerId}.`, true);
          return;
        }
        setProviderModels(providerId, data.models, data.selected || "");
        updateSettingsStatus(`Loaded ${data.models.length} models for ${providerId}.`);
      };

      const fetchProviderModels = async (providerId) => {
        const state = providerState[providerId];
        if (!state) return;
        const key = state.keyInput.value.trim();
        if (!key) {
          updateSettingsStatus(`Enter an API key for ${providerId} before refreshing models.`, true);
          return;
        }
        state.loading = true;
        refreshProviderDisabled();
        updateSettingsStatus(`Refreshing ${providerId} models...`);
        try {
          const response = await fetch("/api/models", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ provider: providerId, api_key: key }),
          });
          if (!response.ok) {
            let detail = `Failed to refresh ${providerId} models.`;
            try {
              const error = await response.json();
              detail = error.detail || detail;
            } catch (_) {}
            updateSettingsStatus(detail, true);
            return;
          }
          const payload = await response.json();
          handleProviderModelsResponse(providerId, payload);
        } catch (error) {
          console.error(error);
          updateSettingsStatus(`Error refreshing ${providerId} models. See console for details.`, true);
        } finally {
          state.loading = false;
          refreshProviderDisabled();
        }
      };

      const initProvider = ({ id, placeholder }) => {
        const state = {
          keyInput: document.getElementById(`${id}-api-key`),
          modelSelect: document.getElementById(`${id}-model`),
          remember: document.getElementById(`remember-${id}`),
          fetchButton: document.querySelector(`.fetch-models[data-provider="${id}"]`),
          models: [],
          loading: false,
        };
        if (!state.keyInput || !state.modelSelect || !state.remember || !state.fetchButton) {
          console.warn(`Missing settings elements for provider ${id}`);
          return;
        }
        if (placeholder) state.keyInput.placeholder = placeholder;

        providerState[id] = state;

        const keys = storageKeys(id);
        if (localStorage.getItem(keys.remember) === "true") {
          state.remember.checked = true;
          state.keyInput.value = localStorage.getItem(keys.key) || "";
        }
        const storedModel = localStorage.getItem(keys.model);
        if (storedModel) setProviderModels(id, [storedModel], storedModel);

        state.keyInput.addEventListener("input", () => {
          persistKey(id);
          refreshProviderDisabled();
        });
        state.remember.addEventListener("change", () => {
          persistKey(id);
          refreshProviderDisabled();
        });
        state.modelSelect.addEventListener("change", () => persistModel(id));
        state.fetchButton.addEventListener("click", () => fetchProviderModels(id));
      };

      const initEmbeddings = () => {
        embeddingsState.urlInput = document.getElementById("ollama-url");
        embeddingsState.modelInput = document.getElementById("embedding-model");
        embeddingsState.applyButton = document.getElementById("apply-embeddings");

        if (!embeddingsState.urlInput || !embeddingsState.modelInput || !embeddingsState.applyButton) {
          console.warn("Embedding configuration elements missing from the page.");
          return;
        }

        embeddingsState.applyButton.addEventListener("click", async () => {
          const url = (embeddingsState.urlInput.value || "").trim();
          const model = (embeddingsState.modelInput.value || "").trim();
          if (!url || !model) {
            updateSettingsStatus("Provide both an Ollama URL and an embedding model.", true);
            return;
          }
          embeddingsState.loading = true;
          refreshProviderDisabled();
          updateSettingsStatus("Updating embeddings and rebuilding knowledge base...");
          try {
            const response = await fetch("/api/embeddings", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ ollama_url: url, embedding_model: model, rebuild: true }),
            });
            if (!response.ok) {
              let detail = "Failed to apply embedding settings.";
              try {
                const err = await response.json();
                detail = err.detail || detail;
              } catch (_) {}
              updateSettingsStatus(detail, true);
              return;
            }
            const payload = await response.json();
            updateSettingsStatus(`Embeddings updated. Model: ${payload.embedding_model}`);
          } catch (error) {
            console.error(error);
            updateSettingsStatus("Error applying embeddings. See console for details.", true);
          } finally {
            embeddingsState.loading = false;
            refreshProviderDisabled();
          }
        });
      };

      const renderOrbitToolList = (payload) => {
        if (!orbitMcpState.toolsBox) return;
        orbitMcpState.toolsBox.innerHTML = "";

        const tools = payload && Array.isArray(payload.tools) ? payload.tools : [];
        const serverInfo = payload && payload.server_info ? payload.server_info : null;
        const protocolVersion = payload && payload.protocol_version ? payload.protocol_version : null;

        if (serverInfo && (serverInfo.name || serverInfo.version || protocolVersion)) {
          const header = document.createElement("div");
          const name = serverInfo.name ? String(serverInfo.name) : "MCP server";
          const version = serverInfo.version ? String(serverInfo.version) : "";
          const proto = protocolVersion ? ` · protocol ${protocolVersion}` : "";
          header.textContent = version ? `${name} ${version}${proto}` : `${name}${proto}`;
          orbitMcpState.toolsBox.appendChild(header);
        }

        if (!tools.length) {
          const empty = document.createElement("div");
          empty.textContent = "No tools returned.";
          orbitMcpState.toolsBox.appendChild(empty);
          return;
        }

        const list = document.createElement("ul");
        tools.forEach((tool) => {
          if (!tool || !tool.name) return;
          const item = document.createElement("li");
          const name = String(tool.name);
          const desc = tool.description ? String(tool.description) : "";
          item.textContent = desc ? `${name} — ${desc}` : name;
          list.appendChild(item);
        });
        orbitMcpState.toolsBox.appendChild(list);
      };

      const renderOrbitLogPath = (path) => {
        if (!orbitMcpState.logPathBox) return;
        orbitMcpState.logPathBox.textContent = path ? `Last log: ${path}` : "";
      };

      const renderOrbitSimStatus = (payload) => {
        if (!orbitMcpState.simStatusBox) return;
        const running = payload && payload.running;
        if (!running) {
          orbitMcpState.simStatusBox.textContent = "Orbit sim: idle";
          return;
        }
        const pid = payload && payload.pid ? ` (pid ${payload.pid})` : "";
        const logPath = payload && payload.log_path ? ` · log: ${payload.log_path}` : "";
        const runDir = payload && payload.run_dir ? ` · run: ${payload.run_dir}` : "";
        orbitMcpState.simStatusBox.textContent = `Orbit sim: RUNNING${pid}${logPath}${runDir}`;
      };

      const formatBytes = (bytes) => {
        const value = Number(bytes) || 0;
        if (value < 1024) return `${value} B`;
        if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
        return `${(value / (1024 * 1024)).toFixed(1)} MB`;
      };

      const renderOrbitLogList = (payload) => {
        if (!orbitMcpState.logsBox) return;
        orbitMcpState.logsBox.innerHTML = "";
        const logs = payload && Array.isArray(payload.logs) ? payload.logs : [];
        if (!logs.length) {
          const empty = document.createElement("div");
          empty.textContent = "No log files yet.";
          orbitMcpState.logsBox.appendChild(empty);
          return;
        }

        logs.forEach((entry) => {
          if (!entry || !entry.name) return;
          const button = document.createElement("button");
          button.type = "button";
          button.className = "mcp-log-item";
          const stamp = entry.mtime_s ? new Date(entry.mtime_s * 1000).toLocaleString() : "";
          const size = entry.size_bytes != null ? formatBytes(entry.size_bytes) : "";
          button.textContent = stamp && size ? `${entry.name} (${size}, ${stamp})` : entry.name;
          button.addEventListener("click", () => fetchOrbitLogContent(entry.name));
          orbitMcpState.logsBox.appendChild(button);
        });
      };

      const fetchOrbitLogContent = async (name) => {
        if (!orbitMcpState.logContentBox) return;
        const logName = String(name || "").trim();
        if (!logName) return;
        updateSettingsStatus(`Loading log: ${logName}...`);
        orbitMcpState.logContentBox.textContent = "";
        try {
          const response = await fetch(`/api/mcp/orbit/logs/${encodeURIComponent(logName)}?max_bytes=200000`);
          if (!response.ok) {
            let detail = "Failed to load log.";
            try {
              const err = await response.json();
              detail = err.detail || detail;
            } catch (_) {}
            updateSettingsStatus(detail, true);
            return;
          }
          const data = await response.json();
          const header = data && data.truncated ? `...(truncated; total ${formatBytes(data.total_bytes)})\n\n` : "";
          orbitMcpState.logContentBox.textContent = header + (data && data.content ? String(data.content) : "");
          updateSettingsStatus(`Loaded log: ${logName}`);
          if (data && data.path) renderOrbitLogPath(data.path);
        } catch (error) {
          console.error(error);
          updateSettingsStatus("Error loading log. See console for details.", true);
        }
      };

      const fetchOrbitLogs = async () => {
        if (!orbitMcpState.logsBox) return;
        orbitMcpState.logsBox.textContent = "Loading logs...";
        try {
          const response = await fetch("/api/mcp/orbit/logs?limit=50");
          if (!response.ok) {
            orbitMcpState.logsBox.textContent = "Failed to load logs.";
            return;
          }
          const data = await response.json();
          renderOrbitLogList(data);
        } catch (error) {
          console.warn("Failed to fetch orbit MCP logs.", error);
          orbitMcpState.logsBox.textContent = "Failed to load logs.";
        }
      };

      const fetchOrbitSimStatus = async () => {
        try {
          const response = await fetch("/api/mcp/orbit/status");
          if (!response.ok) return;
          const data = await response.json();
          renderOrbitSimStatus(data);
        } catch (error) {
          console.warn("Failed to fetch orbit simulation status.", error);
        }
      };

      const stopOrbitSim = async () => {
        if (!orbitMcpState.stopButton) return;
        orbitMcpState.loading = true;
        refreshProviderDisabled();
        updateSettingsStatus("Stopping orbit simulation...");
        try {
          const response = await fetch("/api/mcp/orbit/stop", { method: "POST" });
          if (!response.ok) {
            let detail = "Failed to stop orbit simulation.";
            try {
              const err = await response.json();
              detail = err.detail || detail;
            } catch (_) {}
            updateSettingsStatus(detail, true);
            return;
          }
          const data = await response.json();
          if (data && data.status) renderOrbitSimStatus(data.status);
          fetchOrbitLogs();
          updateSettingsStatus(data && data.stop_requested ? "Stop requested." : "No orbit simulation was running.");
        } catch (error) {
          console.error(error);
          updateSettingsStatus("Error stopping orbit simulation. See console for details.", true);
        } finally {
          orbitMcpState.loading = false;
          refreshProviderDisabled();
        }
      };

      const orbitMcpPayloadFromInputs = () => {
        const server = orbitMcpState.serverInput ? (orbitMcpState.serverInput.value || "").trim() : "";
        const python = orbitMcpState.pythonInput ? (orbitMcpState.pythonInput.value || "").trim() : "";
        const protocol = orbitMcpState.protocolInput ? (orbitMcpState.protocolInput.value || "").trim() : "";
        const framing = orbitMcpState.framingSelect ? (orbitMcpState.framingSelect.value || "").trim() : "ndjson";
        const timeoutRaw = orbitMcpState.timeoutInput ? (orbitMcpState.timeoutInput.value || "").trim() : "";
        const timeout = timeoutRaw ? Number(timeoutRaw) : 30;
        return {
          server_script: server,
          python_executable: python,
          protocol_version: protocol || "2024-11-05",
          init_timeout_s: Number.isFinite(timeout) ? timeout : 30,
          framing: framing || "ndjson",
        };
      };

      const saveOrbitMcp = async () => {
        if (!orbitMcpState.serverInput || !orbitMcpState.pythonInput || !orbitMcpState.saveButton) return;
        const payload = orbitMcpPayloadFromInputs();
        if (!payload.server_script || !payload.python_executable) {
          updateSettingsStatus("Provide both a server script and python executable.", true);
          return;
        }
        orbitMcpState.loading = true;
        refreshProviderDisabled();
        updateSettingsStatus("Saving MCP settings...");
        try {
          const response = await fetch("/api/mcp/orbit", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          if (!response.ok) {
            let detail = "Failed to save MCP settings.";
            try {
              const err = await response.json();
              detail = err.detail || detail;
            } catch (_) {}
            updateSettingsStatus(detail, true);
            return;
          }
          const data = await response.json();
          if (orbitMcpState.serverInput && data.server_script) orbitMcpState.serverInput.value = data.server_script;
          if (orbitMcpState.pythonInput && data.python_executable) orbitMcpState.pythonInput.value = data.python_executable;
          if (orbitMcpState.protocolInput && data.protocol_version) orbitMcpState.protocolInput.value = data.protocol_version;
          if (orbitMcpState.framingSelect && data.framing) orbitMcpState.framingSelect.value = data.framing;
          if (orbitMcpState.timeoutInput && data.init_timeout_s != null) orbitMcpState.timeoutInput.value = String(data.init_timeout_s);
          persistOrbitMcp();
          updateSettingsStatus("MCP settings saved.");
        } catch (error) {
          console.error(error);
          updateSettingsStatus("Error saving MCP settings. See console for details.", true);
        } finally {
          orbitMcpState.loading = false;
          refreshProviderDisabled();
        }
      };

      const testOrbitMcp = async () => {
        if (!orbitMcpState.serverInput || !orbitMcpState.pythonInput || !orbitMcpState.testButton) return;
        const payload = orbitMcpPayloadFromInputs();
        if (!payload.server_script || !payload.python_executable) {
          updateSettingsStatus("Provide both a server script and python executable.", true);
          return;
        }
        orbitMcpState.loading = true;
        refreshProviderDisabled();
        updateSettingsStatus("Testing MCP server...");
        if (orbitMcpState.toolsBox) orbitMcpState.toolsBox.textContent = "";
        try {
          const response = await fetch("/api/mcp/orbit/test", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          if (!response.ok) {
            let detail = "MCP test failed.";
            try {
              const err = await response.json();
              detail = err.detail || detail;
            } catch (_) {}
            updateSettingsStatus(detail, true);
            return;
          }
          const data = await response.json();
          renderOrbitToolList(data);
          if (data && data.log_path) renderOrbitLogPath(String(data.log_path));
          fetchOrbitLogs();
          fetchOrbitSimStatus();
          if (data && data.log_path) {
            const logName = String(data.log_path).split("/").pop();
            if (logName) fetchOrbitLogContent(logName);
          }
          updateSettingsStatus(`MCP connected. Tools: ${(data.tools || []).length}`);
        } catch (error) {
          console.error(error);
          updateSettingsStatus("Error testing MCP server. See console for details.", true);
        } finally {
          orbitMcpState.loading = false;
          refreshProviderDisabled();
        }
      };

      const initOrbitMcp = () => {
        orbitMcpState.serverInput = document.getElementById("mcp-orbit-server");
        orbitMcpState.pythonInput = document.getElementById("mcp-orbit-python");
        orbitMcpState.protocolInput = document.getElementById("mcp-orbit-protocol");
        orbitMcpState.framingSelect = document.getElementById("mcp-orbit-framing");
        orbitMcpState.timeoutInput = document.getElementById("mcp-orbit-timeout");
        orbitMcpState.saveButton = document.getElementById("save-mcp-orbit");
        orbitMcpState.testButton = document.getElementById("test-mcp-orbit");
        orbitMcpState.stopButton = document.getElementById("stop-mcp-orbit");
        orbitMcpState.toolsBox = document.getElementById("mcp-orbit-tools");
        orbitMcpState.simStatusBox = document.getElementById("mcp-orbit-sim-status");
        orbitMcpState.logPathBox = document.getElementById("mcp-orbit-log-path");
        orbitMcpState.logsBox = document.getElementById("mcp-orbit-logs");
        orbitMcpState.logContentBox = document.getElementById("mcp-orbit-log-content");
        orbitMcpState.refreshLogsButton = document.getElementById("refresh-mcp-orbit-logs");

        if (!orbitMcpState.serverInput || !orbitMcpState.pythonInput || !orbitMcpState.protocolInput || !orbitMcpState.framingSelect || !orbitMcpState.timeoutInput || !orbitMcpState.saveButton || !orbitMcpState.testButton) {
          console.warn("Orbit MCP configuration elements missing from the page.");
          return;
        }

        const storedServer = localStorage.getItem(orbitMcpStorage.server);
        const storedPython = localStorage.getItem(orbitMcpStorage.python);
        const storedProtocol = localStorage.getItem(orbitMcpStorage.protocol);
        const storedFraming = localStorage.getItem(orbitMcpStorage.framing);
        const storedTimeout = localStorage.getItem(orbitMcpStorage.timeout);
        if (storedServer) orbitMcpState.serverInput.value = storedServer;
        if (storedPython) orbitMcpState.pythonInput.value = storedPython;
        if (storedProtocol) orbitMcpState.protocolInput.value = storedProtocol;
        if (storedFraming) orbitMcpState.framingSelect.value = storedFraming;
        if (storedTimeout) orbitMcpState.timeoutInput.value = storedTimeout;

        orbitMcpState.serverInput.addEventListener("input", () => {
          persistOrbitMcp();
          refreshProviderDisabled();
        });
        orbitMcpState.pythonInput.addEventListener("input", () => {
          persistOrbitMcp();
          refreshProviderDisabled();
        });
        orbitMcpState.protocolInput.addEventListener("input", () => {
          persistOrbitMcp();
          refreshProviderDisabled();
        });
        orbitMcpState.framingSelect.addEventListener("change", () => {
          persistOrbitMcp();
          refreshProviderDisabled();
        });
        orbitMcpState.timeoutInput.addEventListener("input", () => {
          persistOrbitMcp();
          refreshProviderDisabled();
        });

        orbitMcpState.saveButton.addEventListener("click", () => saveOrbitMcp());
        orbitMcpState.testButton.addEventListener("click", () => testOrbitMcp());
        if (orbitMcpState.refreshLogsButton) orbitMcpState.refreshLogsButton.addEventListener("click", () => fetchOrbitLogs());
        if (orbitMcpState.stopButton) orbitMcpState.stopButton.addEventListener("click", () => stopOrbitSim());
        fetchOrbitLogs();
        fetchOrbitSimStatus();
      };

      const loadInitialEmbeddings = async () => {
        try {
          const response = await fetch("/api/embeddings");
          if (!response.ok) return;
          const payload = await response.json();
          if (embeddingsState.urlInput && payload.ollama_url) {
            embeddingsState.urlInput.value = payload.ollama_url;
          }
          if (embeddingsState.modelInput && payload.embedding_model) {
            embeddingsState.modelInput.value = payload.embedding_model;
          }
        } catch (error) {
          console.warn("Failed to load initial embedding configuration.", error);
        }
      };

      const loadInitialOrbitMcp = async () => {
        try {
          const response = await fetch("/api/mcp/orbit");
          if (!response.ok) return;
          const payload = await response.json();
          if (!payload) return;

          if (orbitMcpState.serverInput && payload.server_script && !localStorage.getItem(orbitMcpStorage.server)) {
            orbitMcpState.serverInput.value = payload.server_script;
          }
          if (orbitMcpState.pythonInput && payload.python_executable && !localStorage.getItem(orbitMcpStorage.python)) {
            orbitMcpState.pythonInput.value = payload.python_executable;
          }
          if (orbitMcpState.protocolInput && payload.protocol_version && !localStorage.getItem(orbitMcpStorage.protocol)) {
            orbitMcpState.protocolInput.value = payload.protocol_version;
          }
          if (orbitMcpState.framingSelect && payload.framing && !localStorage.getItem(orbitMcpStorage.framing)) {
            orbitMcpState.framingSelect.value = payload.framing;
          }
          if (orbitMcpState.timeoutInput && payload.init_timeout_s != null && !localStorage.getItem(orbitMcpStorage.timeout)) {
            orbitMcpState.timeoutInput.value = String(payload.init_timeout_s);
          }
        } catch (error) {
          console.warn("Failed to load initial MCP configuration.", error);
        }
      };

      PROVIDERS.forEach(initProvider);
      initEmbeddings();
      initOrbitMcp();
      loadInitialEmbeddings();
      loadInitialOrbitMcp();
      refreshProviderDisabled();

      const toggleSettings = () => {
        settingsPanel.classList.toggle("hidden");
        if (toggleSettingsButton) {
          toggleSettingsButton.textContent = settingsPanel.classList.contains("hidden") ? "Settings" : "Hide Settings";
        }
      };

      if (toggleSettingsButton) {
        toggleSettingsButton.addEventListener("click", toggleSettings);
      }

      const loadInitialModelSummary = async () => {
        try {
          const response = await fetch("/api/models");
          if (!response.ok) return;
          const payload = await response.json();
          if (!payload || !Array.isArray(payload.providers)) return;
          payload.providers.forEach((entry) => {
            const providerId = entry.provider;
            if (!providerState[providerId]) return;
            setProviderModels(providerId, entry.models || [], entry.selected || "");
          });
          updateSettingsStatus("");
        } catch (error) {
          console.warn("Failed to load initial model lists.", error);
        }
      };

      loadInitialModelSummary();

      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const message = input.value.trim();
        if (!message) return;

        const providerKeysPayload = {};
        const providerModelsPayload = {};
        PROVIDERS.forEach(({ id }) => {
          const state = providerState[id];
          if (!state) return;
          const keyValue = state.keyInput.value.trim();
          if (keyValue) providerKeysPayload[id] = keyValue;
          const modelValue = state.modelSelect.value.trim();
          if (modelValue) providerModelsPayload[id] = modelValue;
        });

        appendMessage("user", message);
        setLoading(true);
        input.value = "";

        try {
          syncStoredSettings();
          const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              message,
              history,
              cerebras_api_key: providerKeysPayload.cerebras || null,
              groq_api_key: providerKeysPayload.groq || null,
              sambanova_api_key: providerKeysPayload.sambanova || null,
              provider_models: providerModelsPayload,
            })
          });

          if (!response.ok) {
            let detail = "Request failed.";
            try {
              const error = await response.json();
              detail = error.detail || detail;
            } catch (_) {}
            appendMessage("assistant", detail);
            updateSettingsStatus(detail, true);
            return;
          }

          const data = await response.json();
          history = data.history || history;
          const assistantNode = appendMessage("assistant", data.answer || "No answer returned.");
          appendProviderTag(assistantNode, data.provider, data.model);
          const providerKey = (data.provider_id || "").toLowerCase();
          if (providerKey && providerState[providerKey] && data.model) {
            const state = providerState[providerKey];
            if (!state.models.includes(data.model)) {
              setProviderModels(providerKey, [data.model, ...state.models], data.model);
            } else {
              state.modelSelect.value = data.model;
              persistModel(providerKey);
            }
          }
          appendSources(data.sources || []);
        } catch (error) {
          appendMessage("assistant", "Failed to reach the server. Check the console for details.");
          console.error(error);
        } finally {
          setLoading(false);
          input.focus();
        }
      });

      const resetChat = (message = "Sat.AI assistant ready. Ask me about the documentation (tip: /orbit <lat> <lon> launches the orbit visualization).") => {
        history = [];
        log.innerHTML = "";
        appendMessage("assistant", message);
        syncStoredSettings();
      };

      newChatButton.addEventListener("click", () => {
        resetChat();
      });

      resetChat();
    })();
  </script>
</body>
</html>
"""


class ChatTurn(BaseModel):
    question: str
    answer: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatTurn] = Field(default_factory=list)
    cerebras_api_key: str | None = None
    groq_api_key: str | None = None
    sambanova_api_key: str | None = None
    provider_models: Dict[str, str] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    history: List[ChatTurn] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    provider_id: str | None = None


class ProviderModelRequest(BaseModel):
    provider: str
    api_key: str


class ProviderModelResponse(BaseModel):
    provider: str
    models: List[str]
    selected: str | None = None


class ProviderModelsSummary(BaseModel):
    providers: List[ProviderModelResponse]


class EmbeddingSettings(BaseModel):
    ollama_url: str
    embedding_model: str


class EmbeddingSettingsRequest(BaseModel):
    ollama_url: Optional[str] = None
    embedding_model: Optional[str] = None
    rebuild: bool = True


class OrbitMcpSettings(BaseModel):
    python_executable: str = Field(..., min_length=1)
    server_script: str = Field(..., min_length=1)
    protocol_version: str = Field("2024-11-05", min_length=1)
    init_timeout_s: float = Field(30.0, ge=0.1, le=300.0)
    framing: str = Field("ndjson", min_length=1)


class OrbitMcpTestResponse(BaseModel):
    server_info: Dict[str, Any] = Field(default_factory=dict)
    protocol_version: str | None = None
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    log_path: str | None = None


class OrbitMcpLogEntry(BaseModel):
    name: str
    path: str
    size_bytes: int
    mtime_s: float


class OrbitMcpLogsResponse(BaseModel):
    directory: str
    logs: List[OrbitMcpLogEntry] = Field(default_factory=list)


class OrbitMcpLogContentResponse(BaseModel):
    name: str
    path: str
    truncated: bool = False
    total_bytes: int = 0
    content: str = ""


class OrbitMcpSimStatusResponse(BaseModel):
    running: bool = False
    pid: int | None = None
    log_path: str | None = None
    run_dir: str | None = None
    started_s: float | None = None


class OrbitMcpStopResponse(BaseModel):
    stop_requested: bool = False
    status: OrbitMcpSimStatusResponse = Field(default_factory=OrbitMcpSimStatusResponse)


class ModelSelectionError(Exception):
    def __init__(self, provider: str, attempts: List[Tuple[str, Exception]]):
        self.provider = provider
        self.attempts = attempts
        summary = ", ".join(model for model, _ in attempts) if attempts else "none"
        super().__init__(f"No supported model found for provider '{provider}' (tried: {summary}).")


def build_provider_chain(
    vector_store: Chroma,
    provider: str,
    model_candidates: List[str],
    api_key: str,
) -> tuple[ConversationalRetrievalChain, str]:
    if not api_key:
        raise EnvironmentError("Missing API key for provider initialization.")

    attempts: List[Tuple[str, Exception]] = []
    if not model_candidates:
        raise ModelSelectionError(provider, attempts)

    for model_name in model_candidates:
        try:
            if provider == "cerebras":
                chain = build_chat_chain(
                    vector_store,
                    model_name,
                    api_key=api_key,
                )
            elif provider == "groq":
                llm = ChatGroq(model_name=model_name, groq_api_key=api_key)
                retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    return_source_documents=True,
                    max_tokens_limit=4096,
                )
            elif provider == "sambanova":
                llm = ChatSambaNova(model=model_name, api_key=api_key)
                retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    return_source_documents=True,
                    max_tokens_limit=4096,
                )
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            return chain, model_name
        except Exception as exc:
            if is_model_error(exc):
                attempts.append((model_name, exc))
                continue
            raise

    raise ModelSelectionError(provider, attempts)


def is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if "429" in text or "too many request" in text or "rate limit" in text or "queue" in text:
        return True
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if status == 429:
        return True
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 429:
        return True
    return False


def is_auth_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(token in text for token in ("unauthorized", "invalid api key", "authentication", "forbidden"))


def is_model_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if "model" in text and ("not found" in text or "does not exist" in text):
        return True
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if status == 404:
        return True
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 404:
        return True
    return False


def _merge_candidates(*groups: Optional[List[str]]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for group in groups:
        if not group:
            continue
        for item in group:
            if not item or item in seen:
                continue
            seen.add(item)
            result.append(item)
    return result


def fetch_provider_models(provider: str, api_key: str) -> List[str]:
    provider = provider.lower()
    endpoint = PROVIDER_MODEL_ENDPOINTS.get(provider)
    if not endpoint:
        raise RuntimeError(f"Unsupported provider '{provider}'.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    try:
        response = requests.get(endpoint, headers=headers, timeout=20)
    except requests.RequestException as exc:  # pragma: no cover - network errors are runtime dependent
        raise RuntimeError(f"Failed to reach {provider} model endpoint: {exc}") from exc

    if response.status_code == 401:
        raise HTTPException(status_code=401, detail=f"{provider.title()} API key was rejected.")
    if response.status_code == 403:
        raise HTTPException(status_code=403, detail=f"{provider.title()} denied access to the model list.")
    if response.status_code >= 500:
        raise HTTPException(status_code=502, detail=f"{provider.title()} model endpoint returned {response.status_code}.")

    if response.status_code not in (200, 201):
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover - propagated to caller
            raise RuntimeError(f"{provider.title()} model endpoint returned {response.status_code}.") from exc

    try:
        payload = response.json()
    except ValueError as exc:  # pragma: no cover
        raise RuntimeError(f"{provider.title()} model response was not valid JSON.") from exc

    items: List[object]
    if isinstance(payload, dict):
        items = payload.get("data") or payload.get("models") or payload.get("items") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    models: List[str] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            model_id = item.get("id") or item.get("name") or item.get("model")
        else:
            model_id = str(item)
        if model_id and model_id not in seen:
            seen.add(model_id)
            models.append(model_id)

    if not models:
        # fallback: some APIs return a dict with nested data
        fallback = payload.get("model") if isinstance(payload, dict) else None
        if fallback:
            models.append(str(fallback))

    if not models:
        raise RuntimeError(f"No models returned by {provider.title()}.")

    return models


def build_chain(
    config: WebConfig,
) -> tuple[
    Chroma,
    ConversationalRetrievalChain | None,
    List[Tuple[str, str]],
    Dict[Tuple[str, str, str], ConversationalRetrievalChain],
    Dict[str, List[str]],
    Dict[str, str],
]:
    embeddings = OllamaEmbeddings(model=config.embedding_model, base_url=config.ollama_url)
    vector_store = get_vector_store(
        persist_dir=config.persist_dir,
        embeddings=embeddings,
        source_url=config.source_url,
        max_depth=config.max_depth,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        rebuild=config.rebuild,
    )

    config.cerebras_models = _merge_candidates([config.cerebras_model], config.cerebras_models, DEFAULT_MODEL_CANDIDATES["cerebras"])
    config.groq_models = _merge_candidates(config.groq_models, DEFAULT_MODEL_CANDIDATES["groq"])
    config.sambanova_models = _merge_candidates(config.sambanova_models, DEFAULT_MODEL_CANDIDATES["sambanova"])

    provider_catalog: Dict[str, List[str]] = {
        "cerebras": config.cerebras_models[:],
        "groq": config.groq_models[:],
        "sambanova": config.sambanova_models[:],
    }

    key_ring: List[Tuple[str, str]] = []
    cache: Dict[Tuple[str, str, str], ConversationalRetrievalChain] = {}
    selected_map: Dict[str, str] = {}
    default_chain: ConversationalRetrievalChain | None = None

    def prefetch(provider: str, api_key: Optional[str]) -> None:
        nonlocal default_chain
        if not api_key:
            return
        candidates = provider_catalog.get(provider) or DEFAULT_MODEL_CANDIDATES.get(provider, [])
        try:
            chain, model_used = build_provider_chain(vector_store, provider, candidates, api_key)
        except ModelSelectionError:
            return
        except Exception:
            return

        key_ring.append((provider, api_key))
        cache[(provider, api_key, model_used)] = chain
        selected_map[provider] = model_used
        if default_chain is None:
            default_chain = chain

    prefetch("cerebras", config.cerebras_api_key)
    prefetch("groq", config.groq_api_key)
    prefetch("sambanova", config.sambanova_api_key)

    return vector_store, default_chain, key_ring, cache, provider_catalog, selected_map


@asynccontextmanager
async def lifespan(app: FastAPI):
    config: WebConfig = app.state.config
    try:
        vector_store, default_chain, key_ring, initial_cache, provider_catalog, selected_map = build_chain(config)
        app.state.vector_store = vector_store
        app.state.default_chain = default_chain
        app.state.key_ring = key_ring
        app.state.chain_cache = dict(initial_cache)
        app.state.provider_catalog = {k: v[:] for k, v in provider_catalog.items()}
        app.state.selected_models = dict(selected_map)
        app.state.key_index = 0
        app.state.startup_error = None
    except Exception as exc:  # noqa: BLE001 - surface setup issues to clients
        app.state.vector_store = None
        app.state.default_chain = None
        app.state.key_ring = []
        app.state.chain_cache = {}
        app.state.provider_catalog = {k: DEFAULT_MODEL_CANDIDATES.get(k, [])[:] for k in DEFAULT_MODEL_CANDIDATES}
        app.state.selected_models = {}
        app.state.key_index = 0
        app.state.startup_error = str(exc)
    yield


def create_app(config: WebConfig) -> FastAPI:
    app = FastAPI(title="Sat.AI", version="0.1.0")

    app.state.config = config
    app.state.vector_store: Chroma | None = None
    app.state.default_chain: ConversationalRetrievalChain | None = None
    app.state.chain_cache: Dict[Tuple[str, str, str], ConversationalRetrievalChain] = {}
    app.state.key_ring: List[Tuple[str, str]] = []
    app.state.key_index: int = 0
    app.state.startup_error: str | None = None
    app.state.provider_catalog: Dict[str, List[str]] = {
        "cerebras": _merge_candidates(config.cerebras_models, DEFAULT_MODEL_CANDIDATES["cerebras"]),
        "groq": _merge_candidates(config.groq_models, DEFAULT_MODEL_CANDIDATES["groq"]),
        "sambanova": _merge_candidates(config.sambanova_models, DEFAULT_MODEL_CANDIDATES["sambanova"]),
    }
    app.state.selected_models: Dict[str, str] = {}
    app.router.lifespan_context = lifespan

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(content=HTML_PAGE)

    @app.get("/healthz")
    def health() -> dict[str, str]:
        if app.state.startup_error:
            return {"status": "error", "detail": app.state.startup_error}
        if app.state.vector_store is None:
            return {"status": "starting"}
        detail = None
        if not app.state.key_ring:
            detail = "Provide a Cerebras, Groq, or SambaNova API key to begin chatting."
        payload = {"status": "ready", "keys": len(app.state.key_ring)}
        if detail:
            payload["detail"] = detail
        return payload

    @app.get("/api/models", response_model=ProviderModelsSummary)
    def list_models() -> ProviderModelsSummary:
        providers: List[ProviderModelResponse] = []
        for provider in ("cerebras", "groq", "sambanova"):
            models = (app.state.provider_catalog.get(provider) or DEFAULT_MODEL_CANDIDATES.get(provider, [])).copy()
            selected = app.state.selected_models.get(provider)
            if not selected:
                if provider == "cerebras":
                    selected = app.state.config.cerebras_model
                elif provider == "groq":
                    selected = (app.state.config.groq_models[0] if app.state.config.groq_models else None)
                elif provider == "sambanova":
                    selected = (app.state.config.sambanova_models[0] if app.state.config.sambanova_models else None)
            providers.append(ProviderModelResponse(provider=provider, models=models, selected=selected))
        return ProviderModelsSummary(providers=providers)

    @app.post("/api/models", response_model=ProviderModelResponse)
    async def refresh_models(request: ProviderModelRequest) -> ProviderModelResponse:
        provider = request.provider.lower().strip()
        if provider not in {"cerebras", "groq", "sambanova"}:
            raise HTTPException(status_code=400, detail=f"Unsupported provider '{provider}'.")
        api_key = request.api_key.strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="API key cannot be empty.")

        try:
            models = await run_in_threadpool(fetch_provider_models, provider, api_key)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        app.state.provider_catalog[provider] = _merge_candidates(models, DEFAULT_MODEL_CANDIDATES.get(provider, []))
        if provider == "cerebras":
            app.state.config.cerebras_api_key = api_key
            app.state.config.cerebras_models = _merge_candidates(models, DEFAULT_MODEL_CANDIDATES["cerebras"])
            if app.state.config.cerebras_models:
                app.state.config.cerebras_model = app.state.config.cerebras_models[0]
        elif provider == "groq":
            app.state.config.groq_api_key = api_key
            app.state.config.groq_models = _merge_candidates(models, DEFAULT_MODEL_CANDIDATES["groq"])
        else:
            app.state.config.sambanova_api_key = api_key
            app.state.config.sambanova_models = _merge_candidates(models, DEFAULT_MODEL_CANDIDATES["sambanova"])

        if models:
            app.state.selected_models[provider] = models[0]

        return ProviderModelResponse(provider=provider, models=models, selected=models[0] if models else None)

    @app.get("/api/embeddings", response_model=EmbeddingSettings)
    def get_embeddings() -> EmbeddingSettings:
        return EmbeddingSettings(
            ollama_url=app.state.config.ollama_url,
            embedding_model=app.state.config.embedding_model,
        )

    @app.post("/api/embeddings", response_model=EmbeddingSettings)
    async def set_embeddings(request: EmbeddingSettingsRequest) -> EmbeddingSettings:
        url = (request.ollama_url or app.state.config.ollama_url or "").strip()
        model = (request.embedding_model or app.state.config.embedding_model or "").strip()
        if not url or not model:
            raise HTTPException(status_code=400, detail="Both ollama_url and embedding_model are required.")

        url = _normalize_ollama_url(url) or url
        app.state.config.ollama_url = url
        app.state.config.embedding_model = model

        if request.rebuild:
            prev_rebuild = app.state.config.rebuild
            try:
                app.state.config.rebuild = True
                (
                    vector_store,
                    default_chain,
                    key_ring,
                    initial_cache,
                    provider_catalog,
                    selected_map,
                ) = await run_in_threadpool(build_chain, app.state.config)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=f"Failed to rebuild knowledge base: {exc}") from exc
            finally:
                app.state.config.rebuild = prev_rebuild

            app.state.vector_store = vector_store
            app.state.default_chain = default_chain
            app.state.key_ring = key_ring
            app.state.chain_cache = dict(initial_cache)
            app.state.provider_catalog = {k: v[:] for k, v in provider_catalog.items()}
            app.state.selected_models = dict(selected_map)
            app.state.key_index = 0

        return EmbeddingSettings(ollama_url=url, embedding_model=model)

    @app.get("/api/mcp/orbit", response_model=OrbitMcpSettings)
    def get_orbit_mcp() -> OrbitMcpSettings:
        return OrbitMcpSettings(
            python_executable=app.state.config.orbit_mcp_python,
            server_script=_format_repo_path(app.state.config.orbit_mcp_server),
            protocol_version=app.state.config.orbit_mcp_protocol_version,
            init_timeout_s=app.state.config.orbit_mcp_init_timeout_s,
            framing=app.state.config.orbit_mcp_framing,
        )

    @app.post("/api/mcp/orbit", response_model=OrbitMcpSettings)
    def set_orbit_mcp(request: OrbitMcpSettings) -> OrbitMcpSettings:
        python_executable = request.python_executable.strip()
        server_script = request.server_script.strip()
        protocol_version = request.protocol_version.strip()
        init_timeout_s = float(request.init_timeout_s)
        framing = (request.framing or "").strip().lower()

        if not python_executable:
            raise HTTPException(status_code=400, detail="python_executable cannot be empty.")
        if not server_script:
            raise HTTPException(status_code=400, detail="server_script cannot be empty.")
        if not protocol_version:
            raise HTTPException(status_code=400, detail="protocol_version cannot be empty.")
        if framing not in {"ndjson", "lsp"}:
            raise HTTPException(status_code=400, detail="framing must be either 'ndjson' or 'lsp'.")

        server_path = _resolve_repo_path(server_script)

        app.state.config.orbit_mcp_python = python_executable
        app.state.config.orbit_mcp_server = server_path
        app.state.config.orbit_mcp_protocol_version = protocol_version
        app.state.config.orbit_mcp_init_timeout_s = init_timeout_s
        app.state.config.orbit_mcp_framing = framing

        return get_orbit_mcp()

    @app.post("/api/mcp/orbit/test", response_model=OrbitMcpTestResponse)
    async def test_orbit_mcp(request: OrbitMcpSettings) -> OrbitMcpTestResponse:
        python_executable = request.python_executable.strip()
        server_script = request.server_script.strip()
        protocol_version = request.protocol_version.strip()
        init_timeout_s = float(request.init_timeout_s)
        framing = (request.framing or "").strip().lower()

        if not python_executable:
            raise HTTPException(status_code=400, detail="python_executable cannot be empty.")
        if not server_script:
            raise HTTPException(status_code=400, detail="server_script cannot be empty.")
        if not protocol_version:
            raise HTTPException(status_code=400, detail="protocol_version cannot be empty.")
        if framing not in {"ndjson", "lsp"}:
            raise HTTPException(status_code=400, detail="framing must be either 'ndjson' or 'lsp'.")

        server_path = _resolve_repo_path(server_script)

        config = OrbitMCPConfig(
            python_executable=python_executable,
            server_script=server_path,
            protocol_version=protocol_version,
            init_timeout_s=init_timeout_s,
            framing=framing,
        )

        log_path = new_orbit_mcp_log_path("orbit_mcp_test", log_dir=default_orbit_mcp_log_dir())
        try:
            result = await run_in_threadpool(test_connection, config=config, log_path=log_path)
        except OrbitMCPError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        initialize_result = result.get("initialize") if isinstance(result, dict) else None
        tools = result.get("tools") if isinstance(result, dict) else None

        server_info: Dict[str, Any] = {}
        protocol: str | None = None
        if isinstance(initialize_result, dict):
            raw_server_info = initialize_result.get("serverInfo")
            if isinstance(raw_server_info, dict):
                server_info = raw_server_info
            raw_protocol = initialize_result.get("protocolVersion")
            if isinstance(raw_protocol, str) and raw_protocol.strip():
                protocol = raw_protocol.strip()

        tools_list: List[Dict[str, Any]] = []
        if isinstance(tools, list):
            tools_list = [item for item in tools if isinstance(item, dict)]

        # Persist tested config for subsequent /orbit calls.
        app.state.config.orbit_mcp_python = python_executable
        app.state.config.orbit_mcp_server = server_path
        app.state.config.orbit_mcp_protocol_version = protocol_version
        app.state.config.orbit_mcp_init_timeout_s = init_timeout_s
        app.state.config.orbit_mcp_framing = framing

        return OrbitMcpTestResponse(
            server_info=server_info,
            protocol_version=protocol,
            tools=tools_list,
            log_path=_format_repo_path(log_path),
        )

    @app.get("/api/mcp/orbit/logs", response_model=OrbitMcpLogsResponse)
    def list_orbit_mcp_logs(limit: int = 50) -> OrbitMcpLogsResponse:
        if limit < 1:
            raise HTTPException(status_code=400, detail="limit must be >= 1.")
        if limit > 200:
            raise HTTPException(status_code=400, detail="limit must be <= 200.")

        log_dir = default_orbit_mcp_log_dir()
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Failed to create log directory: {exc}") from exc

        entries: List[OrbitMcpLogEntry] = []
        for path in sorted(log_dir.glob("*.log")):
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append(
                OrbitMcpLogEntry(
                    name=path.name,
                    path=_format_repo_path(path),
                    size_bytes=int(stat.st_size),
                    mtime_s=float(stat.st_mtime),
                )
            )

        entries.sort(key=lambda entry: entry.mtime_s, reverse=True)
        return OrbitMcpLogsResponse(directory=_format_repo_path(log_dir), logs=entries[:limit])

    @app.get("/api/mcp/orbit/logs/{name}", response_model=OrbitMcpLogContentResponse)
    def read_orbit_mcp_log(name: str, max_bytes: int = 200_000) -> OrbitMcpLogContentResponse:
        if not name or "/" in name or "\\" in name or "\x00" in name or ".." in name:
            raise HTTPException(status_code=400, detail="Invalid log name.")
        if max_bytes < 100:
            raise HTTPException(status_code=400, detail="max_bytes must be >= 100.")
        if max_bytes > 5_000_000:
            raise HTTPException(status_code=400, detail="max_bytes must be <= 5,000,000.")

        log_dir = default_orbit_mcp_log_dir()
        path = (log_dir / name).resolve()
        try:
            # Prevent path traversal; require file to be within the log directory.
            path.relative_to(log_dir.resolve())
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid log path.")

        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="Log file not found.")

        try:
            total_bytes = path.stat().st_size
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Failed to stat log file: {exc}") from exc

        truncated = False
        data: bytes
        try:
            with open(path, "rb") as handle:
                if total_bytes > max_bytes:
                    truncated = True
                    try:
                        handle.seek(-max_bytes, os.SEEK_END)
                    except OSError:
                        handle.seek(0)
                    data = handle.read(max_bytes)
                else:
                    data = handle.read()
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read log file: {exc}") from exc

        content = data.decode("utf-8", errors="replace")
        return OrbitMcpLogContentResponse(
            name=path.name,
            path=_format_repo_path(path),
            truncated=truncated,
            total_bytes=int(total_bytes),
            content=content,
        )

    @app.get("/api/mcp/orbit/status", response_model=OrbitMcpSimStatusResponse)
    def orbit_sim_status() -> OrbitMcpSimStatusResponse:
        status = get_active_orbit_simulation()
        raw_log_path = status.get("log_path")
        log_path: str | None = None
        if isinstance(raw_log_path, str) and raw_log_path.strip():
            try:
                log_path = _format_repo_path(Path(raw_log_path))
            except Exception:
                log_path = raw_log_path

        raw_run_dir = status.get("run_dir")
        run_dir: str | None = None
        if isinstance(raw_run_dir, str) and raw_run_dir.strip():
            try:
                run_dir = _format_repo_path(Path(raw_run_dir))
            except Exception:
                run_dir = raw_run_dir

        pid = status.get("pid") if isinstance(status.get("pid"), int) else None
        started_s = status.get("started_s")
        started_s_value: float | None = None
        if isinstance(started_s, (int, float)):
            started_s_value = float(started_s)

        return OrbitMcpSimStatusResponse(
            running=bool(status.get("running")),
            pid=pid,
            log_path=log_path,
            run_dir=run_dir,
            started_s=started_s_value,
        )

    @app.post("/api/mcp/orbit/stop", response_model=OrbitMcpStopResponse)
    def orbit_sim_stop() -> OrbitMcpStopResponse:
        requested = stop_active_orbit_simulation()
        return OrbitMcpStopResponse(stop_requested=requested, status=orbit_sim_status())

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest) -> ChatResponse:
        message = payload.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        history = list(payload.history)
        chat_history = [(turn.question, turn.answer) for turn in history]

        if is_orbit_command(message):
            orbit_args = parse_orbit_command(message)
            if orbit_args is None:
                answer = "Usage: /orbit <latitude> <longitude> (example: /orbit 40.7128 -74.0060)"
                updated_history = history + [ChatTurn(question=message, answer=answer)]
                return ChatResponse(
                    answer=answer,
                    sources=[],
                    history=updated_history,
                    provider="MCP",
                    model="visualize_orbit",
                    provider_id="satellite-sim",
                )

            lat, lon = orbit_args
            try:
                answer, _run_dir_text, _log_path_text = _start_orbit_from_chat(app.state.config, lat, lon)
            except OrbitMCPError as exc:
                answer = f"Orbit visualization failed to start: {exc}"
            updated_history = history + [ChatTurn(question=message, answer=answer)]
            return ChatResponse(
                answer=answer,
                sources=[],
                history=updated_history,
                provider="MCP",
                model="visualize_orbit",
                provider_id="satellite-sim",
            )

        if _should_attempt_orbit_nl_routing(message, history):
            coords = _extract_lat_lon_from_text(message)
            request_model_overrides = {
                (name or "").strip().lower(): value.strip()
                for name, value in (payload.provider_models or {}).items()
                if name and isinstance(name, str) and isinstance(value, str) and value.strip()
            }

            key_map = {
                "cerebras": (payload.cerebras_api_key or "").strip() or (app.state.config.cerebras_api_key or "").strip(),
                "groq": (payload.groq_api_key or "").strip() or (app.state.config.groq_api_key or "").strip(),
                "sambanova": (payload.sambanova_api_key or "").strip() or (app.state.config.sambanova_api_key or "").strip(),
            }

            router_provider: str | None = None
            router_model: str | None = None
            decision: dict[str, Any] | None = None

            for provider_name in ("cerebras", "groq", "sambanova"):
                key_value = key_map.get(provider_name) or ""
                if not key_value:
                    continue
                model_value = request_model_overrides.get(provider_name) or _select_router_model(
                    app.state.config, app.state.selected_models, provider_name
                )
                if not model_value:
                    continue
                try:
                    llm = _create_router_llm(provider_name, model_value, key_value)
                except Exception:
                    continue
                decision = await run_in_threadpool(_route_orbit_request_with_llm, llm, message)
                router_provider = provider_name
                router_model = model_value
                if decision is not None:
                    break

            action = (decision.get("action") if isinstance(decision, dict) else None) or ""
            action = str(action).strip().lower()

            if action == "orbit":
                lat = decision.get("latitude") if isinstance(decision, dict) else None
                lon = decision.get("longitude") if isinstance(decision, dict) else None
                try:
                    lat_f = float(lat)
                    lon_f = float(lon)
                except Exception:
                    lat_f = lon_f = None  # type: ignore[assignment]

                if isinstance(lat_f, float) and isinstance(lon_f, float) and -90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0:
                    try:
                        answer, _run_dir_text, _log_path_text = _start_orbit_from_chat(app.state.config, lat_f, lon_f)
                    except OrbitMCPError as exc:
                        answer = f"Orbit visualization failed to start: {exc}"
                    updated_history = history + [ChatTurn(question=message, answer=answer)]
                    return ChatResponse(
                        answer=answer,
                        sources=[],
                        history=updated_history,
                        provider="MCP",
                        model="visualize_orbit",
                        provider_id="satellite-sim",
                    )

                if coords is not None:
                    lat_f, lon_f = coords
                    try:
                        answer, _run_dir_text, _log_path_text = _start_orbit_from_chat(app.state.config, lat_f, lon_f)
                    except OrbitMCPError as exc:
                        answer = f"Orbit visualization failed to start: {exc}"
                    updated_history = history + [ChatTurn(question=message, answer=answer)]
                    return ChatResponse(
                        answer=answer,
                        sources=[],
                        history=updated_history,
                        provider="MCP",
                        model="visualize_orbit",
                        provider_id="satellite-sim",
                    )

                hint = "To run the orbit simulation, include a target latitude and longitude (degrees), e.g. `/orbit 20 -110`."
                question = (decision.get("question") if isinstance(decision, dict) else None) or hint
                answer = f"{str(question).strip()}\n{hint}"
                updated_history = history + [ChatTurn(question=message, answer=answer)]
                provider_label = PROVIDER_LABELS.get(router_provider or "", router_provider) if router_provider else None
                return ChatResponse(
                    answer=answer,
                    sources=[],
                    history=updated_history,
                    provider=provider_label,
                    model=router_model,
                    provider_id="orbit-router",
                )

            if action == "clarify":
                hint = "To run the orbit simulation, include a target latitude and longitude (degrees), e.g. `/orbit 20 -110`."
                question = (decision.get("question") if isinstance(decision, dict) else None) or hint
                answer = f"{str(question).strip()}\n{hint}"
                updated_history = history + [ChatTurn(question=message, answer=answer)]
                provider_label = PROVIDER_LABELS.get(router_provider or "", router_provider) if router_provider else None
                return ChatResponse(
                    answer=answer,
                    sources=[],
                    history=updated_history,
                    provider=provider_label,
                    model=router_model,
                    provider_id="orbit-router",
                )

            if coords is not None and any(keyword in message.lower() for keyword in _ORBIT_CONTEXT_KEYWORDS):
                lat_f, lon_f = coords
                try:
                    answer, _run_dir_text, _log_path_text = _start_orbit_from_chat(app.state.config, lat_f, lon_f)
                except OrbitMCPError as exc:
                    answer = f"Orbit visualization failed to start: {exc}"
                updated_history = history + [ChatTurn(question=message, answer=answer)]
                return ChatResponse(
                    answer=answer,
                    sources=[],
                    history=updated_history,
                    provider="MCP",
                    model="visualize_orbit",
                    provider_id="satellite-sim",
                )

            if coords is None:
                hint = "To run the orbit simulation, include a target latitude and longitude (degrees), e.g. `/orbit 20 -110`."
                answer = f"{hint}"
                updated_history = history + [ChatTurn(question=message, answer=answer)]
                return ChatResponse(
                    answer=answer,
                    sources=[],
                    history=updated_history,
                    provider="MCP",
                    model="visualize_orbit",
                    provider_id="satellite-sim",
                )

        if app.state.startup_error:
            raise HTTPException(status_code=500, detail=app.state.startup_error)
        if app.state.vector_store is None:
            raise HTTPException(
                status_code=503,
                detail="Knowledge base is still initializing. Try again shortly.",
            )

        def add_key(provider: str, key: str | None) -> None:
            if not key:
                key = getattr(app.state.config, f"{provider}_api_key", None)
            if not key:
                return
            entry = (provider, key.strip())
            if not entry[1]:
                return
            if entry not in app.state.key_ring:
                app.state.key_ring.append(entry)
            if provider == "cerebras":
                app.state.config.cerebras_api_key = entry[1]
            elif provider == "groq":
                app.state.config.groq_api_key = entry[1]
            elif provider == "sambanova":
                app.state.config.sambanova_api_key = entry[1]

        add_key("cerebras", (payload.cerebras_api_key or "").strip())
        add_key("groq", (payload.groq_api_key or "").strip())
        add_key("sambanova", (payload.sambanova_api_key or "").strip())

        if not app.state.key_ring:
            raise HTTPException(
                status_code=400,
                detail="Provide at least one API key (Cerebras, Groq, or SambaNova) to chat.",
            )

        request_model_overrides = {
            (name or "").strip().lower(): value.strip()
            for name, value in (payload.provider_models or {}).items()
            if name and isinstance(name, str) and isinstance(value, str) and value.strip()
        }

        provider_candidates: Dict[str, List[str]] = {}
        for provider in ("cerebras", "groq", "sambanova"):
            selected = request_model_overrides.get(provider)
            previous_choice = app.state.selected_models.get(provider)
            if provider == "cerebras":
                configured = app.state.config.cerebras_models or []
            elif provider == "groq":
                configured = app.state.config.groq_models or []
            else:
                configured = app.state.config.sambanova_models or []
            catalog = app.state.provider_catalog.get(provider) or []
            defaults = DEFAULT_MODEL_CANDIDATES.get(provider, [])
            selected_list = [selected] if selected else None
            previous_list = [previous_choice] if previous_choice else None
            provider_candidates[provider] = _merge_candidates(selected_list, previous_list, configured, catalog, defaults)

        for provider, candidates in provider_candidates.items():
            if not candidates:
                continue
            current_catalog = app.state.provider_catalog.get(provider, [])
            app.state.provider_catalog[provider] = _merge_candidates(current_catalog, candidates)

        def get_chain(provider: str, key: str) -> tuple[ConversationalRetrievalChain, str]:
            candidates = provider_candidates.get(provider) or DEFAULT_MODEL_CANDIDATES.get(provider, [])
            if not candidates:
                raise ModelSelectionError(provider, [])
            for model_name in candidates:
                cached_chain = app.state.chain_cache.get((provider, key, model_name))
                if cached_chain is not None:
                    return cached_chain, model_name
            chain, model_used = build_provider_chain(app.state.vector_store, provider, candidates, key)
            app.state.chain_cache[(provider, key, model_used)] = chain
            catalog = app.state.provider_catalog.get(provider, [])
            app.state.provider_catalog[provider] = _merge_candidates([model_used], catalog)
            if app.state.default_chain is None:
                app.state.default_chain = chain
            return chain, model_used

        ring_snapshot = list(app.state.key_ring)
        start_index = app.state.key_index % len(ring_snapshot) if ring_snapshot else 0
        last_rate_error: Exception | None = None
        last_auth_error: Exception | None = None
        last_model_errors: List[Tuple[str, str | None, Exception]] = []
        last_chain: ConversationalRetrievalChain | None = None
        keys_to_remove: List[Tuple[str, str]] = []
        selected_entry: Tuple[str, str] | None = None
        selected_provider: str | None = None
        selected_model: str | None = None
        result: dict | None = None

        for offset in range(len(ring_snapshot)):
            idx = (start_index + offset) % len(ring_snapshot)
            provider, key = ring_snapshot[idx]
            model_used: str | None = None

            try:
                chain, model_used = get_chain(provider, key)
                result = await run_in_threadpool(
                    chain,
                    {"question": message, "chat_history": chat_history},
                )
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, ModelSelectionError):
                    if exc.attempts:
                        for model_name, model_exc in exc.attempts:
                            last_model_errors.append((provider, model_name, model_exc))
                    else:
                        last_model_errors.append((provider, None, exc))
                    keys_to_remove.append((provider, key))
                    continue
                if is_rate_limit_error(exc):
                    last_rate_error = exc
                    continue
                if is_auth_error(exc):
                    last_auth_error = exc
                    keys_to_remove.append((provider, key))
                    continue
                if is_model_error(exc):
                    last_model_errors.append((provider, model_used, exc))
                    keys_to_remove.append((provider, key))
                    continue
                raise HTTPException(status_code=500, detail=f"Chat request failed: {exc}") from exc

            selected_entry = (provider, key)
            selected_provider = provider
            selected_model = model_used
            last_chain = chain
            break

        if keys_to_remove:
            for provider, key in keys_to_remove:
                to_delete = [cache_key for cache_key in app.state.chain_cache if cache_key[0] == provider and cache_key[1] == key]
                for cache_key in to_delete:
                    app.state.chain_cache.pop(cache_key, None)
            app.state.key_ring = [entry for entry in app.state.key_ring if entry not in keys_to_remove]

        if selected_entry is None:
            if app.state.key_ring:
                app.state.key_index = app.state.key_index % len(app.state.key_ring)
            else:
                app.state.key_index = 0
            if last_auth_error and not last_rate_error:
                raise HTTPException(
                    status_code=401,
                    detail="All provided API keys were rejected. Please verify the credentials.",
                ) from last_auth_error
            if last_model_errors and not last_rate_error:
                provider_models_map: Dict[str, set[str]] = {}
                for provider_key, model_name, _err in last_model_errors:
                    label = PROVIDER_LABELS.get(provider_key, provider_key)
                    bucket = provider_models_map.setdefault(label, set())
                    if model_name:
                        bucket.add(model_name)
                parts: List[str] = []
                for label, models in provider_models_map.items():
                    if models:
                        parts.append(f"{label} ({', '.join(sorted(models))})")
                    else:
                        parts.append(label)
                providers = ", ".join(parts)
                raise HTTPException(
                    status_code=404,
                    detail=f"Requested model is unavailable for providers: {providers}. Adjust the model configuration or provide keys that support it.",
                ) from last_model_errors[0][2]
            detail = "All available API keys returned rate-limit responses. Please try again shortly."
            if last_rate_error:
                detail += f" ({last_rate_error})"
            raise HTTPException(status_code=429, detail=detail)

        provider, key = selected_entry
        try:
            current_index = app.state.key_ring.index(selected_entry)
        except ValueError:
            current_index = 0
        if app.state.key_ring:
            app.state.key_index = (current_index + 1) % len(app.state.key_ring)
        else:
            app.state.key_index = 0
        if provider == "cerebras":
            app.state.config.cerebras_api_key = key
        elif provider == "groq":
            app.state.config.groq_api_key = key
        elif provider == "sambanova":
            app.state.config.sambanova_api_key = key
        if selected_provider and selected_model:
            app.state.selected_models[selected_provider] = selected_model
            if selected_provider == "cerebras":
                app.state.config.cerebras_model = selected_model
                app.state.config.cerebras_models = _merge_candidates([selected_model], app.state.config.cerebras_models)
            elif selected_provider == "groq":
                app.state.config.groq_models = _merge_candidates([selected_model], app.state.config.groq_models)
            elif selected_provider == "sambanova":
                app.state.config.sambanova_models = _merge_candidates([selected_model], app.state.config.sambanova_models)
        if last_chain is not None:
            app.state.default_chain = last_chain
        if result is None:
            raise HTTPException(status_code=500, detail="No response generated from the language model.")

        answer = (result.get("answer") or "").strip() or "I could not generate an answer. Please try again."

        sources: List[str] = []
        seen = set()
        for doc in result.get("source_documents") or []:
            source = doc.metadata.get("source") if getattr(doc, "metadata", None) else None
            source = source or "unknown source"
            if source not in seen:
                seen.add(source)
                sources.append(source)

        updated_history = history + [ChatTurn(question=message, answer=answer)]
        provider_label = PROVIDER_LABELS.get(selected_provider or "", selected_provider)
        return ChatResponse(
            answer=answer,
            sources=sources,
            history=updated_history,
            provider=provider_label,
            model=selected_model,
        )

    return app


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if not raw:
        return default
    return Path(raw)


def _env_str(name: str, default: str | None = None) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


def _parse_models(value: str | None, default: List[str]) -> List[str]:
    if value is None:
        return default[:]
    models = [item.strip() for item in value.split(",") if item.strip()]
    return models or default[:]


def _normalize_ollama_url(url: str | None) -> str | None:
    if not url:
        return url
    cleaned = url.strip()
    if not cleaned:
        return url
    if not cleaned.startswith(("http://", "https://")):
        cleaned = f"http://{cleaned}"
    parsed = urlparse(cleaned)
    host = parsed.hostname or ""
    if host in {"0.0.0.0", ""}:
        replacement_host = "localhost"
        userinfo = ""
        if parsed.username:
            userinfo = parsed.username
            if parsed.password:
                userinfo += f":{parsed.password}"
            userinfo += "@"
        port = f":{parsed.port}" if parsed.port else ""
        netloc = f"{userinfo}{replacement_host}{port}"
        cleaned = urlunparse(parsed._replace(netloc=netloc))
    return cleaned


def load_config_from_env() -> WebConfig:
    cerebras_models_env = _parse_models(_env_str("SAT_WEB_CEREBRAS_MODELS", None), DEFAULT_MODEL_CANDIDATES["cerebras"])
    cerebras_model_env = _env_str("SAT_WEB_CEREBRAS_MODEL", None)
    cerebras_model = cerebras_model_env or (cerebras_models_env[0] if cerebras_models_env else DEFAULT_CEREBRAS_MODEL)
    cerebras_models = _merge_candidates([cerebras_model], cerebras_models_env, DEFAULT_MODEL_CANDIDATES["cerebras"])

    groq_models_env = _parse_models(_env_str("SAT_WEB_GROQ_MODELS", None), DEFAULT_MODEL_CANDIDATES["groq"])
    groq_model_env = _env_str("SAT_WEB_GROQ_MODEL", None)
    groq_models = _merge_candidates([groq_model_env] if groq_model_env else None, groq_models_env, DEFAULT_MODEL_CANDIDATES["groq"])

    sambanova_models_env = _parse_models(_env_str("SAT_WEB_SAMBANOVA_MODELS", None), DEFAULT_MODEL_CANDIDATES["sambanova"])
    sambanova_model_env = _env_str("SAT_WEB_SAMBANOVA_MODEL", None)
    sambanova_models = _merge_candidates([sambanova_model_env] if sambanova_model_env else None, sambanova_models_env, DEFAULT_MODEL_CANDIDATES["sambanova"])

    ollama_url = _normalize_ollama_url(os.getenv("SAT_WEB_OLLAMA_URL", DEFAULT_OLLAMA_URL))

    orbit_server_raw = os.getenv("SAT_ORBIT_MCP_SERVER")
    orbit_server_default = _REPO_ROOT / "SAT_Orbit_Sim_MCP" / "satellite_server.py"
    orbit_server = _resolve_repo_path(orbit_server_raw) if orbit_server_raw else orbit_server_default
    orbit_python = os.getenv("SAT_ORBIT_MCP_PYTHON", sys.executable)
    orbit_protocol = os.getenv("SAT_ORBIT_MCP_PROTOCOL_VERSION", "2024-11-05")
    orbit_timeout = _env_float("SAT_ORBIT_MCP_INIT_TIMEOUT_S", 30.0)
    orbit_framing = (os.getenv("SAT_ORBIT_MCP_FRAMING", "ndjson") or "ndjson").strip().lower()
    if orbit_framing not in {"ndjson", "lsp"}:
        orbit_framing = "ndjson"

    return WebConfig(
        host=os.getenv("SAT_WEB_HOST", "0.0.0.0"),
        port=_env_int("SAT_WEB_PORT", 8000),
        reload=_env_bool("SAT_WEB_RELOAD", False),
        persist_dir=_env_path("SAT_WEB_PERSIST_DIR", DEFAULT_PERSIST_DIR),
        source_url=os.getenv("SAT_WEB_SOURCE_URL", DEFAULT_SOURCE_URL),
        max_depth=_env_int("SAT_WEB_MAX_DEPTH", 2),
        chunk_size=_env_int("SAT_WEB_CHUNK_SIZE", 1000),
        chunk_overlap=_env_int("SAT_WEB_CHUNK_OVERLAP", 150),
        rebuild=_env_bool("SAT_WEB_REBUILD", False),
        ollama_url=ollama_url,
        embedding_model=os.getenv("SAT_WEB_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        cerebras_model=cerebras_model,
        cerebras_api_key=_env_str("SAT_WEB_CEREBRAS_KEY", _env_str("CEREBRAS_API_KEY")),
        groq_api_key=_env_str("SAT_WEB_GROQ_KEY", _env_str("GROQ_API_KEY")),
        sambanova_api_key=_env_str("SAT_WEB_SAMBANOVA_KEY", _env_str("SAMBANOVA_API_KEY")),
        cerebras_models=cerebras_models,
        groq_models=groq_models,
        sambanova_models=sambanova_models,
        orbit_mcp_python=orbit_python,
        orbit_mcp_server=orbit_server,
        orbit_mcp_protocol_version=orbit_protocol,
        orbit_mcp_init_timeout_s=orbit_timeout,
        orbit_mcp_framing=orbit_framing,
    )


def parse_args(defaults: WebConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Sat.AI web interface.")
    parser.add_argument("--host", default=defaults.host, help="Host interface for the web server.")
    parser.add_argument("--port", type=int, default=defaults.port, help="Port for the web server.")
    parser.add_argument("--reload", action="store_true", default=defaults.reload, help="Enable autoreload (development only).")
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=defaults.persist_dir,
        help="Directory for the Chroma persistence store.",
    )
    parser.add_argument(
        "--source-url",
        default=defaults.source_url,
        help="Root URL to crawl for documentation content.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=defaults.max_depth,
        help="Maximum crawl depth when fetching documentation pages.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=defaults.chunk_size,
        help="Character chunk size for splitting documents.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=defaults.chunk_overlap,
        help="Character overlap between adjacent text chunks.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        default=defaults.rebuild,
        help="Force re-fetching and rebuilding the vector store.",
    )
    parser.add_argument(
        "--ollama-url",
        default=defaults.ollama_url,
        help="Base URL for the Ollama service hosting the embedding model.",
    )
    parser.add_argument(
        "--embedding-model",
        default=defaults.embedding_model,
        help="Embedding model name served by Ollama.",
    )
    parser.add_argument(
        "--cerebras-model",
        default=defaults.cerebras_model,
        help="Cerebras model to use for answering questions.",
    )
    parser.add_argument(
        "--cerebras-api-key",
        default=defaults.cerebras_api_key,
        help="Cerebras API key to initialize the chat pipeline.",
    )
    parser.add_argument(
        "--cerebras-models",
        default=",".join(defaults.cerebras_models),
        help="Comma-separated list of Cerebras model names to try (in priority order).",
    )
    parser.add_argument(
        "--groq-api-key",
        default=defaults.groq_api_key,
        help="Groq API key to participate in round-robin load balancing.",
    )
    parser.add_argument(
        "--groq-models",
        default=",".join(defaults.groq_models),
        help="Comma-separated list of Groq model names to try (in priority order).",
    )
    parser.add_argument(
        "--sambanova-api-key",
        default=defaults.sambanova_api_key,
        help="SambaNova API key to participate in round-robin load balancing.",
    )
    parser.add_argument(
        "--sambanova-models",
        default=",".join(defaults.sambanova_models),
        help="Comma-separated list of SambaNova model names to try (in priority order).",
    )
    return parser.parse_args()


BASE_CONFIG = load_config_from_env()
CLI_CONFIG: WebConfig | None = None


def cli_app_factory() -> FastAPI:
    config = CLI_CONFIG or BASE_CONFIG
    return create_app(config)


app = create_app(BASE_CONFIG)


if __name__ == "__main__":
    args = parse_args(BASE_CONFIG)

    cerebras_model_cli = args.cerebras_model.strip() if args.cerebras_model else BASE_CONFIG.cerebras_model
    cerebras_models_cli = _merge_candidates(
        [cerebras_model_cli],
        _parse_models(args.cerebras_models, BASE_CONFIG.cerebras_models),
        DEFAULT_MODEL_CANDIDATES["cerebras"],
    )

    groq_models_cli = _merge_candidates(
        _parse_models(args.groq_models, BASE_CONFIG.groq_models),
        DEFAULT_MODEL_CANDIDATES["groq"],
    )

    sambanova_models_cli = _merge_candidates(
        _parse_models(args.sambanova_models, BASE_CONFIG.sambanova_models),
        DEFAULT_MODEL_CANDIDATES["sambanova"],
    )

    normalized_ollama = _normalize_ollama_url(args.ollama_url)

    config = WebConfig(
        host=args.host,
        port=args.port,
        reload=args.reload,
        persist_dir=args.persist_dir,
        source_url=args.source_url,
        max_depth=args.max_depth,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        rebuild=args.rebuild,
        ollama_url=normalized_ollama,
        embedding_model=args.embedding_model,
        cerebras_model=cerebras_model_cli,
        cerebras_api_key=(args.cerebras_api_key.strip() if args.cerebras_api_key else None),
        groq_api_key=(args.groq_api_key.strip() if args.groq_api_key else None),
        sambanova_api_key=(args.sambanova_api_key.strip() if args.sambanova_api_key else None),
        cerebras_models=cerebras_models_cli,
        groq_models=groq_models_cli,
        sambanova_models=sambanova_models_cli,
    )
    CLI_CONFIG = config
    if config.reload:
        uvicorn.run(
            "web_chat:cli_app_factory",
            host=config.host,
            port=config.port,
            reload=True,
            factory=True,
        )
    else:
        app = create_app(config)
        uvicorn.run(app, host=config.host, port=config.port, reload=False)
