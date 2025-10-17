"""
FastAPI-powered web chat interface for the NOS3 documentation RAG assistant.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NOS3 RAG Chat</title>
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
        <h1 class="title">NOS3 RAG Chat</h1>
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

      const syncStoredSettings = () => {
        PROVIDERS.forEach(({ id }) => {
          persistKey(id);
          persistModel(id);
        });
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

      PROVIDERS.forEach(initProvider);
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

      const resetChat = (message = "NOS3 RAG assistant ready. Ask me about the documentation.") => {
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
    app = FastAPI(title="NOS3 RAG Chat", version="0.1.0")

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

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest) -> ChatResponse:
        if app.state.startup_error:
            raise HTTPException(status_code=500, detail=app.state.startup_error)
        if app.state.vector_store is None:
            raise HTTPException(
                status_code=503,
                detail="Knowledge base is still initializing. Try again shortly.",
            )

        message = payload.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        history = list(payload.history)
        chat_history = [(turn.question, turn.answer) for turn in history]

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
    )


def parse_args(defaults: WebConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the NOS3 RAG chat web interface.")
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
