from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable
from urllib.parse import urljoin, urlparse

import requests


class MCPOError(RuntimeError):
    pass


@dataclass(frozen=True)
class MCPOServerConfig:
    name: str
    base_url: str
    openapi_url: str | None = None
    headers: Dict[str, str] | None = None


_TOOL_NAME_RE = re.compile(r"[^a-zA-Z0-9_]+")


def _validate_http_url(value: str, field_name: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        raise ValueError(f"{field_name} is required")
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"{field_name} must start with http:// or https://")
    return cleaned


def resolve_openapi_urls(base_url: str, openapi_url: str | None = None) -> tuple[str, str]:
    base = _validate_http_url(base_url, "base_url")
    override = (openapi_url or "").strip() or None

    if override:
        openapi = _validate_http_url(override, "openapi_url")
        return base.rstrip("/"), openapi

    trimmed = base.rstrip("/")
    if trimmed.endswith("/openapi.json"):
        base_only = trimmed[: -len("/openapi.json")]
        if not base_only:
            raise ValueError("base_url cannot be inferred from openapi_url")
        return base_only, trimmed

    return trimmed, f"{trimmed}/openapi.json"


def _request_headers(extra: Dict[str, str] | None) -> Dict[str, str]:
    headers: Dict[str, str] = {"Accept": "application/json"}
    if extra:
        headers.update({k: v for k, v in extra.items() if k and v})
    return headers


def _extract_json_response(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        text = ""
        try:
            text = resp.text[:2000]
        except Exception:
            pass
        raise MCPOError(f"Invalid JSON response: {exc}. Body: {text}") from exc


def fetch_openapi_spec(
    *,
    openapi_url: str,
    headers: Dict[str, str] | None = None,
    timeout_s: float = 10.0,
) -> Dict[str, Any]:
    url = _validate_http_url(openapi_url, "openapi_url")
    try:
        resp = requests.get(url, headers=_request_headers(headers), timeout=(5, timeout_s))
    except requests.RequestException as exc:
        raise MCPOError(f"Failed to fetch OpenAPI spec from {url!r}: {exc}") from exc

    if resp.status_code < 200 or resp.status_code >= 300:
        body = ""
        try:
            body = resp.text[:2000]
        except Exception:
            pass
        raise MCPOError(f"OpenAPI fetch returned HTTP {resp.status_code}: {body}")

    payload = _extract_json_response(resp)
    if not isinstance(payload, dict):
        raise MCPOError("OpenAPI spec must be a JSON object.")
    if "openapi" not in payload or "paths" not in payload:
        raise MCPOError("OpenAPI spec missing required fields (openapi, paths).")
    return payload


def _resolve_ref(ref: str, components: Dict[str, Any], seen: set[str]) -> Dict[str, Any] | None:
    if not ref.startswith("#/components/schemas/"):
        return None
    name = ref.split("/", 3)[-1]
    if not name or name in seen:
        return None
    seen.add(name)
    schemas = components.get("schemas")
    if not isinstance(schemas, dict):
        return None
    target = schemas.get(name)
    if not isinstance(target, dict):
        return None
    return target


def _resolve_schema(schema: Any, components: Dict[str, Any], seen: set[str] | None = None) -> Any:
    if not isinstance(schema, dict):
        return schema
    if seen is None:
        seen = set()

    ref = schema.get("$ref")
    if isinstance(ref, str):
        resolved = _resolve_ref(ref, components, seen)
        if resolved is None:
            return schema
        return _resolve_schema(resolved, components, seen)

    resolved_schema = dict(schema)

    props = resolved_schema.get("properties")
    if isinstance(props, dict):
        resolved_schema["properties"] = {k: _resolve_schema(v, components, seen) for k, v in props.items()}

    if "items" in resolved_schema:
        resolved_schema["items"] = _resolve_schema(resolved_schema["items"], components, seen)

    for key in ("anyOf", "oneOf", "allOf"):
        entries = resolved_schema.get(key)
        if isinstance(entries, list):
            resolved_schema[key] = [_resolve_schema(entry, components, seen) for entry in entries]

    return resolved_schema


def _select_json_content(content: Dict[str, Any]) -> Dict[str, Any] | None:
    if "application/json" in content:
        return content.get("application/json")
    for key, value in content.items():
        if isinstance(key, str) and "json" in key.lower():
            return value if isinstance(value, dict) else None
    return None


def _schema_from_request_body(request_body: Any, components: Dict[str, Any]) -> Dict[str, Any] | None:
    if not isinstance(request_body, dict):
        return None
    content = request_body.get("content")
    if not isinstance(content, dict):
        return None
    entry = _select_json_content(content)
    if not isinstance(entry, dict):
        return None
    schema = entry.get("schema")
    if schema is None:
        return None
    resolved = _resolve_schema(schema, components)
    return resolved if isinstance(resolved, dict) else None


def _schema_from_parameters(parameters: Any, components: Dict[str, Any]) -> Dict[str, Any] | None:
    if not isinstance(parameters, list):
        return None
    properties: Dict[str, Any] = {}
    required: list[str] = []
    for param in parameters:
        if not isinstance(param, dict):
            continue
        name = param.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        location = str(param.get("in") or "").strip().lower()
        if location not in {"query", "path"}:
            continue
        schema = param.get("schema")
        resolved = _resolve_schema(schema, components) if isinstance(schema, dict) else None
        properties[name] = resolved if isinstance(resolved, dict) else {"type": "string"}
        if param.get("required") is True:
            required.append(name)

    if not properties:
        return None
    payload: Dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        payload["required"] = required
    return payload


def _ensure_object_schema(schema: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(schema, dict):
        return None
    if "type" not in schema and ("properties" in schema or "required" in schema):
        schema = dict(schema)
        schema["type"] = "object"
    return schema


def _normalize_tool_name(path: str) -> str:
    cleaned = (path or "").strip()
    cleaned = cleaned.split("?", 1)[0].strip("/")
    if not cleaned:
        return "tool"
    cleaned = cleaned.replace("{", "").replace("}", "")
    cleaned = _TOOL_NAME_RE.sub("_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned or "tool"


def _unique_tool_name(name: str, existing: set[str]) -> str:
    base = name or "tool"
    if base not in existing:
        existing.add(base)
        return base
    idx = 2
    while f"{base}_{idx}" in existing:
        idx += 1
    final = f"{base}_{idx}"
    existing.add(final)
    return final


def _iter_operations(paths: Dict[str, Any]) -> Iterable[tuple[str, str, Dict[str, Any]]]:
    for path, details in paths.items():
        if not isinstance(details, dict):
            continue
        for method in ("post", "get"):
            operation = details.get(method)
            if isinstance(operation, dict):
                yield path, method, operation


def openapi_to_tools(spec: Dict[str, Any]) -> list[Dict[str, Any]]:
    paths = spec.get("paths")
    if not isinstance(paths, dict):
        return []
    components = spec.get("components")
    if not isinstance(components, dict):
        components = {}

    tools: list[Dict[str, Any]] = []
    used_names: set[str] = set()

    for path, method, operation in _iter_operations(paths):
        name = _unique_tool_name(_normalize_tool_name(str(path)), used_names)
        summary = operation.get("summary") if isinstance(operation, dict) else None
        description = operation.get("description") if isinstance(operation, dict) else None
        if not isinstance(summary, str):
            summary = None
        if not isinstance(description, str):
            description = None
        desc_text = (description or summary or "").strip()

        schema = _schema_from_request_body(operation.get("requestBody"), components)
        if schema is None:
            schema = _schema_from_parameters(operation.get("parameters"), components)
        schema = _ensure_object_schema(schema)

        tool: Dict[str, Any] = {
            "name": name,
            "description": desc_text,
            "inputSchema": schema,
            "path": path,
            "method": method.upper(),
            "operation_id": operation.get("operationId") if isinstance(operation, dict) else None,
        }
        tools.append(tool)

    return tools


def _join_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/") + "/"
    return urljoin(base, str(path or "").lstrip("/"))


def _normalize_result(payload: Any) -> Any:
    if isinstance(payload, dict) and "result" in payload:
        return payload.get("result")
    return payload


def test_connection(
    *,
    config: MCPOServerConfig,
    timeout_s: float = 10.0,
) -> Dict[str, Any]:
    try:
        base_url, openapi_url = resolve_openapi_urls(config.base_url, config.openapi_url)
    except Exception as exc:  # noqa: BLE001
        raise MCPOError(f"Invalid MCPO URL settings: {exc}") from exc
    spec = fetch_openapi_spec(openapi_url=openapi_url, headers=config.headers, timeout_s=timeout_s)
    tools = openapi_to_tools(spec)

    info = spec.get("info") if isinstance(spec.get("info"), dict) else {}
    server_info = {
        "name": str(info.get("title") or "").strip() or config.name,
        "version": str(info.get("version") or "").strip() or None,
    }
    return {
        "server_info": {k: v for k, v in server_info.items() if v},
        "openapi_url": openapi_url,
        "tools": tools,
        "base_url": base_url,
    }


def call_tool(
    *,
    config: MCPOServerConfig,
    path: str,
    method: str,
    arguments: Dict[str, Any],
    timeout_s: float = 30.0,
) -> Any:
    try:
        base_url, _openapi_url = resolve_openapi_urls(config.base_url, config.openapi_url)
    except Exception as exc:  # noqa: BLE001
        raise MCPOError(f"Invalid MCPO URL settings: {exc}") from exc
    url = _join_url(base_url, path)
    headers = _request_headers(config.headers)
    method_lc = (method or "POST").strip().lower()
    args = arguments if isinstance(arguments, dict) else {}

    try:
        if method_lc == "get":
            resp = requests.get(url, params=args, headers=headers, timeout=(5, timeout_s))
        else:
            resp = requests.post(url, json=args, headers=headers, timeout=(5, timeout_s))
    except requests.RequestException as exc:
        raise MCPOError(f"Failed to call MCPO tool {method_lc.upper()} {url!r}: {exc}") from exc

    if resp.status_code < 200 or resp.status_code >= 300:
        body = ""
        try:
            body = resp.text[:2000]
        except Exception:
            pass
        raise MCPOError(f"MCPO tool call returned HTTP {resp.status_code}: {body}")

    payload = _extract_json_response(resp)
    return _normalize_result(payload)


def call_tool_text(
    *,
    config: MCPOServerConfig,
    path: str,
    method: str,
    arguments: Dict[str, Any],
    timeout_s: float = 30.0,
) -> str:
    result = call_tool(config=config, path=path, method=method, arguments=arguments, timeout_s=timeout_s)
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    return json.dumps(result, indent=2, ensure_ascii=False).strip()
