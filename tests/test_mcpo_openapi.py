import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from mcpo_openapi import MCPOServerConfig, call_tool, openapi_to_tools, resolve_openapi_urls, test_connection


class TestMcpoOpenApiParsing(unittest.TestCase):
    def test_openapi_to_tools_resolves_ref(self):
        spec = {
            "openapi": "3.1.0",
            "info": {"title": "MCPO Demo", "version": "0.1.0"},
            "components": {
                "schemas": {
                    "CityRequest": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "country": {"type": "string"},
                        },
                        "required": ["city", "country"],
                    }
                }
            },
            "paths": {
                "/city_lla": {
                    "post": {
                        "summary": "Lookup city coordinates.",
                        "requestBody": {
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/CityRequest"}}}
                        },
                    }
                }
            },
        }

        tools = openapi_to_tools(spec)
        self.assertEqual(len(tools), 1)
        tool = tools[0]
        self.assertEqual(tool["name"], "city_lla")
        self.assertIn("inputSchema", tool)
        schema = tool["inputSchema"]
        self.assertIsInstance(schema, dict)
        self.assertEqual(schema.get("type"), "object")
        self.assertIn("properties", schema)
        self.assertIn("city", schema["properties"])
        self.assertIn("country", schema["properties"])

    def test_resolve_openapi_urls(self):
        base_url, openapi_url = resolve_openapi_urls("http://localhost:8000", None)
        self.assertEqual(base_url, "http://localhost:8000")
        self.assertEqual(openapi_url, "http://localhost:8000/openapi.json")


class _McpoTestHandler(BaseHTTPRequestHandler):
    openapi_spec = {
        "openapi": "3.1.0",
        "info": {"title": "MCPO Demo", "version": "0.1.0"},
        "paths": {
            "/city_lla": {
                "post": {
                    "summary": "Lookup city coordinates.",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "city": {"type": "string"},
                                        "country": {"type": "string"},
                                    },
                                    "required": ["city", "country"],
                                }
                            }
                        }
                    },
                }
            }
        },
    }

    def log_message(self, fmt, *args):
        return

    def _send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/openapi.json":
            self._send_json(self.openapi_spec)
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        if self.path != "/city_lla":
            self._send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length > 0 else b"{}"
        payload = json.loads(body.decode("utf-8"))
        city = payload.get("city")
        country = payload.get("country")
        if not city or not country:
            self._send_json({"error": "missing fields"}, status=400)
            return
        self._send_json({"latitude": 48.85, "longitude": 2.35, "altitude_m": 35})


class TestMcpoOpenApiIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _McpoTestHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.server.server_address
        cls.base_url = f"http://{host}:{port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_mcpo_roundtrip(self):
        config = MCPOServerConfig(name="local", base_url=self.base_url)
        result = test_connection(config=config, timeout_s=5.0)
        tools = result.get("tools") or []
        self.assertTrue(any(tool.get("name") == "city_lla" for tool in tools))

        tool = next(tool for tool in tools if tool.get("name") == "city_lla")
        response = call_tool(
            config=config,
            path=tool.get("path"),
            method=tool.get("method"),
            arguments={"city": "Paris", "country": "FR"},
            timeout_s=5.0,
        )
        self.assertIn("latitude", response)
        self.assertIn("longitude", response)
