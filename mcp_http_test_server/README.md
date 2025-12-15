## Sat.AI Remote MCP HTTP Test Server

This is a small, local HTTP MCP server (SSE transport) you can run to test the **Remote MCP Servers** feature in the `web_chat.py` settings panel.

### Run

From the repo root:

```bash
python3 -m mcp_http_test_server.server --host 0.0.0.0 --port 8001
```

On Windows (venv example):

```powershell
.\venv\Scripts\python.exe -m mcp_http_test_server.server --host 0.0.0.0 --port 8001
```

### Connect From Sat.AI

1. Start `web_chat.py`.
2. Open **Settings** → **MCP (Remote Servers)**.
3. Add a server:
   - Name: `Local Test Server`
   - URL: `http://localhost:8001`
4. Select it and click **Test & List Tools**.

After testing, the tool list is cached and the chat router can agentically call these tools based on natural language prompts.

