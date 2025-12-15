# SAT AI – NOS3 RAG Chat

Retrieval-augmented chat assistant that answers questions about the NASA Operational Simulator for Small Satellites (NOS3). The app crawls the NOS3 documentation, builds a persistent Chroma vector store, and uses LangChain with Cerebras GPT-OSS models plus Ollama embeddings to power an interactive terminal chat.

## Features
- Crawls and chunks `https://nos3.readthedocs.io/en/latest/`, keeping content in a local vector store for fast reuse.
- Uses `embeddinggemma:latest` served via Ollama for embeddings.
- Streams answers from the Cerebras Inference API (`gpt-oss-120b` by default) with LangChain’s conversational retrieval chain.
- Provides source attribution so you can trace each response back to the underlying documentation pages.

## Requirements
- Python 3.9 or later
- Access to an Ollama runtime hosting the embedding model (default URL `http://c240010:11434`)
- Cerebras Inference API key available as the `CEREBRAS_API_KEY` environment variable
- Internet access on first run (or whenever rebuilding the document index)
- Dependencies listed in `requirements.txt`
- (Optional) A local GUI + additional deps for the orbit visualization (`requirements-orbit-sim.txt`)

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate        # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
export CEREBRAS_API_KEY="your-api-key"
# ensure the embedding model is available locally
ollama pull embeddinggemma:latest
python rag_chat.py
```

The first launch crawls the NOS3 docs and writes the vector store to `storage/nos3_docs_chroma`. Subsequent runs reuse those embeddings.

## Web Interface
Start a browser-based chat by running:
```bash
python web_chat.py
```
The FastAPI server defaults to `http://0.0.0.0:8000/` and serves a lightweight UI that keeps the conversation history client-side. Open the **Settings** drawer to enter your Cerebras, Groq, and SambaNova API keys, refresh the live model list from each provider, and pick the exact alias you want to route to—each key/model choice can be remembered locally per browser. The backend automatically round-robins across every working key, always targeting the `gpt-oss-120b` family while transparently falling back to the provider-specific aliases you select, and any invalid key/model pair is dropped from the rotation. Use the **New Chat** button to clear the conversation history, and enjoy Markdown-aware rendering for tables, code blocks, and inline formatting. All vector store and model parameters mirror the CLI defaults; override them with flags (for example, `--port 8080 --reload`) or export environment variables such as `SAT_WEB_PERSIST_DIR`, `SAT_WEB_SOURCE_URL`, `SAT_WEB_REBUILD=1`, `SAT_WEB_CEREBRAS_KEY`, `SAT_WEB_GROQ_KEY`, `SAT_WEB_SAMBANOVA_KEY`, `SAT_WEB_CEREBRAS_MODELS`, `SAT_WEB_GROQ_MODELS`, or `SAT_WEB_SAMBANOVA_MODELS` before launching.

---

## Orbit Visualization (MCP)

This repo includes an MCP server under `SAT_Orbit_Sim_MCP/` that can launch a Matplotlib orbit visualization for a target latitude/longitude.

Install optional dependencies:
```bash
pip install -r requirements.txt -r requirements-orbit-sim.txt
```

Then, from either the CLI chat or the web UI, run:
```text
/orbit <latitude> <longitude>
```

You can also ask in natural language (web UI), for example:
```text
Run the orbit sim with target 20 -110
```

Notes:
- The visualization window opens on the machine running `web_chat.py` / `rag_chat.py` (it is not rendered in-browser).
- In `rag_chat.py` the `/orbit` command blocks until you close the window; in `web_chat.py` it runs in the background so the UI stays responsive.
- Override the server script path with `SAT_ORBIT_MCP_SERVER` if needed.
- Override the TLE source (path or URL) with `SAT_ORBIT_TLE_SOURCE` (defaults to `SAT_Orbit_Sim_MCP/stations.txt` when present).

## Remote MCP Servers (HTTP/SSE)

The web UI Settings panel also supports adding **remote** MCP servers over HTTP (SSE transport). After you **Test & List Tools**, the returned tools are cached and the chat router can agentically call enabled remote tools.

For local testing, you can run the included toy server:

```bash
python -m mcp_http_test_server.server --port 8001
```

Then add `http://localhost:8001` under **Settings → MCP (Remote Servers)** and click **Test & List Tools**.

## Running via Docker

You can run the entire NOS3 RAG Chat app inside a Docker container—either in **CLI mode** or **web chat mode**.

### 1. Create a `.env` file

Set your environment variables in a `.env` file at the project root:

```bash
CEREBRAS_API_KEY=csk-...
GROQ_API_KEY=gsk_...
SAMBANOVA_API_KEY=...
OLLAMA_HOST=http://c240010:11434
DEFAULT_EMBED_MODEL=embeddinggemma:latest
```

### 2. Build the Docker image

Run the following command to build the image:

```bash
docker build . -t nos3_agent
```

### 3. Run the container (Web UI)

To start the **web chat interface** from inside the container:

```bash
docker run -it -p 8000:8000 --env-file .env nos3_agent
python web_chat.py
```

The FastAPI server will be available at:

```
http://localhost:8000
```

### 4. Run the container (CLI chat)

To use the interactive terminal chat instead:

```bash
docker run -it --env-file .env nos3_agent
python rag_chat.py
```

---

## Command Reference

```bash
python rag_chat.py [options]
```

| Option                   | Default                                  | Purpose                                                |
| ------------------------ | ---------------------------------------- | ------------------------------------------------------ |
| `--persist-dir PATH`     | `storage/nos3_docs_chroma`               | Override where the Chroma index is stored.             |
| `--source-url URL`       | `https://nos3.readthedocs.io/en/latest/` | Change the documentation root to crawl.                |
| `--max-depth N`          | `2`                                      | Limit crawl depth; lower numbers fetch fewer pages.    |
| `--chunk-size N`         | `1000`                                   | Character length for each text chunk before embedding. |
| `--chunk-overlap N`      | `150`                                    | Overlap between adjacent chunks to preserve context.   |
| `--rebuild`              | `False`                                  | Force a fresh crawl and vector-store rebuild.          |
| `--ollama-url URL`       | `http://c240010:11434`                   | Point to a different Ollama endpoint.                  |
| `--embedding-model NAME` | `embeddinggemma:latest`                  | Specify an alternate Ollama-served embedding model.    |
| `--cerebras-model NAME`  | `gpt-oss-120b`                           | Pick a different Cerebras LLM for answering.           |

Exit the chat with `exit` or `quit`. When available, the assistant prints the documentation sources used to answer each question.

---

That’s the **only modified section** — everything else stays exactly as in your original README, but the Docker section now matches your current Dockerfile and workflow.
