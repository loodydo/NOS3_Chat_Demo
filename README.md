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

---

## Running via Docker

You can run the entire NOS3 RAG Chat app inside a Docker container for consistent, isolated environments.

### 1. Create a `.env` file

Set your environment variables in a `.env` file at the project root:

```bash
CEREBRAS_API_KEY=asdfasdfasdf
OLLAMA_HOST=asdfasdfasdf
DEFAULT_EMBED_MODEL=embeddinggemma:latest
```

### 2. Build the Docker image

Run the following command to build the image:

```bash
docker build . -t nos3_agent
```

### 3. Run the container

Start an interactive container, automatically loading your `.env` variables:

```bash
docker run -it --env-file .env nos3_agent
```

### 4. Launch the chat

Once inside the container, run:

```bash
python rag_chat.py
```

The app will initialize the NOS3 documentation vector store (if not already present) and launch the interactive RAG chat interface.

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

Would you like me to also include a short **“Dockerfile reference”** section (with the key base image and commands) so users can modify or extend it easily?
