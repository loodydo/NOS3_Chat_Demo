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

## Command Reference
```bash
python rag_chat.py [options]
```

| Option | Default | Purpose |
|--------|---------|---------|
| `--persist-dir PATH` | `storage/nos3_docs_chroma` | Override where the Chroma index is stored. |
| `--source-url URL` | `https://nos3.readthedocs.io/en/latest/` | Change the documentation root to crawl. |
| `--max-depth N` | `2` | Limit crawl depth; lower numbers fetch fewer pages. |
| `--chunk-size N` | `1000` | Character length for each text chunk before embedding. |
| `--chunk-overlap N` | `150` | Overlap between adjacent chunks to preserve context. |
| `--rebuild` | `False` | Force a fresh crawl and vector-store rebuild. |
| `--ollama-url URL` | `http://c240010:11434` | Point to a different Ollama endpoint. |
| `--embedding-model NAME` | `embeddinggemma:latest` | Specify an alternate Ollama-served embedding model. |
| `--cerebras-model NAME` | `gpt-oss-120b` | Pick a different Cerebras LLM for answering. |

Exit the chat with `exit` or `quit`. When available, the assistant prints the documentation sources used to answer each question.

## Repository Layout
- `rag_chat.py` – main entry point that orchestrates crawling, indexing, and chatting.
- `requirements.txt` – pinned dependencies for the LangChain, Cerebras, and Ollama integrations.
- `storage/` – default location where the persistent Chroma database is saved (created on first run).
- `CONTINUE.md` – session handoff notes and ideas for next steps.

## Development Tips
- Delete `storage/nos3_docs_chroma` if you want to fully rebuild the index without using `--rebuild`.
- Adjust `--max-depth` to trade off between coverage and crawl time.
- Wrap calls in `python rag_chat.py --help` to confirm available flags and defaults.
- If you encounter authentication errors, confirm `CEREBRAS_API_KEY` and connectivity to the Cerebras endpoint.
- When running in new environments, re-run `ollama pull embeddinggemma:latest` to ensure the embedding model is cached locally.

## Troubleshooting
- **Cerebras authentication**: set `CEREBRAS_API_KEY` (e.g., export in the shell or add to a `.env` file).
- **Ollama connectivity**: ensure the Ollama service is reachable from your machine and that the embedding model has been pulled.
- **Empty crawl**: verify network access and that the target URL is correct; you can test with `python rag_chat.py --source-url <url> --rebuild`.
- **Stale data**: pass `--rebuild` or delete the persistence directory to refresh the knowledge base.

Happy chatting!
