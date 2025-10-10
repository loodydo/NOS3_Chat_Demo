# Session Handoff

## Current State
- Virtual environment lives at `venv/`. Dependencies installed via `pip install -r requirements.txt`.
- RAG app implemented in `rag_chat.py` using:
  - `langchain_cerebras.ChatCerebras` (requires `CEREBRAS_API_KEY` env var).
  - `langchain_ollama.OllamaEmbeddings` pointed to `http://c240010:11434` with `embeddinggemma:latest`.
  - Chroma persistence stored under `storage/nos3_docs_chroma`.
- Successful rebuild tested via `python rag_chat.py --rebuild`; chat loop runs as expected.

## How to Resume
1. Activate the venv (`source venv/bin/activate` on Unix or `.\venv\Scripts\activate` on Windows).
2. Ensure `CEREBRAS_API_KEY` is exported in the shell.
3. Confirm Ollama is running with the embedding model available.
4. Run `python rag_chat.py` to continue chatting; add `--rebuild` to refresh docs.

## Ideas / Next Steps
- Add caching for downloaded HTML to avoid repeated network calls when rebuilding.
- Expose the chat via a simple web UI (e.g., FastAPI + frontend) for non-terminal users.
- Integrate prompt templates or system prompts to shape responses further.
