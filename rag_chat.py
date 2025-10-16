"""
LangChain-powered RAG chat interface for the NOS3 documentation.

Features:
  * Crawls https://nos3.readthedocs.io/en/latest/ (configurable) and stores the
    processed documents in a persistent Chroma vector store.
  * Uses Ollama (embeddinggemma:latest) for embedding generation.
  * Generates answers with Cerebras GPT-OSS-120B via the LangChain ChatCerebras wrapper.
  * Provides a simple terminal chat loop with source attribution.

Usage:
  python rag_chat.py
  python rag_chat.py --help  # for customization options
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

from bs4 import BeautifulSoup
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cerebras import ChatCerebras
from langchain_ollama import OllamaEmbeddings
from pydantic import ValidationError

DEFAULT_SOURCE_URL = "https://nos3.readthedocs.io/en/latest/"
DEFAULT_PERSIST_DIR = Path("storage") / "nos3_docs_chroma"
DEFAULT_COLLECTION_NAME = "nos3_docs"
DEFAULT_EMBED_MODEL = "embeddinggemma:latest"
DEFAULT_CEREBRAS_MODEL = "gpt-oss-120b"
DEFAULT_OLLAMA_URL = "http://c240010:11434"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with the NOS3 documentation using LangChain RAG."
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=DEFAULT_PERSIST_DIR,
        help="Directory for the Chroma persistence store.",
    )
    parser.add_argument(
        "--source-url",
        default=DEFAULT_SOURCE_URL,
        help="Root URL to crawl for documentation content.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum crawl depth when fetching documentation pages.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Character chunk size for splitting documents.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Character overlap between text chunks.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force re-fetching and rebuilding the vector store.",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="Base URL for the Ollama service hosting embeddinggemma:latest.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBED_MODEL,
        help="Embedding model name served by Ollama.",
    )
    parser.add_argument(
        "--cerebras-model",
        default=DEFAULT_CEREBRAS_MODEL,
        help="Cerebras model to use for answering questions.",
    )
    return parser.parse_args()


def extract_text_from_soup(html: str) -> str:
    """Extract a focused text representation from raw HTML."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    target_tags = ["h1", "h2", "h3", "p", "li", "pre", "code"]
    parts: List[str] = []
    for tag in soup.find_all(target_tags):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)
    return "\n".join(parts)


def load_nos3_documents(url: str, max_depth: int) -> List:
    """Fetch NOS3 documentation pages and return LangChain Document objects."""
    loader = RecursiveUrlLoader(
        url=url,
        max_depth=max_depth,
        extractor=extract_text_from_soup,
        base_url="https://nos3.readthedocs.io",
        prevent_outside=True,
    )
    return loader.load()


def get_vector_store(
    *,
    persist_dir: Path,
    embeddings: OllamaEmbeddings,
    source_url: str,
    max_depth: int,
    chunk_size: int,
    chunk_overlap: int,
    rebuild: bool,
) -> Chroma:
    """Load an existing vector store or build a new one from the documentation."""
    persist_dir.mkdir(parents=True, exist_ok=True)

    store_exists = any(persist_dir.iterdir())
    if store_exists and not rebuild:
        return Chroma(
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
            collection_name=DEFAULT_COLLECTION_NAME,
        )

    print("Fetching NOS3 documentation...")
    documents = load_nos3_documents(source_url, max_depth)
    if not documents:
        raise RuntimeError(
            f"No documents were loaded from {source_url}. Check connectivity and URL."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise RuntimeError("Document splitting produced no chunks; adjust chunk settings.")

    print(f"Creating vector store with {len(chunks)} chunks...")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name=DEFAULT_COLLECTION_NAME,
    )


def build_chat_chain(
    vector_store: Chroma,
    model_name: str,
    *,
    api_key: str | None = None,
) -> ConversationalRetrievalChain:
    """Create a conversational retrieval chain using Cerebras and the provided retriever."""
    resolved_key = api_key or os.getenv("CEREBRAS_API_KEY")
    if not resolved_key:
        raise EnvironmentError("Set the CEREBRAS_API_KEY environment variable before running.")

    try:
        llm = ChatCerebras(model=model_name, cerebras_api_key=resolved_key)
    except ValidationError as exc:
        raise RuntimeError(
            "Failed to initialize ChatCerebras. Verify your API key and model name."
        ) from exc

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        max_tokens_limit=4096,
    )


def interactive_chat(chain: ConversationalRetrievalChain) -> None:
    """Run a terminal-based interactive chat loop."""
    print("NOS3 RAG assistant ready. Type 'exit' or 'quit' to finish.")
    chat_history: List[Tuple[str, str]] = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        response = chain({"question": user_input, "chat_history": chat_history})
        answer = response.get("answer", "").strip()
        if not answer:
            answer = "I could not generate an answer. Please try rephrasing your question."

        print("\nAssistant:", answer)

        sources = response.get("source_documents") or []
        if sources:
            print("Sources:")
            seen_sources = set()
            for doc in sources:
                source = doc.metadata.get("source") or "unknown source"
                if source not in seen_sources:
                    seen_sources.add(source)
                    print(f"- {source}")

        chat_history.append((user_input, answer))


def main() -> None:
    args = parse_args()

    embeddings = OllamaEmbeddings(
        model=args.embedding_model,
        base_url=args.ollama_url,
    )

    vector_store = get_vector_store(
        persist_dir=args.persist_dir,
        embeddings=embeddings,
        source_url=args.source_url,
        max_depth=args.max_depth,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        rebuild=args.rebuild,
    )

    chain = build_chat_chain(vector_store, args.cerebras_model)
    interactive_chat(chain)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - top-level guard for CLI UX
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
