"""
Ingest a GitHub repository into ChromaDB for code search.

Usage:
    python ingest.py --repo https://github.com/org/repo
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Generator

import chromadb
import click
import git
import voyageai
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    CHROMA_PERSIST_DIR,
    EMBED_BATCH_SIZE,
    MAX_FILE_BYTES,
    REPOS_CACHE_DIR,
    SKIP_DIRS,
    SKIP_EXTENSIONS,
    TREESITTER_LANGUAGES,
    VOYAGE_API_KEY,
    VOYAGE_MODEL,
)
from models import CodeChunk
from chunkers import TreeSitterChunker, FixedSizeChunker
from chunkers.base import BaseChunker

console = Console()


def derive_collection_name(repo_url: str) -> str:
    """Normalize GitHub URL to a valid ChromaDB collection name."""
    # https://github.com/org/repo  →  org__repo
    match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", repo_url.rstrip("/"))
    if match:
        name = match.group(1).replace("/", "__").replace("-", "_")
    else:
        name = hashlib.sha256(repo_url.encode()).hexdigest()[:16]
    # ChromaDB requires 3-63 chars, alphanumeric + underscore + hyphen
    name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)[:63]
    if len(name) < 3:
        name = name + "_qa"
    return name


def clone_or_pull(repo_url: str, cache_dir: str = REPOS_CACHE_DIR) -> tuple[git.Repo, Path]:
    collection_name = derive_collection_name(repo_url)
    local_dir = Path(cache_dir) / collection_name
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    if local_dir.exists():
        console.print(f"[dim]Pulling latest changes for {local_dir}...[/dim]")
        repo = git.Repo(local_dir)
        repo.remotes.origin.pull()
    else:
        console.print(f"[dim]Cloning {repo_url}...[/dim]")
        repo = git.Repo.clone_from(repo_url, local_dir)

    return repo, local_dir


def _is_binary(file_path: Path) -> bool:
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(8192)
        return b"\x00" in chunk
    except OSError:
        return True


def walk_repo_files(repo_dir: Path) -> Generator[Path, None, None]:
    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue
        # Skip hidden/vendor directories
        parts = set(path.relative_to(repo_dir).parts)
        if parts & SKIP_DIRS:
            continue
        # Skip by extension
        if path.suffix.lower() in SKIP_EXTENSIONS:
            continue
        # Skip large files
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        # Skip binary files
        if _is_binary(path):
            continue
        yield path


def chunk_file(file_path: Path, repo_root: Path, repo_url: str) -> list[CodeChunk]:
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    if not content.strip():
        return []

    rel_path = str(file_path.relative_to(repo_root))
    chunker_instance: BaseChunker
    language = BaseChunker.detect_language(str(file_path))

    if language in TREESITTER_LANGUAGES:
        chunker_instance = TreeSitterChunker()
    else:
        chunker_instance = FixedSizeChunker()

    chunks = chunker_instance.chunk(rel_path, content, language)
    for chunk in chunks:
        chunk["repo_url"] = repo_url
        chunk["file_path"] = rel_path
    return chunks


_FREE_TIER_BATCH_SIZE = 8    # keeps each call well under 10K TPM
_FREE_TIER_SLEEP_SEC = 21   # 3 RPM → wait 21s between calls


@retry(stop=stop_after_attempt(8), wait=wait_exponential(multiplier=2, min=30, max=120))
def _embed_batch(voyage_client: voyageai.Client, texts: list[str]) -> list[list[float]]:
    result = voyage_client.embed(texts, model=VOYAGE_MODEL, input_type="document")
    return result.embeddings


def embed_chunks_batch(chunks: list[CodeChunk], voyage_client: voyageai.Client) -> list[CodeChunk]:
    import time

    embedded = list(chunks)
    texts = [c["content"] for c in embedded]
    batch_size = _FREE_TIER_BATCH_SIZE

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        embeddings = _embed_batch(voyage_client, batch_texts)
        for j, emb in enumerate(embeddings):
            embedded[i + j]["embedding"] = emb  # type: ignore[typeddict-unknown-key]
        # Respect VoyageAI free-tier 3 RPM limit
        if i + batch_size < len(texts):
            time.sleep(_FREE_TIER_SLEEP_SEC)

    return embedded


def store_in_chroma(
    chunks: list[CodeChunk],
    chroma_client: chromadb.PersistentClient,
    collection_name: str,
) -> None:
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for chunk in chunks:
        emb = chunk.get("embedding")  # type: ignore[call-overload]
        if emb is None:
            continue
        ids.append(chunk["chunk_id"])
        documents.append(chunk["content"])
        metadatas.append(
            {
                "file_path": chunk["file_path"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "language": chunk["language"],
                "symbol_name": chunk.get("symbol_name") or "",
                "repo_url": chunk["repo_url"],
            }
        )
        embeddings.append(emb)

    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )


def ingest(repo_url: str, chroma_persist_dir: str = CHROMA_PERSIST_DIR) -> str:
    if not VOYAGE_API_KEY:
        console.print("[red]VOYAGE_API_KEY is not set.[/red]")
        sys.exit(1)

    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
    chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)
    collection_name = derive_collection_name(repo_url)

    _, repo_dir = clone_or_pull(repo_url)

    all_files = list(walk_repo_files(repo_dir))
    console.print(f"Found [bold]{len(all_files)}[/bold] files to index.")

    all_chunks: list[CodeChunk] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        chunk_task = progress.add_task("Chunking files...", total=len(all_files))
        for file_path in all_files:
            chunks = chunk_file(file_path, repo_dir, repo_url)
            all_chunks.extend(chunks)
            progress.advance(chunk_task)

    console.print(f"Created [bold]{len(all_chunks)}[/bold] chunks. Embedding...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        embed_task = progress.add_task("Embedding chunks...", total=len(all_chunks))
        embedded_chunks: list[CodeChunk] = []
        # embed_chunks_batch handles rate-limit pacing internally
        embedded_chunks = embed_chunks_batch(all_chunks, voyage_client)
        progress.advance(embed_task, len(all_chunks))

    console.print("Storing in ChromaDB...")
    store_in_chroma(embedded_chunks, chroma_client, collection_name)

    collection = chroma_client.get_collection(collection_name)
    console.print(
        f"[green]Done.[/green] Collection [bold]{collection_name}[/bold] has "
        f"[bold]{collection.count()}[/bold] documents."
    )
    return collection_name


@click.command()
@click.option("--repo", required=True, help="GitHub repository URL")
@click.option("--chroma-dir", default=CHROMA_PERSIST_DIR, help="ChromaDB persist directory")
def main(repo: str, chroma_dir: str):
    ingest(repo, chroma_dir)


if __name__ == "__main__":
    main()
