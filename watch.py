"""
watch.py — File watcher for proactive error detection.

Watches the cloned repo directory and, on each saved file:
  1. Instant syntax check (Python only, via ast.parse — no API call).
  2. Re-indexes only that file (incremental, using the manifest).
  3. Runs a fast LLM error check using chunks fetched directly from ChromaDB.

Usage (via main.py):
    python main.py --repo https://github.com/org/repo --watch
    python main.py --repo https://github.com/org/repo --index-first --watch
"""
from __future__ import annotations

import ast
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chromadb
import voyageai
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer
from rich.console import Console

from config import (
    CHROMA_PERSIST_DIR,
    GROQ_FAST_MODEL,
    MAX_FILE_BYTES,
    REPOS_CACHE_DIR,
    SKIP_DIRS,
    SKIP_EXTENSIONS,
    SYSTEM_PROMPT,
    WATCH_DEBOUNCE_SECONDS,
)
from ingest import (
    _is_binary,
    chunk_file,
    derive_collection_name,
    embed_chunks_batch,
    store_in_chroma,
)
from manifest import hash_file, load_manifest, save_manifest

console = Console()


def _syntax_check(abs_path: Path) -> str | None:
    """Return an error message if the file has a syntax error, else None. Python only."""
    if abs_path.suffix.lower() != ".py":
        return None
    try:
        source = abs_path.read_text(encoding="utf-8", errors="replace")
        ast.parse(source, filename=str(abs_path))
        return None
    except SyntaxError as e:
        return f"Line {e.lineno}: {e.msg} — `{(e.text or '').strip()}`"


class ChangeHandler(FileSystemEventHandler):
    def __init__(
        self,
        repo_url: str,
        repo_dir: Path,
        collection_name: str,
        voyage_client: voyageai.Client,
        chroma_client: chromadb.PersistentClient,
        graph_config: dict,
        chroma_persist_dir: str,
    ):
        self._repo_url = repo_url
        self._repo_dir = repo_dir
        self._collection_name = collection_name
        self._voyage_client = voyage_client
        self._chroma_client = chroma_client
        self._groq_client = graph_config["configurable"]["anthropic_client"]
        self._chroma_persist_dir = chroma_persist_dir
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        # max_workers=1 serializes processing so concurrent saves don't race on ChromaDB writes
        self._executor = ThreadPoolExecutor(max_workers=1)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        # Some editors (vim, emacs) do an atomic save: write temp → rename to target
        if not event.is_directory:
            self._schedule(event.dest_path)

    def _schedule(self, abs_path_str: str) -> None:
        with self._lock:
            existing = self._timers.pop(abs_path_str, None)
            if existing:
                existing.cancel()
            t = threading.Timer(WATCH_DEBOUNCE_SECONDS, self._dispatch, args=[abs_path_str])
            self._timers[abs_path_str] = t
            t.start()

    def _dispatch(self, abs_path_str: str) -> None:
        with self._lock:
            self._timers.pop(abs_path_str, None)
        self._executor.submit(self._process, abs_path_str)

    def _process(self, abs_path_str: str) -> None:
        abs_path = Path(abs_path_str)
        if not abs_path.is_file():
            return

        # Apply the same guards as walk_repo_files
        try:
            rel_path = str(abs_path.relative_to(self._repo_dir))
        except ValueError:
            return

        parts = set(Path(rel_path).parts)
        if parts & SKIP_DIRS:
            return
        if abs_path.suffix.lower() in SKIP_EXTENSIONS:
            return
        try:
            if abs_path.stat().st_size > MAX_FILE_BYTES:
                return
        except OSError:
            return
        if _is_binary(abs_path):
            return

        # Skip if content hash is unchanged (editor may have touched mtime without changing content)
        old_hashes = load_manifest(self._chroma_persist_dir, self._collection_name)
        new_hash = hash_file(abs_path)
        if old_hashes.get(rel_path) == new_hash:
            return

        console.print(f"\n[bold yellow]File changed:[/bold yellow] {rel_path}")

        # Instant syntax check — no API call, catches missing parens/colons immediately
        syntax_error = _syntax_check(abs_path)
        if syntax_error:
            console.print(f"[bold red]Syntax error:[/bold red] {syntax_error}")

        # Delete stale chunks for this file
        collection = self._chroma_client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )
        if rel_path in old_hashes:
            try:
                collection.delete(where={"file_path": rel_path})
            except Exception:
                pass

        # Re-chunk, embed, store only this file
        chunks = chunk_file(abs_path, self._repo_dir, self._repo_url)
        if not chunks:
            console.print(f"[dim]No chunks produced for {rel_path}.[/dim]")
            return

        console.print(f"[dim]Re-embedding {len(chunks)} chunk(s)...[/dim]")
        embedded = embed_chunks_batch(chunks, self._voyage_client)
        store_in_chroma(embedded, self._chroma_client, self._collection_name)

        # Update manifest for this file only
        updated_hashes = dict(old_hashes)
        updated_hashes[rel_path] = new_hash
        save_manifest(
            self._chroma_persist_dir,
            self._collection_name,
            self._repo_url,
            updated_hashes,
        )
        console.print(f"[green]Re-indexed:[/green] {rel_path}")

        # Fast LLM error check: fetch chunks directly by file_path (no VoyageAI query needed)
        result = collection.get(
            where={"file_path": rel_path},
            include=["documents", "metadatas"],
        )
        if not result["documents"]:
            return

        context = "\n\n".join(
            f"### {m['file_path']}:{m['start_line']}-{m['end_line']}\n{d}"
            for d, m in zip(result["documents"], result["metadatas"])
        )

        console.print(f"[bold cyan]Running LLM error check...[/bold cyan]")
        try:
            response = self._groq_client.chat.completions.create(
                model=GROQ_FAST_MODEL,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Code context:\n{context}\n\n"
                            f"Find any bugs, logic errors, or issues in {rel_path}. "
                            "Be concise. If no issues, say so."
                        ),
                    },
                ],
            )
            answer = response.choices[0].message.content or "No findings."
            console.print(f"\n[bold]Findings:[/bold]\n{answer}\n")
        except Exception as exc:
            console.print(f"[red]LLM check failed:[/red] {exc}")


def start_watcher(
    repo_url: str,
    voyage_client: voyageai.Client,
    chroma_client: chromadb.PersistentClient,
    graph,
    graph_config: dict,
    chroma_persist_dir: str = CHROMA_PERSIST_DIR,
    repos_cache_dir: str = REPOS_CACHE_DIR,
) -> None:
    collection_name = derive_collection_name(repo_url)
    repo_dir = (Path(repos_cache_dir) / collection_name).resolve()

    if not repo_dir.exists():
        console.print(f"[red]Repo directory not found:[/red] {repo_dir}")
        console.print("Run with [bold]--index-first[/bold] to index the repository first.")
        return

    handler = ChangeHandler(
        repo_url=repo_url,
        repo_dir=repo_dir,
        collection_name=collection_name,
        voyage_client=voyage_client,
        chroma_client=chroma_client,
        graph_config=graph_config,
        chroma_persist_dir=chroma_persist_dir,
    )

    observer = Observer()
    observer.schedule(handler, str(repo_dir), recursive=True)
    observer.start()
    console.print(
        f"[bold green]Watching[/bold green] {repo_dir} for changes. "
        "Press [bold]Ctrl+C[/bold] to stop."
    )
    try:
        while observer.is_alive():
            observer.join(timeout=1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    console.print("[dim]Watcher stopped.[/dim]")
