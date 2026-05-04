# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the bot

```bash
# First-time setup: index a repo then ask a question
python main.py --repo https://github.com/org/repo --index-first --question "How does auth work?"

# Re-use an existing index (incremental — only re-embeds changed files)
python main.py --repo https://github.com/org/repo --question "How does auth work?"

# Interactive multi-turn session
python main.py --repo https://github.com/org/repo --interactive

# Watch mode: auto-reindex on save + instant error detection
python main.py --repo https://github.com/org/repo --watch
python main.py --repo https://github.com/org/repo --index-first --watch

# Index only (no question)
python ingest.py --repo https://github.com/org/repo

# Disable GitHub MCP (skips commit/issue context, no GITHUB_TOKEN needed)
python main.py --repo https://github.com/org/repo --question "..." --no-github
```

## Prerequisites

- **Python 3.11+** — pinned via `.python-version` (pyenv). The project uses `from __future__ import annotations` in every module for 3.11 compatibility; this must be the first import in any new file that uses `list[...]` or `dict[...]` type hints.
- **Node.js 18+** — required for the GitHub MCP server (`npx @modelcontextprotocol/server-github`). Without it, GitHub context is silently disabled.
- **API keys** in `.env` (see `.env.example`): `GROQ_API_KEY`, `VOYAGE_API_KEY`, `GITHUB_TOKEN` (optional).

## Architecture

The system has four modules that work together:

**1. Ingest** (`ingest.py`) — Clones/pulls into `.repos/<collection>/`, walks files, chunks them, embeds via VoyageAI `voyage-code-3`, and upserts into ChromaDB at `.chroma/`. Collection names are derived from the GitHub URL (`org/repo` → `org__repo`). Re-running is **incremental**: only files whose `sha256` content hash has changed since the last run are re-embedded. Deleted files have their chunks removed via `collection.delete(where={"file_path": ...})`.

**2. Manifest** (`manifest.py`) — Tracks which files have been indexed. Stored at `.chroma/<collection_name>.manifest.json` as `{rel_path: sha256_hex}`. Written atomically via `os.replace()`. On the first run (no manifest), all files are treated as new. `ingest.py` and `watch.py` both read/write the manifest.

**3. LangGraph graph** (`graph.py`) — The core Q&A state machine. Built with `StateGraph(BotState)` and compiled once in `build_graph()`. Nodes are plain functions `(state: BotState, config: RunnableConfig) -> dict`. All external clients (Groq, VoyageAI, ChromaDB, MCP) are injected via `config["configurable"]` — the key `"anthropic_client"` holds the `Groq` instance (legacy naming from when the project used Anthropic).

Graph flow:
```
embed_question → retrieve_code → fetch_github_context → route
  route → [score ≥ 0.50] → synthesize_answer → assess_confidence
            [score < 0.50, attempts < 2] → expand_query → retrieve_code (loop)
            [score < 0.50, attempts ≥ 2] → detect_owner → create_github_issue
            [ambiguous question] → ask_clarifying_question
  assess_confidence → [conf ≥ 0.70] → END
                      [conf < 0.70] → detect_owner → create_github_issue
```

**4. Watch mode** (`watch.py`) — Monitors `.repos/<collection>/` with `watchdog` (FSEvents on macOS). On each saved file:
1. **Instant syntax check** via `ast.parse()` — catches missing parens/colons with exact line numbers, no API call.
2. **Incremental re-index** — re-chunks and re-embeds only the changed file.
3. **Fast LLM error check** — fetches the file's chunks directly from ChromaDB by `file_path` metadata filter (no VoyageAI query embedding), then calls `GROQ_FAST_MODEL` directly.

The watcher uses a 1.5s debounce timer per file path to collapse editor save bursts. Processing is serialized via `ThreadPoolExecutor(max_workers=1)` to prevent concurrent ChromaDB writes. Handles `on_modified`, `on_created`, and `on_moved` (for editors that use atomic rename saves).

**5. GitHub MCP** (`github_mcp.py`) — `GitHubMCPClient` is a context manager that manages an `npx` subprocess over stdio JSON-RPC 2.0. Used to read commits/issues and write new issues. If the subprocess is unavailable, all methods return empty lists gracefully.

## Key design constraints

**Chunking** (`chunkers/`): tree-sitter splits by function/class boundary for Python, JS, TS, Go, Rust. All other languages and parse failures fall back to `FixedSizeChunker` (400-token windows, 80-token overlap). Chunk IDs are `sha256(file_path + ":" + start_line)[:16]` — this is the upsert key.

**VoyageAI free tier**: 3 RPM limit. `ingest.py` enforces a 21-second sleep between embedding batches (batch size 8). `embed_query` in `retrieval.py` uses tenacity with `min=21s` wait. If you hit `RateLimitError`, the retry will eventually succeed — don't reduce the wait. Single-file re-indexing in watch mode typically stays within one batch so the sleep is skipped.

**LangGraph node signatures**: Nodes that need clients must declare `config: RunnableConfig` as the second parameter (typed exactly as `RunnableConfig` from `langchain_core.runnables`). LangGraph inspects the signature — a plain `dict` type annotation causes the config to not be passed.

**Answer format**: Groq is prompted to cite sources as `[[file_path:line_number]]` and end responses with `<confidence>0.XX</confidence>`. Both are parsed in `synthesize.py` via regex. The confidence tag is stripped before display.

**ChromaDB distances**: ChromaDB returns cosine *distance* in `[0, 2]`. Similarity is computed as `score = 1 - distance / 2`. The routing threshold `RETRIEVAL_CONFIDENCE_THRESHOLD = 0.50` applies to this converted score.

**Watch mode path resolution**: `watchdog` emits absolute paths in file events. `repo_dir` in `watch.py` must be resolved to absolute via `.resolve()` before passing to `ChangeHandler`, otherwise `abs_path.relative_to(repo_dir)` raises `ValueError` and all events are silently dropped.

## Tunable constants (`config.py`)

| Constant | Default | Effect |
|---|---|---|
| `RETRIEVAL_CONFIDENCE_THRESHOLD` | 0.50 | Below this → retry or file ticket |
| `SYNTHESIS_CONFIDENCE_THRESHOLD` | 0.70 | Below this → file ticket even if answer was generated |
| `MAX_RETRIEVAL_ATTEMPTS` | 2 | Max retrieval loops before giving up |
| `TOP_K_CHUNKS` | 8 | Chunks fetched per query |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Main synthesis model |
| `GROQ_FAST_MODEL` | `llama-3.1-8b-instant` | Used for watch mode error checks, rephrase, clarify, issue draft |
| `WATCH_DEBOUNCE_SECONDS` | 1.5 | Seconds to wait after last save event before processing |
