# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the bot

```bash
# First-time setup: index a repo then ask a question
python main.py --repo https://github.com/org/repo --index-first --question "How does auth work?"

# Re-use an existing index (faster — skips clone + embed)
python main.py --repo https://github.com/org/repo --question "How does auth work?"

# Interactive multi-turn session
python main.py --repo https://github.com/org/repo --interactive

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

The system has three distinct phases that run in sequence for each question:

**1. Ingest** (`ingest.py`) — one-time per repo. Clones/pulls into `.repos/<collection>/`, walks files, chunks them, embeds via VoyageAI `voyage-code-3`, and upserts into a local ChromaDB collection at `.chroma/`. Collection names are derived from the GitHub URL (`org/repo` → `org__repo`). Re-running is idempotent via `collection.upsert()`.

**2. LangGraph graph** (`graph.py`) — the core state machine. Built with `StateGraph(BotState)` and compiled once in `build_graph()`. Nodes are plain functions `(state: BotState, config: RunnableConfig) -> dict`. All external clients (Groq, VoyageAI, ChromaDB, MCP) are injected via `config["configurable"]` — the key `"anthropic_client"` holds the `Groq` instance (legacy naming from when the project used Anthropic).

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

**3. GitHub MCP** (`github_mcp.py`) — `GitHubMCPClient` is a context manager that manages an `npx` subprocess over stdio JSON-RPC 2.0. Used to read commits/issues and write new issues. If the subprocess is unavailable, all methods return empty lists gracefully.

## Key design constraints

**Chunking** (`chunkers/`): tree-sitter splits by function/class boundary for Python, JS, TS, Go, Rust. All other languages and parse failures fall back to `FixedSizeChunker` (400-token windows, 80-token overlap). Chunk IDs are `sha256(file_path + str(start_line))[:16]` — this is the upsert key.

**VoyageAI free tier**: 3 RPM limit. `ingest.py` enforces a 21-second sleep between embedding batches (batch size 8). `embed_query` in `retrieval.py` uses tenacity with `min=21s` wait. If you hit `RateLimitError`, the retry will eventually succeed — don't reduce the wait.

**LangGraph node signatures**: Nodes that need clients must declare `config: RunnableConfig` as the second parameter (typed exactly as `RunnableConfig` from `langchain_core.runnables`). LangGraph inspects the signature — a plain `dict` type annotation causes the config to not be passed.

**Answer format**: Claude/Groq is prompted to cite sources as `[[file_path:line_number]]` and end responses with `<confidence>0.XX</confidence>`. Both are parsed in `synthesize.py` via regex. The confidence tag is stripped before display.

**ChromaDB distances**: ChromaDB returns cosine *distance* in `[0, 2]`. Similarity is computed as `score = 1 - distance / 2`. The routing threshold `RETRIEVAL_CONFIDENCE_THRESHOLD = 0.50` applies to this converted score.

## Tunable constants (`config.py`)

| Constant | Default | Effect |
|---|---|---|
| `RETRIEVAL_CONFIDENCE_THRESHOLD` | 0.50 | Below this → retry or file ticket |
| `SYNTHESIS_CONFIDENCE_THRESHOLD` | 0.70 | Below this → file ticket even if answer was generated |
| `MAX_RETRIEVAL_ATTEMPTS` | 2 | Max retrieval loops before giving up |
| `TOP_K_CHUNKS` | 8 | Chunks fetched per query |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Main synthesis model |
| `GROQ_FAST_MODEL` | `llama-3.1-8b-instant` | Used for rephrase, clarify, issue draft |
