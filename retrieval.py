"""
Hybrid retrieval: code vector search + GitHub context (commits, issues, CODEOWNERS).
"""
from __future__ import annotations

import re
from typing import Optional

import chromadb
import voyageai
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    GROQ_FAST_MODEL,
    TOP_K_CHUNKS,
    VOYAGE_MODEL,
)
from github_mcp import GitHubMCPClient, parse_repo_url, parse_codeowners
from models import BotState, CodeChunk, GitHubContext, RetrievedChunk


@retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=21, max=120))
def embed_query(question: str, voyage_client: voyageai.Client) -> list[float]:
    """Embed a search query using the query input_type (different from document ingest)."""
    result = voyage_client.embed([question], model=VOYAGE_MODEL, input_type="query")
    return result.embeddings[0]


def vector_search(
    query_embedding: list[float],
    collection: chromadb.Collection,
    top_k: int = TOP_K_CHUNKS,
) -> list[RetrievedChunk]:
    """Search ChromaDB and convert distances to similarity scores."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[RetrievedChunk] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        # ChromaDB cosine distance is in [0, 2]; convert to similarity [0, 1]
        score = max(0.0, 1.0 - dist / 2.0)
        chunk = CodeChunk(
            chunk_id="",
            repo_url=meta.get("repo_url", ""),
            file_path=meta.get("file_path", ""),
            start_line=int(meta.get("start_line", 0)),
            end_line=int(meta.get("end_line", 0)),
            content=doc,
            language=meta.get("language", "unknown"),
            symbol_name=meta.get("symbol_name") or None,
        )
        chunks.append(RetrievedChunk(chunk=chunk, score=score))

    return sorted(chunks, key=lambda r: r["score"], reverse=True)


def rephrase_query(
    original_question: str,
    retrieved_chunks: list[RetrievedChunk],
    groq_client: Groq,
) -> str:
    """Use a fast Groq model to rephrase a low-scoring query for better code retrieval."""
    top_snippets = "\n".join(
        f"- {r['chunk']['file_path']}:{r['chunk']['start_line']} (score {r['score']:.2f}): "
        f"{r['chunk']['content'][:100].strip()}"
        for r in retrieved_chunks[:3]
    )

    response = groq_client.chat.completions.create(
        model=GROQ_FAST_MODEL,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": (
                    f"A code search for the following question returned low-relevance results.\n\n"
                    f"Original question: {original_question}\n\n"
                    f"Low-relevance results found:\n{top_snippets}\n\n"
                    f"Rephrase the question to be more specific and technical for searching source code. "
                    f"Use exact technical terms, function names, or error messages if implied. "
                    f"Output only the rephrased question, nothing else."
                ),
            }
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _keyword_overlap(text: str, question: str) -> bool:
    """Check if text shares significant keywords with the question."""
    question_words = set(re.findall(r"\b\w{4,}\b", question.lower()))
    text_words = set(re.findall(r"\b\w{4,}\b", text.lower()))
    return bool(question_words & text_words)


def fetch_github_context(state: BotState, mcp_client: GitHubMCPClient) -> GitHubContext:
    """Fetch commits, open issues, and CODEOWNERS from GitHub via MCP."""
    owner, repo = parse_repo_url(state["repo_url"])
    question = state["question"]

    # Get file paths from top retrieved chunks
    top_files = list(
        dict.fromkeys(
            r["chunk"]["file_path"]
            for r in state.get("retrieved_chunks", [])[:3]
            if r["chunk"]["file_path"]
        )
    )

    # Fetch commits for relevant files
    all_commits: list[dict] = []
    seen_shas: set[str] = set()
    for file_path in top_files:
        commits = mcp_client.get_commits(owner, repo, path=file_path, limit=10)
        for c in commits:
            sha = c.get("sha", "")
            if sha and sha not in seen_shas:
                seen_shas.add(sha)
                all_commits.append(c)

    # If no file-specific commits, get recent repo commits
    if not all_commits:
        all_commits = mcp_client.get_commits(owner, repo, limit=15)

    # Fetch open issues and filter by keyword overlap with the question
    all_issues = mcp_client.list_issues(owner, repo, state="open", limit=30)
    relevant_issues = [
        issue
        for issue in all_issues
        if _keyword_overlap(
            issue.get("title", "") + " " + issue.get("body", ""), question
        )
    ][:10]

    # Try to read CODEOWNERS from common locations
    codeowners: dict[str, str] = {}
    for codeowners_path in ["CODEOWNERS", ".github/CODEOWNERS", "docs/CODEOWNERS"]:
        content = mcp_client.get_file_content(owner, repo, codeowners_path)
        if content:
            codeowners = parse_codeowners(content)
            break

    return GitHubContext(
        recent_commits=all_commits[:20],
        open_issues=relevant_issues,
        codeowners=codeowners,
    )
