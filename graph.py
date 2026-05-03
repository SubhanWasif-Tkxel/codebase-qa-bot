"""
LangGraph state machine: embed → retrieve → github context → route → synthesize/expand/ticket/ask.
"""
from __future__ import annotations

import re
from typing import Optional

import chromadb
import voyageai
from groq import Groq
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from config import (
    MAX_RETRIEVAL_ATTEMPTS,
    RETRIEVAL_CONFIDENCE_THRESHOLD,
    SYNTHESIS_CONFIDENCE_THRESHOLD,
)
from github_mcp import GitHubMCPClient, match_owner_for_files, parse_repo_url
from models import BotState
from retrieval import embed_query, fetch_github_context, rephrase_query, vector_search
from synthesize import (
    draft_issue_body,
    generate_clarifying_question,
    synthesize_answer,
)

# Ambiguity: pronouns without technical context
_PRONOUN_RE = re.compile(r"\b(it|that|this|they|them|those|these|thing)\b", re.IGNORECASE)
_TECH_TERM_RE = re.compile(r"[A-Z_]{2,}|[a-z]+[A-Z]|\w+\.\w+|\w+Error|\w+Exception", re.I)


def _is_ambiguous(question: str) -> bool:
    words = question.split()
    if len(words) < 5:
        return True
    has_pronoun = bool(_PRONOUN_RE.search(question))
    has_tech = bool(_TECH_TERM_RE.search(question))
    return has_pronoun and not has_tech


def _get_clients(config: RunnableConfig) -> tuple[Groq, voyageai.Client, chromadb.PersistentClient, Optional[GitHubMCPClient]]:
    cfg = config.get("configurable", {})
    return (
        cfg["anthropic_client"],
        cfg["voyage_client"],
        cfg["chroma_client"],
        cfg.get("mcp_client"),
    )


# ─── Nodes ────────────────────────────────────────────────────────────────────

def node_embed_question(state: BotState, config: RunnableConfig) -> dict:
    _, voyage_client, _, _ = _get_clients(config)
    embedding = embed_query(state["question"], voyage_client)
    return {"query_embedding": embedding}


def node_retrieve_code(state: BotState, config: RunnableConfig) -> dict:
    _, _, chroma_client, _ = _get_clients(config)
    collection = chroma_client.get_collection(state["collection_name"])
    chunks = vector_search(state["query_embedding"], collection)
    best_score = max((r["score"] for r in chunks), default=0.0)
    attempts = state.get("retrieval_attempts", 0) + 1
    return {
        "retrieved_chunks": chunks,
        "best_retrieval_score": best_score,
        "retrieval_attempts": attempts,
    }


def node_fetch_github_context(state: BotState, config: RunnableConfig) -> dict:
    mcp_client = (config.get("configurable") or {}).get("mcp_client")
    if mcp_client is None:
        return {"github_context": None}
    try:
        ctx = fetch_github_context(state, mcp_client)
        return {"github_context": ctx}
    except Exception as e:
        return {"github_context": None, "error": str(e)}


def node_route(state: BotState, config: RunnableConfig) -> dict:
    score = state.get("best_retrieval_score", 0.0)
    attempts = state.get("retrieval_attempts", 0)
    question = state["question"]

    # Only check for ambiguity on the very first attempt to avoid loop
    if attempts == 1 and _is_ambiguous(question):
        decision = "ask_clarifying_question"
    elif score >= RETRIEVAL_CONFIDENCE_THRESHOLD:
        decision = "synthesize_answer"
    elif attempts < MAX_RETRIEVAL_ATTEMPTS:
        decision = "expand_query"
    else:
        decision = "detect_owner"

    return {"routing_decision": decision}


def node_expand_query(state: BotState, config: RunnableConfig) -> dict:
    anthropic_client, voyage_client, _, _ = _get_clients(config)
    rephrased = rephrase_query(state["question"], state["retrieved_chunks"], anthropic_client)
    new_embedding = embed_query(rephrased, voyage_client)
    return {"query_embedding": new_embedding}


def node_synthesize_answer(state: BotState, config: RunnableConfig) -> dict:
    anthropic_client, _, _, _ = _get_clients(config)
    return synthesize_answer(state, anthropic_client)


def node_assess_confidence(state: BotState, config: RunnableConfig) -> dict:
    # Pure routing node — no external calls. Just reads state.
    return {}


def node_detect_owner(state: BotState, config: RunnableConfig) -> dict:
    codeowners = (state.get("github_context") or {}).get("codeowners", {})
    cited = state.get("cited_files", [])

    if not cited:
        # Fall back to top retrieved chunk files
        cited = [r["chunk"]["file_path"] for r in state.get("retrieved_chunks", [])[:3]]

    owner = match_owner_for_files(cited, codeowners)
    return {"owner_handle": owner}


def node_create_github_issue(state: BotState, config: RunnableConfig) -> dict:
    anthropic_client, _, _, mcp_client = _get_clients(config)
    if mcp_client is None:
        return {"error": "No MCP client available to create GitHub issue"}

    try:
        owner, repo = parse_repo_url(state["repo_url"])
    except ValueError as e:
        return {"error": str(e)}

    body = draft_issue_body(state, anthropic_client)
    assignees = []
    if state.get("owner_handle"):
        assignees = [state["owner_handle"].lstrip("@")]

    title = f"[Q&A Bot] Unanswerable: {state['question'][:80]}"
    try:
        result = mcp_client.create_issue(
            owner=owner,
            repo=repo,
            title=title,
            body=body,
            assignees=assignees,
            labels=["needs-investigation", "bot-generated"],
        )
        url = result.get("html_url") or result.get("url") or ""
        return {"created_issue_url": url}
    except Exception as e:
        return {"error": f"Failed to create issue: {e}"}


def node_ask_clarifying_question(state: BotState, config: RunnableConfig) -> dict:
    anthropic_client, _, _, _ = _get_clients(config)
    question = generate_clarifying_question(state, anthropic_client)
    return {"clarifying_question": question}


# ─── Conditional edge functions ───────────────────────────────────────────────

def route_after_route_node(state: BotState) -> str:
    return state.get("routing_decision", "detect_owner")


def route_after_assess(state: BotState) -> str:
    conf = state.get("answer_confidence", 0.0) or 0.0
    if conf >= SYNTHESIS_CONFIDENCE_THRESHOLD:
        return "end_answer"
    return "detect_owner"


# ─── Graph builder ────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(BotState)

    g.add_node("embed_question", node_embed_question)
    g.add_node("retrieve_code", node_retrieve_code)
    g.add_node("fetch_github_context", node_fetch_github_context)
    g.add_node("route", node_route)
    g.add_node("expand_query", node_expand_query)
    g.add_node("synthesize_answer", node_synthesize_answer)
    g.add_node("assess_confidence", node_assess_confidence)
    g.add_node("detect_owner", node_detect_owner)
    g.add_node("create_github_issue", node_create_github_issue)
    g.add_node("ask_clarifying_question", node_ask_clarifying_question)

    g.set_entry_point("embed_question")

    g.add_edge("embed_question", "retrieve_code")
    g.add_edge("retrieve_code", "fetch_github_context")
    g.add_edge("fetch_github_context", "route")

    g.add_conditional_edges(
        "route",
        route_after_route_node,
        {
            "synthesize_answer": "synthesize_answer",
            "expand_query": "expand_query",
            "detect_owner": "detect_owner",
            "ask_clarifying_question": "ask_clarifying_question",
        },
    )

    g.add_edge("expand_query", "retrieve_code")  # the retry loop

    g.add_edge("synthesize_answer", "assess_confidence")
    g.add_conditional_edges(
        "assess_confidence",
        route_after_assess,
        {"end_answer": END, "detect_owner": "detect_owner"},
    )

    g.add_edge("detect_owner", "create_github_issue")
    g.add_edge("create_github_issue", END)
    g.add_edge("ask_clarifying_question", END)

    return g.compile()
