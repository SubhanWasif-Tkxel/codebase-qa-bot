"""
Groq-powered answer synthesis.
"""
from __future__ import annotations

import re
from typing import Optional

from groq import Groq

from config import (
    GROQ_FAST_MODEL,
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    SYSTEM_PROMPT,
)
from models import BotState, GitHubContext, RetrievedChunk

MAX_HISTORY_TURNS = 6
MAX_CHUNK_LINES = 100


def format_code_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for rc in chunks:
        c = rc["chunk"]
        header = f"### {c['file_path']}:{c['start_line']}-{c['end_line']}"
        if c.get("symbol_name"):
            header += f" (symbol: {c['symbol_name']})"
        header += f" [relevance: {rc['score']:.2f}]"

        lines = c["content"].splitlines()
        if len(lines) > MAX_CHUNK_LINES:
            lines = lines[:MAX_CHUNK_LINES] + ["... (truncated)"]
        code_block = f"```{c['language']}\n" + "\n".join(lines) + "\n```"
        parts.append(f"{header}\n{code_block}")

    return "\n\n".join(parts)


def format_github_context(ctx: Optional[GitHubContext]) -> str:
    if not ctx:
        return ""

    parts = []

    if ctx["recent_commits"]:
        commit_lines = []
        for c in ctx["recent_commits"][:10]:
            sha = c.get("sha", "")[:7]
            msg = c.get("commit", {}).get("message", c.get("message", ""))
            msg = msg.split("\n")[0][:80]
            author = (
                c.get("commit", {}).get("author", {}).get("name", "")
                or c.get("author", {}).get("login", "unknown")
            )
            date = c.get("commit", {}).get("author", {}).get("date", "")[:10]
            commit_lines.append(f"- {sha} - {msg} ({author}, {date})")
        parts.append("**Recent commits:**\n" + "\n".join(commit_lines))

    if ctx["open_issues"]:
        issue_lines = []
        for issue in ctx["open_issues"][:5]:
            num = issue.get("number", "?")
            title = issue.get("title", "")[:80]
            labels = [lb.get("name", lb) if isinstance(lb, dict) else lb for lb in issue.get("labels", [])]
            label_str = f" [{', '.join(labels)}]" if labels else ""
            issue_lines.append(f"- #{num} - {title}{label_str}")
        parts.append("**Related open issues:**\n" + "\n".join(issue_lines))

    return "\n\n".join(parts)


def _extract_confidence(text: str) -> float:
    match = re.search(r"<confidence>([\d.]+)</confidence>", text)
    if match:
        try:
            return min(1.0, max(0.0, float(match.group(1))))
        except ValueError:
            pass
    return 0.5


def _extract_cited_files(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\[\[([^\]:]+):\d+\]\]", text)))


def build_messages(state: BotState) -> list[dict]:
    """Build messages list in OpenAI/Groq format (plain string content)."""
    code_ctx = format_code_context(state.get("retrieved_chunks", []))
    github_ctx = format_github_context(state.get("github_context"))

    context_text = "<code_context>\n" + code_ctx + "\n</code_context>"
    if github_ctx:
        context_text += "\n\n<github_context>\n" + github_ctx + "\n</github_context>"

    messages: list[dict] = [{"role": "user", "content": context_text}]

    # Replay conversation history (multi-turn), trimmed to last N turns
    history = state.get("conversation_history", [])[-MAX_HISTORY_TURNS * 2:]
    for msg in history:
        messages.append({"role": msg["role"], "content": str(msg.get("content", ""))})

    messages.append({"role": "user", "content": f"Question: {state['question']}"})
    return messages


def synthesize_answer(state: BotState, client: Groq) -> dict:
    messages = build_messages(state)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=GROQ_MAX_TOKENS,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
    )

    answer_text = response.choices[0].message.content or ""
    confidence = _extract_confidence(answer_text)
    cited_files = _extract_cited_files(answer_text)
    clean_answer = re.sub(r"\s*<confidence>[\d.]+</confidence>\s*$", "", answer_text).strip()

    return {
        "answer": clean_answer,
        "answer_confidence": confidence,
        "cited_files": cited_files,
    }


def generate_clarifying_question(state: BotState, client: Groq) -> str:
    response = client.chat.completions.create(
        model=GROQ_FAST_MODEL,
        max_tokens=256,
        messages=[
            {
                "role": "user",
                "content": (
                    f"A user asked this ambiguous question about a codebase: '{state['question']}'\n\n"
                    f"Ask one specific clarifying question to help understand what they want. "
                    f"Output only the clarifying question, nothing else."
                ),
            }
        ],
    )
    return (response.choices[0].message.content or "").strip()


def draft_issue_body(state: BotState, client: Groq) -> str:
    top_chunks = "\n".join(
        f"- {r['chunk']['file_path']}:{r['chunk']['start_line']} "
        f"(score: {r['score']:.2f}, symbol: {r['chunk'].get('symbol_name') or 'N/A'})"
        for r in state.get("retrieved_chunks", [])[:5]
    )

    commits = "\n".join(
        f"- {c.get('sha', '')[:7]}: {c.get('commit', {}).get('message', '')[:80].split(chr(10))[0]}"
        for c in (state.get("github_context") or {}).get("recent_commits", [])[:5]
    )

    existing_answer = state.get("answer", "")
    answer_section = f"\n## Partial Answer Attempted\n{existing_answer}\n" if existing_answer else ""

    prompt = (
        f"A Q&A bot failed to confidently answer this question about the codebase.\n\n"
        f"**Question:** {state['question']}\n\n"
        f"**Code locations searched:**\n{top_chunks or 'None found'}\n\n"
        f"**Recent relevant commits:**\n{commits or 'None found'}\n"
        f"{answer_section}\n"
        f"Write a concise GitHub issue body in Markdown explaining:\n"
        f"1. What was asked\n"
        f"2. What was searched and why it was insufficient\n"
        f"3. What a human reviewer should investigate\n\n"
        f"Keep it under 400 words. Use Markdown formatting."
    )

    response = client.chat.completions.create(
        model=GROQ_FAST_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.choices[0].message.content or "").strip()
