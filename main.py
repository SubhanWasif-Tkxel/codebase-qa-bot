"""
CLI entry point for the Codebase Q&A Bot.

Usage:
    # Index repo and ask a question
    python main.py --repo https://github.com/org/repo --index-first \\
                   --question "Why is auth failing in production?"

    # Ask without re-indexing
    python main.py --repo https://github.com/org/repo \\
                   --question "How does rate limiting work?"

    # Interactive multi-turn session
    python main.py --repo https://github.com/org/repo --interactive
"""
from __future__ import annotations

import sys
from typing import Optional

import chromadb
from groq import Groq
import click
import voyageai
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from config import (
    CHROMA_PERSIST_DIR,
    GITHUB_TOKEN,
    GROQ_API_KEY,
    VOYAGE_API_KEY,
)
from github_mcp import GitHubMCPClient, parse_repo_url
from graph import build_graph
from ingest import derive_collection_name, ingest
from models import BotState

console = Console()


def _validate_env():
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not VOYAGE_API_KEY:
        missing.append("VOYAGE_API_KEY")
    if missing:
        console.print(f"[red]Missing required env vars: {', '.join(missing)}[/red]")
        console.print("Copy .env.example to .env and fill in your keys.")
        sys.exit(1)


def _make_initial_state(repo_url: str, question: str, collection_name: str) -> BotState:
    return BotState(
        repo_url=repo_url,
        question=question,
        collection_name=collection_name,
        query_embedding=None,
        retrieved_chunks=[],
        retrieval_attempts=0,
        best_retrieval_score=0.0,
        github_context=None,
        answer=None,
        answer_confidence=None,
        cited_files=[],
        clarifying_question=None,
        owner_handle=None,
        created_issue_url=None,
        routing_decision=None,
        error=None,
        conversation_history=[],
    )


def _display_result(result: BotState) -> str:
    """Pretty-print the bot result. Returns 'answer', 'ticket', or 'clarify'."""
    if result.get("error") and not result.get("answer") and not result.get("created_issue_url"):
        console.print(Panel(f"[red]Error:[/red] {result['error']}", title="Error"))
        return "error"

    if result.get("answer"):
        score = result.get("answer_confidence", 0.0) or 0.0
        color = "green" if score >= 0.7 else "yellow"
        title = f"Answer [dim](confidence: [{color}]{score:.0%}[/{color}])[/dim]"
        console.print(Panel(Markdown(result["answer"]), title=title, border_style=color))

        if result.get("cited_files"):
            console.print(f"[dim]Files cited: {', '.join(result['cited_files'])}[/dim]")

        if result.get("created_issue_url"):
            console.print(
                f"\n[yellow]Low confidence — also filed issue:[/yellow] {result['created_issue_url']}"
            )
        return "answer"

    if result.get("created_issue_url"):
        console.print(
            Panel(
                f"Could not answer confidently.\n\n"
                f"GitHub issue created: [link]{result['created_issue_url']}[/link]\n"
                + (f"Assigned to: {result.get('owner_handle', 'unassigned')}" if result.get("owner_handle") else ""),
                title="[yellow]Ticket Filed[/yellow]",
                border_style="yellow",
            )
        )
        return "ticket"

    if result.get("clarifying_question"):
        console.print(
            Panel(result["clarifying_question"], title="[blue]Clarification Needed[/blue]", border_style="blue")
        )
        return "clarify"

    console.print("[dim]No output produced.[/dim]")
    return "error"


def show_commits(repo: str, mcp_client: GitHubMCPClient, limit: int) -> None:
    owner, repo_name = parse_repo_url(repo)
    console.print(f"\n[bold]Commit history for {owner}/{repo_name}[/bold] (limit: {limit})\n")
    with console.status("Fetching commits..."):
        commits = mcp_client.get_commits(owner, repo_name, limit=limit)

    if not commits:
        console.print("[yellow]No commits returned. Check your GITHUB_TOKEN permissions.[/yellow]")
        return

    from rich.table import Table
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("SHA", style="dim", width=8, no_wrap=True)
    table.add_column("Message", ratio=3)
    table.add_column("Author", ratio=1)
    table.add_column("Date", width=11, no_wrap=True)

    for c in commits:
        sha = c.get("sha", "")[:7]
        msg = c.get("commit", {}).get("message", c.get("message", ""))
        msg = msg.split("\n")[0][:80]
        author = (
            c.get("commit", {}).get("author", {}).get("name", "")
            or c.get("author", {}).get("login", "unknown")
        )
        date = c.get("commit", {}).get("author", {}).get("date", "")[:10]
        table.add_row(sha, msg, author, date)

    console.print(table)
    console.print(f"\n[dim]{len(commits)} commit(s) shown.[/dim]")


def run_single(
    graph,
    initial_state: BotState,
    config: dict,
) -> None:
    console.print(f"\n[bold]Question:[/bold] {initial_state['question']}\n")
    with console.status("Thinking..."):
        result = graph.invoke(initial_state, config=config)
    _display_result(result)


def interactive_loop(graph, initial_state: BotState, config: dict) -> None:
    console.print(
        Panel(
            f"Codebase Q&A Bot — [bold]{initial_state['repo_url']}[/bold]\n"
            "Type your question, or 'exit' to quit.",
            title="Interactive Mode",
        )
    )

    current_state = initial_state.copy()
    waiting_for_clarification = False

    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if waiting_for_clarification:
            # Append the clarification to conversation history and reset question
            current_state["conversation_history"].append(
                {"role": "assistant", "content": current_state.get("clarifying_question", "")}
            )
            current_state["conversation_history"].append(
                {"role": "user", "content": user_input}
            )
            # Keep original question, add clarification context
            current_state["question"] = current_state["question"] + f" (clarification: {user_input})"
            current_state["clarifying_question"] = None
            current_state["retrieval_attempts"] = 0
            current_state["routing_decision"] = None
            waiting_for_clarification = False
        else:
            # New question
            current_state["question"] = user_input
            current_state["retrieval_attempts"] = 0
            current_state["best_retrieval_score"] = 0.0
            current_state["retrieved_chunks"] = []
            current_state["github_context"] = None
            current_state["answer"] = None
            current_state["answer_confidence"] = None
            current_state["cited_files"] = []
            current_state["clarifying_question"] = None
            current_state["created_issue_url"] = None
            current_state["owner_handle"] = None
            current_state["routing_decision"] = None
            current_state["error"] = None

        console.print()
        with console.status("Thinking..."):
            result = graph.invoke(current_state, config=config)

        outcome = _display_result(result)

        if outcome == "clarify":
            waiting_for_clarification = True
        elif outcome in ("answer", "ticket"):
            # Update history for multi-turn
            current_state["conversation_history"].append(
                {"role": "user", "content": user_input}
            )
            if result.get("answer"):
                current_state["conversation_history"].append(
                    {"role": "assistant", "content": result["answer"]}
                )
            # Trim history
            current_state["conversation_history"] = current_state["conversation_history"][-12:]
            waiting_for_clarification = False
            console.print("\n[dim]Ask another question or type 'exit'.[/dim]")


@click.command()
@click.option("--repo", required=True, help="GitHub repository URL")
@click.option("--question", default=None, help="Ask a single question and exit")
@click.option("--index-first", is_flag=True, help="(Re)index the repository before querying")
@click.option("--interactive", is_flag=True, help="Start an interactive Q&A session")
@click.option("--commits", is_flag=True, help="Print full commit history and exit")
@click.option("--limit", default=100, show_default=True, help="Max commits to fetch with --commits")
@click.option("--chroma-dir", default=CHROMA_PERSIST_DIR, help="ChromaDB persist directory")
@click.option("--no-github", is_flag=True, help="Disable GitHub MCP (skip commit/issue context)")
def main(
    repo: str,
    question: Optional[str],
    index_first: bool,
    interactive: bool,
    commits: bool,
    limit: int,
    chroma_dir: str,
    no_github: bool,
):
    # --commits: bypass index/graph entirely, just print git log
    if commits:
        if not GITHUB_TOKEN:
            console.print("[red]GITHUB_TOKEN is required for --commits.[/red]")
            sys.exit(1)
        with GitHubMCPClient(GITHUB_TOKEN) as mcp_client:
            show_commits(repo, mcp_client, limit)
        return

    _validate_env()

    if index_first:
        console.print(f"[bold]Indexing {repo}...[/bold]")
        collection_name = ingest(repo, chroma_dir)
    else:
        collection_name = derive_collection_name(repo)

    # Verify collection exists
    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    try:
        col = chroma_client.get_collection(collection_name)
        console.print(
            f"[dim]Using collection [bold]{collection_name}[/bold] "
            f"({col.count()} chunks)[/dim]"
        )
    except Exception:
        console.print(
            f"[red]Collection '{collection_name}' not found.[/red]\n"
            "Run with [bold]--index-first[/bold] to index the repository first."
        )
        sys.exit(1)

    groq_client = Groq(api_key=GROQ_API_KEY)
    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)

    graph = build_graph()

    if no_github or not GITHUB_TOKEN:
        if not GITHUB_TOKEN:
            console.print("[yellow]GITHUB_TOKEN not set — GitHub context disabled.[/yellow]")
        config = {
            "configurable": {
                "anthropic_client": groq_client,
                "voyage_client": voyage_client,
                "chroma_client": chroma_client,
                "mcp_client": None,
            }
        }
        _run(graph, config, repo, collection_name, question, interactive)
    else:
        with GitHubMCPClient(GITHUB_TOKEN) as mcp_client:
            config = {
                "configurable": {
                    "anthropic_client": groq_client,
                    "voyage_client": voyage_client,
                    "chroma_client": chroma_client,
                    "mcp_client": mcp_client,
                }
            }
            _run(graph, config, repo, collection_name, question, interactive)


def _run(graph, config, repo, collection_name, question, interactive):
    if not question and not interactive:
        console.print("[yellow]Provide --question or --interactive.[/yellow]")
        sys.exit(0)

    if interactive:
        initial_state = _make_initial_state(repo, "", collection_name)
        interactive_loop(graph, initial_state, config)
    else:
        initial_state = _make_initial_state(repo, question, collection_name)
        run_single(graph, initial_state, config)


if __name__ == "__main__":
    main()
