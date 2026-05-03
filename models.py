from __future__ import annotations

from typing import Dict, List, Literal, Optional, TypedDict


class CodeChunk(TypedDict):
    chunk_id: str
    repo_url: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str
    symbol_name: Optional[str]


class RetrievedChunk(TypedDict):
    chunk: CodeChunk
    score: float


class GitHubContext(TypedDict):
    recent_commits: List[Dict]
    open_issues: List[Dict]
    codeowners: Dict[str, str]


class BotState(TypedDict):
    # Input
    repo_url: str
    question: str
    collection_name: str

    # Retrieval
    query_embedding: Optional[List[float]]
    retrieved_chunks: List[RetrievedChunk]
    retrieval_attempts: int
    best_retrieval_score: float
    github_context: Optional[GitHubContext]

    # Synthesis
    answer: Optional[str]
    answer_confidence: Optional[float]
    cited_files: List[str]
    clarifying_question: Optional[str]

    # Ticket routing
    owner_handle: Optional[str]
    created_issue_url: Optional[str]

    # Control
    routing_decision: Optional[Literal["synthesize_answer", "ask_clarifying_question", "detect_owner", "expand_query"]]
    error: Optional[str]
    conversation_history: List[Dict]
