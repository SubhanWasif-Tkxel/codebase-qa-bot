from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from models import CodeChunk

EXTENSION_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".h": "c",
    ".java": "java",
    ".rb": "ruby",
    ".md": "markdown",
    ".txt": "text",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".sh": "bash",
}


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, file_path: str, content: str, language: str) -> list[CodeChunk]:
        ...

    @staticmethod
    def detect_language(file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        return EXTENSION_MAP.get(ext, "unknown")
