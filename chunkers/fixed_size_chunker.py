from __future__ import annotations

import hashlib

import tiktoken

from config import FIXED_CHUNK_SIZE, FIXED_CHUNK_OVERLAP
from models import CodeChunk
from .base import BaseChunker

_enc = tiktoken.get_encoding("cl100k_base")


class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = FIXED_CHUNK_SIZE, overlap: int = FIXED_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _snap_to_line(self, content: str, char_offset: int) -> int:
        """Snap a char offset backward to the nearest newline."""
        idx = content.rfind("\n", 0, char_offset)
        return idx + 1 if idx != -1 else 0

    def chunk(self, file_path: str, content: str, language: str) -> list[CodeChunk]:
        if not content.strip():
            return []

        tokens = _enc.encode(content)
        if not tokens:
            return []

        chunks: list[CodeChunk] = []
        step = max(1, self.chunk_size - self.overlap)
        token_start = 0

        while token_start < len(tokens):
            token_end = min(token_start + self.chunk_size, len(tokens))
            chunk_text = _enc.decode(tokens[token_start:token_end])

            # Find line number by counting newlines up to chunk start
            prefix_text = _enc.decode(tokens[:token_start])
            start_line = prefix_text.count("\n") + 1
            end_line = start_line + chunk_text.count("\n")

            chunk_id = hashlib.sha256(
                f"{file_path}:{start_line}".encode()
            ).hexdigest()[:16]

            chunks.append(
                CodeChunk(
                    chunk_id=chunk_id,
                    repo_url="",
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    content=chunk_text,
                    language=language,
                    symbol_name=None,
                )
            )

            if token_end >= len(tokens):
                break
            token_start += step

        return chunks
