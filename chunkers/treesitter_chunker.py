from __future__ import annotations

import hashlib
from typing import Optional

import tiktoken

from config import MAX_CHUNK_TOKENS
from models import CodeChunk
from .base import BaseChunker
from .fixed_size_chunker import FixedSizeChunker

_enc = tiktoken.get_encoding("cl100k_base")

# Node types that define a named symbol boundary per language
_SYMBOL_NODE_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {
        "function_declaration", "function_expression", "arrow_function",
        "class_declaration", "method_definition", "export_statement",
    },
    "typescript": {
        "function_declaration", "function_expression", "arrow_function",
        "class_declaration", "method_definition", "export_statement",
        "interface_declaration", "type_alias_declaration",
    },
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "enum_item", "trait_item"},
}


def _load_language(language: str):
    """Lazily load tree-sitter Language object."""
    try:
        if language == "python":
            import tree_sitter_python as ts_lang
        elif language in ("javascript",):
            import tree_sitter_javascript as ts_lang
        elif language in ("typescript",):
            import tree_sitter_typescript as ts_lang
            return ts_lang.language_typescript()
        elif language == "go":
            import tree_sitter_go as ts_lang
        elif language == "rust":
            import tree_sitter_rust as ts_lang
        else:
            return None
        return ts_lang.language()
    except ImportError:
        return None


def _get_symbol_name(node, content_bytes: bytes, language: str) -> Optional[str]:
    """Extract the identifier name from a symbol node."""
    name_field = node.child_by_field_name("name")
    if name_field:
        return content_bytes[name_field.start_byte:name_field.end_byte].decode("utf-8", errors="replace")
    return None


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


class TreeSitterChunker(BaseChunker):
    def __init__(self):
        self._fallback = FixedSizeChunker()

    def chunk(self, file_path: str, content: str, language: str) -> list[CodeChunk]:
        try:
            return self._chunk_with_treesitter(file_path, content, language)
        except Exception:
            return self._fallback.chunk(file_path, content, language)

    def _chunk_with_treesitter(self, file_path: str, content: str, language: str) -> list[CodeChunk]:
        from tree_sitter import Language, Parser

        lang_obj = _load_language(language)
        if lang_obj is None:
            return self._fallback.chunk(file_path, content, language)

        ts_language = Language(lang_obj)
        parser = Parser(ts_language)

        content_bytes = content.encode("utf-8")
        tree = parser.parse(content_bytes)

        symbol_types = _SYMBOL_NODE_TYPES.get(language, set())
        symbols: list[tuple] = []  # (node, symbol_name)

        def walk(node):
            if node.type in symbol_types:
                name = _get_symbol_name(node, content_bytes, language)
                symbols.append((node, name))
            else:
                for child in node.children:
                    walk(child)

        walk(tree.root_node)

        if not symbols:
            return self._fallback.chunk(file_path, content, language)

        chunks: list[CodeChunk] = []
        for node, symbol_name in symbols:
            chunk_content = content_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            # If chunk is too large, split it via fixed-size
            if _count_tokens(chunk_content) > MAX_CHUNK_TOKENS:
                sub_chunks = self._fallback.chunk(file_path, chunk_content, language)
                for i, sc in enumerate(sub_chunks):
                    sc["start_line"] = start_line + sc["start_line"] - 1
                    sc["end_line"] = start_line + sc["end_line"] - 1
                    sc["symbol_name"] = symbol_name
                    sc["file_path"] = file_path
                    chunks.extend(sub_chunks)
                continue

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
                    content=chunk_content,
                    language=language,
                    symbol_name=symbol_name,
                )
            )

        return chunks
