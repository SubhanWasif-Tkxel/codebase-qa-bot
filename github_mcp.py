"""
GitHub MCP client: wraps the @modelcontextprotocol/server-github Node.js process
over stdio using JSON-RPC 2.0.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_fixed

from config import GITHUB_TOKEN, MCP_SERVER_CMD


class MCPError(Exception):
    pass


class GitHubMCPClient:
    def __init__(self, github_token: str = GITHUB_TOKEN):
        self._token = github_token
        self._proc: Optional[subprocess.Popen] = None
        self._request_id = 0

    def __enter__(self) -> "GitHubMCPClient":
        env = {**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": self._token}
        self._proc = subprocess.Popen(
            MCP_SERVER_CMD,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        # MCP handshake
        self._send_raw(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "codebase-qa-bot", "version": "1.0.0"},
            },
        )
        # Send initialized notification
        notif = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        self._proc.stdin.write(json.dumps(notif) + "\n")
        self._proc.stdin.flush()
        return self

    def __exit__(self, *args):
        if self._proc:
            try:
                self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            self._proc = None

    def _send_raw(self, method: str, params: dict) -> dict:
        if not self._proc:
            raise MCPError("MCP client not started. Use as context manager.")
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()

        # Read until we get a response with matching id
        while True:
            line = self._proc.stdout.readline()
            if not line:
                stderr = self._proc.stderr.read()
                raise MCPError(f"MCP server closed. stderr: {stderr}")
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Skip notifications (no id)
            if "id" not in msg:
                continue
            if msg.get("id") != self._request_id:
                continue

            if "error" in msg:
                raise MCPError(f"MCP error: {msg['error']}")
            return msg.get("result", {})

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        result = self._send_raw("tools/call", {"name": tool_name, "arguments": arguments})
        # Result content is a list of content blocks
        content = result.get("content", [])
        if content and isinstance(content, list):
            text_block = next((b for b in content if b.get("type") == "text"), None)
            if text_block:
                try:
                    return json.loads(text_block["text"])
                except (json.JSONDecodeError, KeyError):
                    return {"raw": text_block.get("text", "")}
        return result

    def list_issues(
        self, owner: str, repo: str, state: str = "open", limit: int = 20
    ) -> list[dict]:
        try:
            result = self.call_tool(
                "list_issues",
                {"owner": owner, "repo": repo, "state": state, "per_page": limit},
            )
            if isinstance(result, list):
                return result
            return result.get("items", result.get("data", []))
        except Exception:
            return []

    def get_commits(
        self,
        owner: str,
        repo: str,
        path: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        try:
            params: dict = {"owner": owner, "repo": repo, "per_page": limit}
            if path:
                params["path"] = path
            result = self.call_tool("list_commits", params)
            if isinstance(result, list):
                return result
            return result.get("items", result.get("data", []))
        except Exception:
            return []

    def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        assignees: list[str],
        labels: list[str],
    ) -> dict:
        params: dict = {
            "owner": owner,
            "repo": repo,
            "title": title,
            "body": body,
        }
        if assignees:
            params["assignees"] = assignees
        if labels:
            params["labels"] = labels
        result = self.call_tool("create_issue", params)
        return result

    def get_file_content(self, owner: str, repo: str, path: str) -> str:
        try:
            result = self.call_tool(
                "get_file_contents", {"owner": owner, "repo": repo, "path": path}
            )
            if isinstance(result, dict):
                # GitHub API returns base64-encoded content
                import base64

                content = result.get("content", "")
                if content:
                    # Strip newlines from base64
                    return base64.b64decode(content.replace("\n", "")).decode(
                        "utf-8", errors="replace"
                    )
                return result.get("raw", "")
            return ""
        except Exception:
            return ""


def parse_repo_url(repo_url: str) -> tuple[str, str]:
    """Extract (owner, repo) from a GitHub URL."""
    match = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$", repo_url.rstrip("/"))
    if not match:
        raise ValueError(f"Cannot parse GitHub URL: {repo_url}")
    return match.group(1), match.group(2)


def parse_codeowners(content: str) -> dict[str, str]:
    """Parse CODEOWNERS file, returning {pattern: primary_owner} dict."""
    owners: dict[str, str] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            pattern = parts[0]
            owner = parts[1]  # first owner wins
            owners[pattern] = owner
    return owners


def match_owner_for_files(
    cited_files: list[str], codeowners: dict[str, str]
) -> Optional[str]:
    """Find the most specific CODEOWNERS match for the cited files."""
    best_match: Optional[str] = None
    best_len = 0

    for file_path in cited_files:
        for pattern, owner in codeowners.items():
            # Simple glob matching: strip leading /
            clean_pattern = pattern.lstrip("/")
            # Check if file path starts with the pattern prefix (ignoring wildcards)
            pattern_base = clean_pattern.rstrip("*").rstrip("/")
            if pattern_base and file_path.startswith(pattern_base):
                if len(pattern_base) > best_len:
                    best_len = len(pattern_base)
                    best_match = owner

    return best_match
