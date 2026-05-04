from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_VERSION = 1


def manifest_path(chroma_persist_dir: str, collection_name: str) -> Path:
    return Path(chroma_persist_dir) / f"{collection_name}.manifest.json"


def load_manifest(chroma_persist_dir: str, collection_name: str) -> dict[str, str]:
    """Return {rel_path: sha256_hex} from the saved manifest, or {} if missing/corrupt."""
    path = manifest_path(chroma_persist_dir, collection_name)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("files", {})
    except (OSError, json.JSONDecodeError, KeyError):
        return {}


def save_manifest(
    chroma_persist_dir: str,
    collection_name: str,
    repo_url: str,
    file_hashes: dict[str, str],
) -> None:
    """Atomically write manifest JSON via a temp file + os.replace."""
    path = manifest_path(chroma_persist_dir, collection_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    data = {
        "version": MANIFEST_VERSION,
        "repo_url": repo_url,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "files": file_hashes,
    }
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def hash_file(file_path: Path) -> str:
    """Return sha256 hex digest of file contents, reading in 64 KB chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()
