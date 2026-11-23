"""Atomic task snapshots with checksum-verified JSONL memory streams.

``save_snapshot(path, document)`` writes the main JSON file plus a sidecar
directory ``<file name>.memory/`` holding each task's memory as content-addressed
JSONL files (``passive`` = objective events, ``current`` and ``archive`` =
active memory). The main file records the SHA-256 of each stream, and loading
verifies them. After the main file is replaced, stream files no longer
referenced by it are deleted, so repeated saves do not accumulate copies.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

SECTIONS = ("passive", "current", "archive")


def atomic_write(path: Path, data: bytes) -> None:
    """Write to a temporary file in the same directory, flush it, then rename."""
    path = Path(path)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _memory_dir(path: Path) -> Path:
    return path.with_name(path.name + ".memory")


def _memory_path(path: Path, section: str, digest: str) -> Path:
    if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
        raise ValueError("Invalid memory digest")
    return _memory_dir(path) / section / (digest + ".jsonl")


def _tasks(document, pool: bool):
    """Yield every task document, including child tasks of generated graphs."""
    pending = list(document.get("tasks", [])) if pool else [document]
    while pending:
        task = pending.pop()
        yield task
        for execution in task.get("executions", []):
            for graph in execution.get("memory_state", {}).get("graphs", {}).values():
                if "task" in graph:
                    pending.append(graph["task"])


def _prune(path: Path, referenced: set[tuple[str, str]]) -> None:
    for section in SECTIONS:
        directory = _memory_dir(path) / section
        if not directory.is_dir():
            continue
        for stream in directory.glob("*.jsonl"):
            if (section, stream.stem) not in referenced:
                stream.unlink(missing_ok=True)


def save_snapshot(path, document, *, pool: bool = False) -> None:
    """Persist a task (or task pool) document. ``document`` is modified in place."""
    path = Path(path)
    referenced: set[tuple[str, str]] = set()
    for task in _tasks(document, pool):
        streams = {"passive": task.pop("events"), **task.pop("active_memory")}
        manifest = {}
        for section, rows in streams.items():
            data = "".join(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
            ).encode("utf-8")
            digest = hashlib.sha256(data).hexdigest()
            target = _memory_path(path, section, digest)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.read_bytes() != data:
                atomic_write(target, data)
            manifest[section] = digest
            referenced.add((section, digest))
        task["memory_streams"] = manifest
    atomic_write(path, json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8"))
    _prune(path, referenced)


def load_snapshot(path, *, pool: bool = False):
    path = Path(path)
    document = json.loads(path.read_bytes())
    for task in _tasks(document, pool):
        if "memory_streams" not in task:
            continue  # older snapshot with embedded memory
        manifest = task.pop("memory_streams")
        if set(manifest) != set(SECTIONS) or {"events", "active_memory"} & task.keys():
            raise ValueError("Invalid task memory manifest")
        sections = {}
        for section, digest in manifest.items():
            data = _memory_path(path, section, digest).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Damaged {section} memory stream")
            sections[section] = [json.loads(line) for line in data.splitlines()]
        task["events"] = sections.pop("passive")
        task["active_memory"] = sections
    return document
