"""Environment and run history of graphs saved in skill packages.

Next to each saved graph ``graphs/<id>.json`` DGS keeps ``graphs/<id>.meta.json``,
written by the framework only (never by a model):

* ``saved``: when the graph was saved and the environment at that moment;
* ``runs``: how many runs completed and failed;
* ``last_run`` / ``last_success``: time, outcome, error, node runs, child task
  and the environment of the latest run and the latest successful run.

The environment of a graph is what it depends on: whether its node types are
still registered, the allowances its nodes use and the tools those allowances
permit (as digests of their definitions), facts the host declares in
``engine.host_environment`` (for example the model in use) and the Python
version. When a skill is opened, the current environment is compared with the
one recorded at the last successful run (or at saving), and each listed graph
gets a ``status`` (``same``, ``changed``, ``incompatible`` or ``unknown``),
the concrete changes and a one-line ``hint`` for the model.

The metadata is advisory. It is not covered by the graph's hash; a damaged or
outdated file (one recorded for a different version of the graph) is ignored.
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from workflow_memory_store import atomic_write
from workflow_types import JsonDict

META_SUFFIX = ".meta.json"
MAX_CHANGES_IN_HINT = 5
MAX_VALUE_CHARS = 80


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _digest(value) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def _short(value) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= MAX_VALUE_CHARS else text[: MAX_VALUE_CHARS - 1] + "…"


def meta_path(root: Path, graph_id: str) -> Path:
    return root / "graphs" / f"{graph_id}{META_SUFFIX}"


def _graph_sha256(root: Path, graph_id: str) -> str:
    return hashlib.sha256((root / "graphs" / f"{graph_id}.json").read_bytes()).hexdigest()


# -- environment -------------------------------------------------------------


def host_environment(engine) -> JsonDict:
    """The host's declared facts, as JSON (``engine.host_environment``)."""
    facts = getattr(engine, "host_environment", None) or {}
    if callable(facts):
        facts = facts() or {}
    if not isinstance(facts, dict):
        return {}
    return json.loads(json.dumps(facts, ensure_ascii=False, default=str))


def graph_environment(engine, graph: JsonDict) -> JsonDict:
    """What the graph depends on, as available to ``engine`` now."""
    nodes = graph.get("nodes", [])
    node_types = sorted({node["type"] for node in nodes})
    allowances: dict[str, str | None] = {}
    tool_names: set[str] = set()
    for allowance_id in sorted({n["allowance"] for n in nodes if n.get("allowance")}):
        try:
            policy = engine.allowances.resolve(allowance_id)
        except (KeyError, ValueError):
            policy = None
        allowances[allowance_id] = _digest(policy.to_dict()) if policy is not None else None
        if policy is not None:
            tool_names |= {a[5:] for a in policy.actions if a.startswith("tool:")}
    lister = getattr(engine.tools, "list_tools", None)
    catalog = {tool.name: tool for tool in (lister() if callable(lister) else [])}
    tools: dict[str, str | None] = {}
    for name in sorted(tool_names):
        if name in catalog:
            tools[name] = _digest(catalog[name].to_dict())
        elif name == "graph.create" or (
            name.startswith("knowledge.") and getattr(engine, "knowledge", None) is not None
        ):
            tools[name] = "framework"
        else:
            tools[name] = None
    return {
        "python": platform.python_version(),
        "host": host_environment(engine),
        "node_types": {node_type: node_type in engine.handlers for node_type in node_types},
        "allowances": allowances,
        "tools": tools,
    }


def compare_environments(recorded: JsonDict, current: JsonDict) -> list[JsonDict]:
    """Differences that matter for reusing a graph, most serious first."""
    changes: list[JsonDict] = []

    def add(severity: str, message: str) -> None:
        changes.append({"severity": severity, "message": message})

    for node_type, was in recorded.get("node_types", {}).items():
        if was and not current.get("node_types", {}).get(node_type, False):
            add("incompatible", f"node type {node_type} is no longer registered")
    for kind, label in (("allowances", "allowance"), ("tools", "tool")):
        now = current.get(kind, {})
        for name, before in recorded.get(kind, {}).items():
            after = now.get(name)
            if before and after is None:
                add("incompatible", f"{label} {name} is no longer available")
            elif before and after != before:
                add("changed", f"{label} {name} changed")
    before_host, after_host = recorded.get("host", {}), current.get("host", {})
    for key in sorted(set(before_host) | set(after_host)):
        if before_host.get(key) != after_host.get(key):
            add(
                "changed",
                f"{key}: {_short(before_host.get(key, 'unset'))} → "
                f"{_short(after_host.get(key, 'unset'))}",
            )
    if recorded.get("python") and recorded.get("python") != current.get("python"):
        add("info", f"Python {recorded['python']} → {current.get('python')}")
    order = {"incompatible": 0, "changed": 1, "info": 2}
    return sorted(changes, key=lambda change: order[change["severity"]])


# -- the metadata file -------------------------------------------------------


def read_meta(root: Path, graph_id: str) -> JsonDict | None:
    """The metadata of a graph, or None if missing, damaged or for another version."""
    path = meta_path(root, graph_id)
    if not path.is_file() or path.is_symlink():
        return None
    try:
        meta = json.loads(path.read_text("utf-8"))
        if not isinstance(meta, dict) or meta.get("graph_sha256") != _graph_sha256(root, graph_id):
            return None
    except (OSError, ValueError):
        return None
    return meta


def _write(root: Path, graph_id: str, meta: JsonDict) -> None:
    path = meta_path(root, graph_id)
    if path.parent.is_symlink() or path.is_symlink():
        raise PermissionError("Skill graph metadata cannot be written through a link.")
    if not path.resolve().is_relative_to(root.resolve()):
        raise PermissionError("Skill graph metadata escapes the package.")
    atomic_write(path, json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))


def _fresh(root: Path, graph_id: str) -> JsonDict:
    return {
        "graph_id": graph_id,
        "graph_sha256": _graph_sha256(root, graph_id),
        "saved": None,
        "runs": {"total": 0, "completed": 0, "failed": 0},
        "last_run": None,
        "last_success": None,
    }


def record_run(
    root: Path,
    graph_id: str,
    environment: JsonDict,
    outcome: str,
    *,
    error: str | None = None,
    steps: int | None = None,
    child_task_id: str | None = None,
    saved: bool = False,
) -> JsonDict:
    """Add one run (``completed`` or ``failed``) to the graph's metadata.

    With ``saved=True`` the run is the one the graph was saved from, and the
    environment is also recorded as the environment at saving.
    """
    meta = read_meta(root, graph_id) or _fresh(root, graph_id)
    at = _now()
    if saved:
        meta["saved"] = {"at": at, "environment": copy.deepcopy(environment)}
    meta["runs"]["total"] += 1
    meta["runs"][outcome] += 1
    meta["last_run"] = {
        "at": at,
        "outcome": outcome,
        "error": error,
        "steps": steps,
        "child_task_id": child_task_id,
        "environment": copy.deepcopy(environment),
    }
    if outcome == "completed":
        meta["last_success"] = {"at": at, "environment": copy.deepcopy(environment)}
    _write(root, graph_id, meta)
    return meta


# -- advice for the model ----------------------------------------------------


def assess(root: Path, graph_id: str, graph: JsonDict, engine) -> JsonDict:
    """History, environment comparison and a hint for one saved graph."""
    meta = read_meta(root, graph_id)
    last_run = (meta or {}).get("last_run")
    history = {
        "saved_at": ((meta or {}).get("saved") or {}).get("at"),
        "runs": (meta or {}).get("runs", {"total": 0, "completed": 0, "failed": 0}),
        "last_run": None
        if not last_run
        else {
            "at": last_run["at"],
            "outcome": last_run["outcome"],
            "error": _short(last_run["error"]) if last_run.get("error") else None,
        },
    }
    baseline, compared_with = None, None
    for field, label in (("last_success", "last successful run"), ("saved", "saving")):
        if meta and meta.get(field):
            baseline, compared_with = meta[field]["environment"], label
            break
    if baseline is None and last_run:
        baseline, compared_with = last_run["environment"], "last run"
    if baseline is None or engine is None:
        return {
            "history": history,
            "environment": {"status": "unknown", "compared_with": None, "changes": []},
            "hint": "No run of this graph has been recorded yet; check its result.",
        }
    changes = compare_environments(baseline, graph_environment(engine, graph))
    severities = {change["severity"] for change in changes}
    status = (
        "incompatible"
        if "incompatible" in severities
        else "changed"
        if "changed" in severities
        else "same"
    )
    listed = "; ".join(c["message"] for c in changes[:MAX_CHANGES_IN_HINT])
    if status == "incompatible":
        hint = f"Do not reuse this graph as it is: {listed}. Adapt it or work without it."
    elif status == "changed":
        hint = (
            f"The environment changed since the {compared_with} ({listed}). "
            "Reuse it only if these changes do not matter for the task, and check its result."
        )
    elif last_run and last_run["outcome"] == "failed":
        hint = (
            f"Same environment, but the last run failed ({history['last_run']['error']}). "
            "Consider adapting the graph."
        )
    else:
        hint = f"Same environment as at the {compared_with}; safe to reuse."
    return {
        "history": history,
        "environment": {"status": status, "compared_with": compared_with, "changes": changes},
        "hint": hint,
    }
