"""Agent Skills packages exposed as ordinary, allowance-checked workflow tools.

``register_skill(module, directory)`` installs these tools on a ``LocalToolModule``
for a package that follows the Agent Skills layout (``SKILL.md`` with YAML
frontmatter, whose ``name`` matches the directory name):

* ``skill.<name>.invoke`` (low)  - open an invocation, returning the instructions and
  the graphs saved in the package (``graphs/<id>.json``): how the skill was run before;
* ``skill.<name>.read`` (low)    - read a file of the package;
* ``skill.<name>.finish`` (low)  - close the invocation;
* ``skill.<name>.script`` (high, human check) - only if the host registered scripts;
* ``skill.<name>.run_graph`` (high) - run one of the saved graphs as a reviewed child
  task, through the package's ``SkillGraphInterface``.

Open invocations are reconstructed from the task journal, so they survive a
save and reload. Scripts run as a separate Python process with the verified
bytes; this is not an operating-system sandbox.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

from workflow_approvals import HumanCheckRequired
from workflow_definitions import WorkflowDefinition
from workflow_graphs import DirectedGraph
from workflow_logging import record_event
from workflow_skill_graph_meta import META_SUFFIX, assess
from workflow_tools import ToolDefinition
from workflow_types import JsonDict
from workflow_utils import new_id

MAX_FILE_BYTES = 256 * 1024
MAX_OUTPUT_BYTES = 16 * 1024
SKILL_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
MANIFEST_FIELDS = ("name", "description", "compatibility", "license", "allowed-tools", "metadata")
GRAPH_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
MAX_LISTED_GRAPHS = 50
SCENARIO_PREVIEW_CHARS = 1500


def _file(root: Path, relative: str) -> Path:
    """Resolve a forward-slash path that must stay inside the package."""
    if (
        not isinstance(relative, str)
        or not relative
        or ":" in relative
        or "\\" in relative
        or PureWindowsPath(relative).anchor
        or ".." in PureWindowsPath(relative).parts
    ):
        raise ValueError("Expected a package-relative path using forward slashes.")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise PermissionError("Skill file is outside the package or not a regular file.")
    return path


def _bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("Skill file exceeds 256 KiB.")
    return data


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_manifest(root: Path) -> tuple[JsonDict, str, str]:
    """Parse and validate ``SKILL.md``; return (metadata, instructions, sha256)."""
    import yaml  # optional dependency, needed only for Agent Skills

    filename = next((n for n in ("SKILL.md", "skill.md") if (root / n).exists()), None)
    if filename is None:
        raise FileNotFoundError(f"No SKILL.md in {root}.")
    data = _bytes(_file(root, filename))
    lines = data.decode("utf-8-sig").splitlines(keepends=True)
    end = next((i for i, line in enumerate(lines[1:], 1) if line.rstrip("\r\n") == "---"), None)
    if not lines or lines[0].rstrip("\r\n") != "---" or end is None:
        raise ValueError("SKILL.md needs a delimited YAML frontmatter block.")
    meta = yaml.safe_load("".join(lines[1:end]))
    if (
        not isinstance(meta, dict)
        or meta.get("name") != root.name
        or not SKILL_NAME.fullmatch(root.name)
        or len(root.name) > 64
    ):
        raise ValueError("Invalid skill name or package directory name mismatch.")
    for field, limit in (("description", 1024), ("compatibility", 500)):
        if field == "description" or field in meta:
            value = meta.get(field)
            if not isinstance(value, str) or not value.strip() or len(value) > limit:
                raise ValueError(f"Invalid {field} in SKILL.md.")
    for field in ("allowed-tools", "license"):
        if field in meta and not isinstance(meta[field], str):
            raise ValueError(f"Invalid {field} in SKILL.md.")
    extra = meta.get("metadata", {})
    if not isinstance(extra, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in extra.items()
    ):
        raise ValueError("Skill metadata must map strings to strings.")
    metadata = {key: meta[key] for key in MANIFEST_FIELDS if key in meta}
    return metadata, "".join(lines[end + 1 :]), _digest(data)


def load_skill_graph(root: Path, graph_id: str) -> JsonDict:
    """Read and verify ``graphs/<graph_id>.json`` of a skill package.

    The file has the format ``graph.create`` writes with ``save_to_skill``: the
    graph, its start nodes, the explanation and the name and SHA-256 of the
    Markdown usage scenario next to it. Raises if any part is missing, the
    scenario was changed or the graph is invalid.
    """
    if not isinstance(graph_id, str) or not GRAPH_ID.fullmatch(graph_id):
        raise ValueError("Invalid skill graph id.")
    document = json.loads(_bytes(_file(root, f"graphs/{graph_id}.json")).decode("utf-8"))
    if not isinstance(document, dict) or not all(
        isinstance(document.get(key), expected)
        for key, expected in (
            ("graph", dict),
            ("start", list),
            ("explanation", str),
            ("usage_scenario_file", str),
            ("usage_scenario_sha256", str),
        )
    ):
        raise ValueError("Skill graph file is incomplete.")
    scenario_name = document["usage_scenario_file"]
    if "/" in scenario_name or not scenario_name.endswith(".md"):
        raise ValueError("Skill graph scenario must be a Markdown file next to the graph.")
    scenario = _bytes(_file(root, f"graphs/{scenario_name}"))
    if _digest(scenario) != document["usage_scenario_sha256"]:
        raise ValueError("Skill graph scenario does not match its recorded hash.")
    graph = DirectedGraph.from_dict(document["graph"])
    WorkflowDefinition(
        graph_id, "Skill graph", graph.nodes, graph.edges, document["start"]
    ).assert_valid()
    return {
        "id": graph_id,
        "graph": document["graph"],
        "start": document["start"],
        "explanation": document["explanation"],
        "usage_scenario": scenario.decode("utf-8"),
    }


def _title(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.strip():
            return line.strip().lstrip("#").strip()[:120]
    return ""


def list_skill_graphs(root: Path, engine=None) -> list[JsonDict]:
    """Summaries of the graphs saved in a package, newest first.

    Files that fail verification are listed with ``invalid`` and the reason
    instead of their content, so they can be noticed and are never run. With
    an ``engine``, each graph also gets its run ``history``, an ``environment``
    comparison with its last successful run and a ``hint`` (see
    ``workflow_skill_graph_meta``).
    """
    directory = root / "graphs"
    if not directory.is_dir() or directory.is_symlink():
        return []
    files = sorted(
        (
            path
            for path in directory.glob("*.json")
            if path.is_file() and not path.is_symlink() and not path.name.endswith(META_SUFFIX)
        ),
        key=lambda path: (-path.stat().st_mtime, path.name),
    )
    summaries = []
    for path in files[:MAX_LISTED_GRAPHS]:
        saved_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        try:
            graph = load_skill_graph(root, path.stem)
        except Exception as exc:
            summaries.append(
                {"id": path.stem, "saved_at": saved_at, "invalid": f"{type(exc).__name__}: {exc}"}
            )
            continue
        scenario = graph["usage_scenario"]
        summary = {
            "id": graph["id"],
            "title": _title(scenario),
            "saved_at": saved_at,
            "usage_scenario": scenario[:SCENARIO_PREVIEW_CHARS],
            "explanation": graph["explanation"],
            "nodes": [
                {"id": node["id"], "name": node.get("name", "")} for node in graph["graph"]["nodes"]
            ],
        }
        if engine is not None:
            summary.update(assess(root, graph["id"], graph["graph"], engine))
            summary["saved_at"] = summary["history"]["saved_at"] or saved_at
        summaries.append(summary)
    return summaries


def open_skill_calls(context) -> dict[str, JsonDict]:
    """Invocations opened and not yet closed in this node execution."""
    calls: dict[str, JsonDict] = {}
    for event in context.task.events:
        if event["execution_id"] != context.execution.id or event["task_id"] != context.task.id:
            continue
        details = event["details"]
        if event["type"] == "skill.invoke.opened":
            calls[details["invoke_id"]] = details
        elif event["type"] == "skill.invoke.closed":
            calls.pop(details["invoke_id"], None)
    return calls


class SkillWriteInterface:
    """Reserved host adapter for creating skills. Not implemented."""

    def write_skill(self, context, name, request):
        raise NotImplementedError("Skill writing is reserved for future implementation.")


class SkillAgentInterface:
    """Reserved: run a skill as an independent agent. Not implemented."""

    def invoke_agent(self, context, invoke_id, request):
        raise NotImplementedError(
            "Independent skill agents are reserved for future implementation."
        )


class SkillGraphInterface:
    """Runs a graph saved in a skill package, for ``skill.<name>.run_graph``.

    The default runs it as a generated graph of the calling node: the graph file
    is verified (``load_skill_graph``), every child allowance must be a subset of
    the node's, it is reviewed when ``graph_creation_human_check`` is on, and
    depth and step limits apply. The node does not need ``allow_create_graph``:
    the permission is the ``run_graph`` action in its allowance, and only graphs
    already in the package can run. Subclass it to run saved graphs another way
    and pass the instance to ``register_skill(graph_runner=...)``.
    """

    def execute_graph(self, context, invoke_id: str, graph_id: str):
        return context._require_runtime().graph_tools.run_skill_graph(context, invoke_id, graph_id)


_STRING = {"type": "string"}
_DESCRIPTIONS = {
    "read": "Read a bundled skill resource.",
    "script": "Run a host-approved skill script.",
    "finish": "Close the skill invocation.",
    "run_graph": (
        "Run a graph saved in this skill, by the id listed when the skill was opened, "
        "as a child task after human review. Returns the child task's data."
    ),
}


class SkillPackage:
    """Tool handlers for one registered skill package."""

    def __init__(
        self,
        root: Path,
        scripts,
        timeout_seconds: int,
        graph_runner: SkillGraphInterface | None = None,
    ) -> None:
        self.root = root
        self.graph_runner = graph_runner
        self.metadata, _, self.manifest_hash = read_manifest(root)
        self.name = self.metadata["name"]
        self.timeout_seconds = timeout_seconds
        self.trusted_scripts = {relative: self._script_digest(relative) for relative in scripts}

    def _script_digest(self, relative: str) -> str:
        path = _file(self.root, relative)
        if (
            not relative.startswith("scripts/")
            or path.suffix != ".py"
            or not path.is_relative_to(self.root / "scripts")
        ):
            raise ValueError("Only explicitly registered scripts/*.py files are executable.")
        return _digest(_bytes(path))

    def argument_specs(self) -> dict[str, tuple[list[str], JsonDict]]:
        specs = {
            "invoke": ([], {"parent_invoke_id": {"type": ["string", "null"]}}),
            "read": (
                ["invoke_id", "path"],
                {"invoke_id": _STRING, "path": _STRING, "encoding": {"enum": ["utf-8", "base64"]}},
            ),
            "finish": (["invoke_id"], {"invoke_id": _STRING}),
        }
        if self.trusted_scripts:
            specs["script"] = (
                ["invoke_id", "path"],
                {
                    "invoke_id": _STRING,
                    "path": {"enum": list(self.trusted_scripts)},
                    "argv": {"type": "array", "items": _STRING},
                },
            )
        if self.graph_runner is not None:
            specs["run_graph"] = (
                ["invoke_id", "graph_id"],
                {"invoke_id": _STRING, "graph_id": {"type": "string", "pattern": GRAPH_ID.pattern}},
            )
        return specs

    def tool_definitions(self) -> list[ToolDefinition]:
        definitions = []
        for op, (required, properties) in self.argument_specs().items():
            config = {"manifest_sha256": self.manifest_hash}
            if op == "script":
                config["script_hashes"] = dict(self.trusted_scripts)
            description = self.metadata["description"] if op == "invoke" else _DESCRIPTIONS[op]
            definitions.append(
                ToolDefinition(
                    f"skill.{self.name}.{op}",
                    "high" if op in {"script", "run_graph"} else "low",
                    description=description,
                    human_check=op == "script",
                    config=config,
                    input_schema={
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                )
            )
        return definitions

    # -- dispatch ----------------------------------------------------------

    def _event(self, ctx, kind: str, info: JsonDict, **details) -> None:
        record_event(ctx.task, "skill." + kind, {**info, **details}, ctx.execution)

    def __call__(self, ctx, tool: ToolDefinition, args: JsonDict):
        op = tool.name.rsplit(".", 1)[1]
        calls = open_skill_calls(ctx)
        if op == "invoke":
            invoke, parent = new_id("invoke"), args.get("parent_invoke_id")
        else:
            invoke = args.get("invoke_id")
            known = isinstance(invoke, str) and invoke in calls
            parent = calls[invoke]["parent_invoke_id"] if known else None
        info = {"skill": self.name, "invoke_id": invoke, "parent_invoke_id": parent}
        self._event(ctx, op + ".requested", info, arguments=args)
        try:
            required, properties = self.argument_specs()[op]
            if set(required) - args.keys() or args.keys() - properties.keys():
                raise ValueError("Unexpected or missing skill arguments.")
            if op == "invoke":
                return self._invoke(ctx, info, parent, calls)
            if not isinstance(invoke, str) or calls.get(invoke, {}).get("skill") != self.name:
                raise ValueError("Unknown, closed, or foreign skill invocation.")
            if op == "finish":
                return self._finish(ctx, info, invoke, calls)
            if op == "read":
                return self._read(ctx, info, args)
            if op == "run_graph":
                return self._run_graph(ctx, info, args)
            return self._script(ctx, info, args)
        except HumanCheckRequired:
            raise  # waiting for a review is not a failure
        except Exception as exc:
            self._event(ctx, op + ".failed", info, error_type=type(exc).__name__, error=str(exc))
            raise

    def _invoke(self, ctx, info, parent, calls):
        if parent is not None and (not isinstance(parent, str) or parent not in calls):
            raise ValueError("Parent must be an open invocation in this node execution.")
        self._event(ctx, "load.started", info)
        try:
            metadata, instructions, digest = read_manifest(self.root)
            if digest != self.manifest_hash:
                raise ValueError(
                    "Skill manifest changed; host must rebuild its skill registration."
                )
        except Exception as exc:
            self._event(ctx, "load.failed", info, error=str(exc))
            raise
        graphs = list_skill_graphs(self.root, getattr(ctx, "runtime", None))
        self._event(
            ctx,
            "load.completed",
            info,
            sha256=digest,
            metadata=metadata,
            graphs=[graph["id"] for graph in graphs],
        )
        self._event(ctx, "invoke.opened", info)
        return {**info, "metadata": metadata, "instructions": instructions, "graphs": graphs}

    def _finish(self, ctx, info, invoke, calls):
        if any(call["parent_invoke_id"] == invoke for call in calls.values()):
            raise ValueError("Close child invocations first.")
        self._event(ctx, "invoke.closed", info)
        return {"closed": invoke}

    def _read(self, ctx, info, args):
        relative = args["path"]
        data = _bytes(_file(self.root, relative))
        encoding = args.get("encoding", "utf-8")
        if encoding not in ("utf-8", "base64"):
            raise ValueError("Unsupported resource encoding.")
        if encoding == "utf-8":
            content = data.decode()
        else:
            content = base64.b64encode(data).decode("ascii")
        self._event(
            ctx, "read.completed", info, path=relative, bytes=len(data), sha256=_digest(data)
        )
        return {"path": relative, "encoding": encoding, "content": content}

    def _run_graph(self, ctx, info, args):
        result = self.graph_runner.execute_graph(ctx, info["invoke_id"], args["graph_id"])
        self._event(ctx, "run_graph.completed", info, graph_id=args["graph_id"])
        return result

    def _script(self, ctx, info, args):
        relative = args["path"]
        data = _bytes(_file(self.root, relative))
        digest = _digest(data)
        if self.trusted_scripts.get(relative) != digest:
            raise PermissionError("Script was not registered or its contents changed.")
        argv = args.get("argv", [])
        if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
            raise ValueError("argv must be a list of strings.")
        self._event(ctx, "script.started", info, path=relative, sha256=digest, argv=argv)
        # Run the verified bytes, not a path whose content could change after hashing.
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            process = subprocess.run(
                [sys.executable, "-I", "-", *argv],
                input=data,
                cwd=self.root,
                stdout=stdout,
                stderr=stderr,
                timeout=self.timeout_seconds,
                check=False,
            )
            result = {
                "exit_code": process.returncode,
                "stdout": _head(stdout),
                "stderr": _head(stderr),
            }
        self._event(ctx, "script.exited", info, path=relative, **result)
        if process.returncode:
            raise RuntimeError(f"Skill script exited with status {process.returncode}.")
        return result


def _head(stream) -> str:
    stream.seek(0)
    return stream.read(MAX_OUTPUT_BYTES).decode("utf-8", errors="replace")


_DEFAULT_GRAPH_RUNNER = SkillGraphInterface()


def register_skill(
    module,
    directory,
    *,
    scripts=(),
    timeout_seconds: int = 30,
    graph_runner: SkillGraphInterface | None = _DEFAULT_GRAPH_RUNNER,
) -> None:
    """Install a skill package's tools on a ``LocalToolModule``.

    Only the listed ``scripts/*.py`` files become executable, and only while
    their contents match the hash taken here. ``allowed-tools`` in the manifest
    is metadata; it grants nothing. ``graph_runner`` runs the package's saved
    graphs for ``skill.<name>.run_graph``; pass ``None`` to leave that tool out.
    """
    if type(timeout_seconds) is not int or timeout_seconds < 1:
        raise ValueError("timeout_seconds must be a positive integer.")
    if graph_runner is not None and not isinstance(graph_runner, SkillGraphInterface):
        raise TypeError("graph_runner must be a SkillGraphInterface.")
    package = SkillPackage(
        Path(directory).resolve(strict=True), scripts, timeout_seconds, graph_runner
    )
    if any(tool.name.startswith(f"skill.{package.name}.") for tool in module.list_tools()):
        raise ValueError("Skill already registered; rebuild the module to replace its permissions.")
    if not hasattr(module, "skill_packages"):
        module.skill_packages = {}
    module.skill_packages[package.name] = {
        "root": str(package.root),
        "manifest_hash": package.manifest_hash,
    }
    for definition in package.tool_definitions():
        module.register(definition, package)
