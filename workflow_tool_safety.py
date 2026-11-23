"""Host-configured safety policy for the bundled filesystem tools."""

from __future__ import annotations

import copy
import json
import os
import warnings
from pathlib import Path

from workflow_logging import record_event

DEFAULT_TOOL_SAFETY_CONFIG = Path(__file__).parent / "tools" / "safety.json"


class ToolSafetyPolicy:
    def __init__(self, config_path=DEFAULT_TOOL_SAFETY_CONFIG):
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or set(raw) - {
            "delete_requires_human",
            "block_system_edits",
            "additional_protected_paths",
        }:
            raise ValueError("Invalid tool safety configuration.")
        for name in ("delete_requires_human", "block_system_edits"):
            if type(raw.get(name, True)) is not bool:
                raise ValueError(f"{name} must be a Boolean.")
            setattr(self, name, raw.get(name, True))
        extra = raw.get("additional_protected_paths", [])
        if not isinstance(extra, list) or any(
            not isinstance(p, str) or not Path(p).is_absolute() for p in extra
        ):
            raise ValueError("additional_protected_paths must contain absolute paths.")
        if os.name == "nt":
            defaults = [
                os.environ.get("SystemRoot", "C:/Windows"),
                os.environ.get("WINDIR", "C:/Windows"),
                os.environ.get("ProgramFiles", "C:/Program Files"),
                os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"),
                os.environ.get("ProgramData", "C:/ProgramData"),
            ]
            drive = Path(defaults[0]).anchor
            defaults += [
                str(Path(drive) / name)
                for name in (
                    "Boot",
                    "EFI",
                    "System Volume Information",
                    "bootmgr",
                    "pagefile.sys",
                    "swapfile.sys",
                    "hiberfil.sys",
                )
            ]
        else:
            defaults = [
                "/etc",
                "/bin",
                "/sbin",
                "/usr",
                "/lib",
                "/lib64",
                "/boot",
                "/dev",
                "/proc",
                "/sys",
                "/System",
                "/Library",
            ]
        # Keep both lexical and resolved roots, including symlink targets.
        self.protected_paths = tuple(
            {
                p
                for raw_path in defaults + extra
                for p in (Path(os.path.abspath(raw_path)), Path(raw_path).resolve())
            }
        )

    def check_system_path(self, context, tool, raw_path, workspace_root, notify):
        if not self.block_system_edits or tool.name not in {
            "fs.write_text",
            "fs.make_directory",
            "fs.delete_file",
        }:
            return
        candidate = workspace_root / raw_path
        candidates = (Path(os.path.abspath(candidate)), candidate.resolve())
        if any(p.is_relative_to(root) for p in candidates for root in self.protected_paths):
            event = {
                "kind": "system_edit_blocked",
                "tool": tool.name,
                "path": str(candidates[1]),
                "message": "System-file modification blocked.",
            }
            emit_security_event(context, event, notify)
            raise PermissionError(event["message"])


def emit_security_event(context, event, notify):
    """Persist before notifying. Callback return values never grant permission."""
    event = {**event, "node_id": context.node.id, "execution_id": context.execution.id}
    context.execution.local.setdefault("tool_security_events", []).append(copy.deepcopy(event))
    context.task.revision += 1
    record_event(context.task, "security.alert", event, context.execution)
    if notify is None:
        warnings.warn(event["message"], UserWarning, stacklevel=3)
    else:
        notify(copy.deepcopy(event))
