"""User-editable file/folder checks. Run with python example_edge_checks.py."""

from pathlib import Path
from tempfile import TemporaryDirectory

from workflow_core import EdgeCheckContext, Task, WorkflowDefinition, WorkflowEngine
from workflow_types import JsonDict


def files_created(context: EdgeCheckContext, config: JsonDict) -> bool:
    folder = Path(context.task.data["output_folder"])
    return all((folder / name).is_file() for name in config["files"])


def folder_ready(context: EdgeCheckContext, config: JsonDict) -> bool:
    folder = Path(context.task.data["output_folder"])
    return folder.is_dir() and sum(
        path.is_file() for path in folder.glob(config.get("pattern", "*"))
    ) >= config.get("min_files", 1)


def main():
    workflow = WorkflowDefinition.load(Path(__file__).parent / "examples" / "edge_checks.json")
    engine = WorkflowEngine()
    engine.register_entry_check("files_created", files_created)
    engine.register_entry_check("folder_ready", folder_ready)
    with TemporaryDirectory() as directory:
        task = Task.create(workflow, {"output_folder": directory})
        print("Before files exist:", engine.choose_next_nodes(workflow, task, "build"))
        (Path(directory) / "report.txt").write_text("Example report", encoding="utf-8")
        print("After files exist:", engine.choose_next_nodes(workflow, task, "build"))


if __name__ == "__main__":
    main()
