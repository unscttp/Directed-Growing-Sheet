"""User-editable hooks; run with python example_hooks.py. No model required."""

from pathlib import Path
from tempfile import TemporaryDirectory

from workflow_core import NodeResult, Task, WorkflowDefinition, WorkflowEngine


def inspect_folder(context, config):
    folder = Path(context.task.data["folder"])
    return {"files": sorted(path.name for path in folder.glob(config["pattern"]) if path.is_file())}


def write_summary(context, result, config):
    # Automatic application code; the agent does not need to invoke this.
    path = Path(context.task.data["folder"]) / config["filename"]
    path.write_text(result.values["summary"], encoding="utf-8")


def reason(context, history):
    files = context.pre_run_data["files"]
    return NodeResult(values={"summary": "Available files: " + ", ".join(files)})


def main():
    workflow = WorkflowDefinition.load(Path(__file__).parent / "examples" / "node_hooks.json")
    engine = WorkflowEngine()
    engine.register_pre_run("inspect_folder", inspect_folder)
    engine.register_post_run("write_summary", write_summary)
    engine.register_react("agent.summarize", reason)
    with TemporaryDirectory() as directory:
        (Path(directory) / "input.txt").write_text("Example input", encoding="utf-8")
        task = Task.create(workflow, {"folder": directory})
        result = engine.run_execution(workflow, task, task.executions[0])
        print("Node status:", result.status)
        print("Pre-run information:", task.executions[0].local["pre_run"])
        print(
            "Automatically written:", (Path(directory) / "summary.txt").read_text(encoding="utf-8")
        )


if __name__ == "__main__":
    main()
