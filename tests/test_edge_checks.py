import tempfile
import unittest
from pathlib import Path

from example_edge_checks import files_created, folder_ready
from workflow_core import EdgeDefinition, NodeResult, Task, WorkflowDefinition, WorkflowEngine


class EdgeCheckTests(unittest.TestCase):
    def setup_workflow(self, edges):
        workflow = WorkflowDefinition.from_dict(
            {
                "id": "test",
                "start": ["source"],
                "nodes": [
                    {"id": name, "type": "test"} for name in ("source", "target", "fallback")
                ],
                "edges": edges,
            }
        )
        return workflow, Task.create(workflow), WorkflowEngine()

    def edge(self, **kwargs):
        return {"from": "source", "to": "target", **kwargs}

    def test_omitted_and_null_default_true(self):
        for extra in ({}, {"entry_check": None}):
            workflow, task, engine = self.setup_workflow([self.edge(**extra)])
            self.assertTrue(engine.can_enter_edge(workflow, task, workflow.edges[0]))
            self.assertEqual(engine.choose_next_nodes(workflow, task, "source"), ["target"])

    def test_false_blocks_actual_scheduling_true_allows(self):
        for allowed in (False, True):
            with self.subTest(allowed=allowed):
                workflow, task, engine = self.setup_workflow([self.edge(entry_check="gate")])
                engine.register_entry_check("gate", lambda context, config: allowed)
                engine.register("test", lambda context: NodeResult())
                engine.run_execution(workflow, task, task.executions[0])
                self.assertEqual(
                    [e.node_id for e in task.executions[1:] if e.status != "skipped"],
                    ["target"] if allowed else [],
                )
                self.assertEqual(task.status, "running" if allowed else "completed")

    def test_when_short_circuits_check(self):
        workflow, task, engine = self.setup_workflow(
            [self.edge(when={"eq": [1, 2]}, entry_check="unregistered")]
        )
        self.assertEqual(engine.choose_next_nodes(workflow, task, "source"), [])

    def test_fallback_checks_and_blocked_regular_edge(self):
        workflow, task, engine = self.setup_workflow(
            [
                self.edge(entry_check="regular"),
                self.edge(to="fallback", otherwise=True, entry_check="backup"),
            ]
        )
        engine.register_entry_check("regular", lambda context, config: True)
        # An unused fallback does not need to be evaluated.
        self.assertEqual(engine.choose_next_nodes(workflow, task, "source"), ["target"])
        engine.register_entry_check("regular", lambda context, config: False)
        engine.register_entry_check("backup", lambda context, config: True)
        self.assertEqual(engine.choose_next_nodes(workflow, task, "source"), ["fallback"])
        engine.register_entry_check("backup", lambda context, config: False)
        self.assertEqual(engine.choose_next_nodes(workflow, task, "source"), [])

    def test_json_round_trip_and_config_isolation(self):
        raw = self.edge(entry_check="gate", entry_check_config={"files": ["report.txt"]})
        workflow, task, engine = self.setup_workflow([raw])

        def check(context, config):
            self.assertIs(context.workflow, workflow)
            self.assertIs(context.task, task)
            self.assertIs(context.edge, workflow.edges[0])
            config["files"].append("temporary")
            return True

        engine.register_entry_check("gate", check)
        engine.choose_next_nodes(workflow, task, "source")
        self.assertEqual(workflow.edges[0].to_dict(), raw)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workflow.json"
            workflow.save(path)
            self.assertEqual(WorkflowDefinition.load(path).to_dict(), workflow.to_dict())
        self.assertEqual(EdgeDefinition("source", "target").to_dict(), self.edge())

    def test_files_and_folder_criteria(self):
        workflow, task, engine = self.setup_workflow(
            [
                self.edge(entry_check="files", entry_check_config={"files": ["report.txt"]}),
                self.edge(
                    to="fallback",
                    entry_check="folder",
                    entry_check_config={"pattern": "*.txt", "min_files": 2},
                ),
            ]
        )
        engine.register_entry_check("files", files_created)
        engine.register_entry_check("folder", folder_ready)
        with tempfile.TemporaryDirectory() as directory:
            task.data["output_folder"] = directory
            self.assertEqual(engine.choose_next_nodes(workflow, task, "source"), [])
            (Path(directory) / "report.txt").write_text("done", encoding="utf-8")
            self.assertEqual(engine.choose_next_nodes(workflow, task, "source"), ["target"])
            (Path(directory) / "second.txt").write_text("done", encoding="utf-8")
            self.assertEqual(
                engine.choose_next_nodes(workflow, task, "source"), ["target", "fallback"]
            )

    def test_checks_see_committed_node_outputs(self):
        workflow, task, engine = self.setup_workflow([self.edge(entry_check="ready")])
        workflow.nodes[0].outputs = {"ready": "/ready"}
        engine.register("test", lambda context: NodeResult(values={"ready": True}))
        engine.register_entry_check("ready", lambda context, config: context.task.data["ready"])
        engine.run_execution(workflow, task, task.executions[0])
        self.assertEqual(task.executions[1].node_id, "target")

    def test_errors_prevent_all_successor_creation(self):
        def raises(context, config):
            raise RuntimeError("check failed")

        for callback, error in (
            (None, KeyError),
            (raises, RuntimeError),
            (lambda c, p: 1, TypeError),
            (lambda c, p: "True", TypeError),
            (lambda c, p: None, TypeError),
        ):
            with self.subTest(error=error, callback=callback):
                workflow, task, engine = self.setup_workflow(
                    [self.edge(), self.edge(to="fallback", entry_check="gate")]
                )
                if callback is not None:
                    engine.register_entry_check("gate", callback)
                engine.register("test", lambda context: NodeResult())
                result = engine.run_execution(workflow, task, task.executions[0])
                self.assertEqual(result.status, "failed")
                self.assertIn(f"Scheduling: {error.__name__}", result.error)
                self.assertEqual(len(task.executions), 1)
                self.assertEqual(task.executions[0].status, "failed")
                self.assertEqual(task.status, "failed")

    def test_scheduling_retry_resumes_without_rerunning_handler(self):
        workflow, task, engine = self.setup_workflow([self.edge(entry_check="gate")])
        runs = []
        engine.register("test", lambda context: runs.append(context.node.id) or NodeResult())
        result = engine.run_execution(workflow, task, task.executions[0])
        self.assertIn("Unknown entry check", result.error)
        self.assertEqual(task.status, "failed")
        engine.register_entry_check("gate", lambda context, config: True)
        engine.retry_execution(task, task.executions[0])
        self.assertEqual(task.status, "running")
        source = task.executions[0]
        self.assertEqual(engine.run_execution(workflow, task, source).status, "completed")
        self.assertEqual([e.node_id for e in task.executions], ["source", "target"])
        engine.run_execution(workflow, task, task.executions[1])
        self.assertEqual(runs, ["source", "target"])
        self.assertEqual(task.status, "completed")

    def test_invalid_configuration(self):
        for reference in ("", " ", 1, False, {}):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                self.setup_workflow([self.edge(entry_check=reference)])
        for config in (None, [], "bad"):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.setup_workflow([self.edge(entry_check_config=config)])
        engine = WorkflowEngine()
        with self.assertRaises(ValueError):
            engine.register_entry_check("", lambda c, p: True)
        with self.assertRaises(TypeError):
            engine.register_entry_check("gate", True)


if __name__ == "__main__":
    unittest.main()
