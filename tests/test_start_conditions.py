import json
import tempfile
import unittest
from pathlib import Path

from workflow_core import (
    EdgeDefinition,
    JoinConflict,
    NodeDefinition,
    NodeResult,
    Task,
    TaskExecution,
    TaskPool,
    WorkflowDefinition,
    WorkflowEngine,
    merge_task_data,
    pending_human_checks,
    resolve_human_check,
)


class StartConditionTests(unittest.TestCase):
    def workflow(self, scope="task", **rule_fields):
        rule = {"scope": scope, "required_edges": ["left-arrival", "right-arrival"], **rule_fields}
        if scope == "group":
            rule.setdefault("group_by", "/batch")
        workflow = WorkflowDefinition(
            "join",
            "Join",
            [
                NodeDefinition("left", "branch"),
                NodeDefinition("right", "branch"),
                NodeDefinition("join", "join", start_condition=rule),
            ],
            [
                EdgeDefinition("left", "join", id="left-arrival"),
                EdgeDefinition("right", "join", id="right-arrival"),
            ],
            ["left", "right"],
        )
        workflow.assert_valid()
        return workflow

    def engine(self, pool=None):
        engine = WorkflowEngine(task_pool=pool)
        engine.register("branch", lambda ctx: NodeResult())
        engine.register("join", lambda ctx: NodeResult())
        return engine

    def single_branch(self, workflow, name, data):
        task = Task.create(workflow, data)
        task.executions = [e for e in task.executions if e.node_id == name]
        return task

    def run_branch(self, engine, workflow, task, name):
        execution = next(e for e in task.executions if e.node_id == name and e.status == "ready")
        return engine.run_execution(workflow, task, execution)

    def joined_ready(self, engine):
        return [
            (task, execution)
            for task, execution in engine.task_pool.ready()
            if execution.node_id == "join"
        ]

    def test_two_branches_wait_then_release_once(self):
        workflow, engine = self.workflow(), self.engine()
        task = Task.create(workflow)
        self.run_branch(engine, workflow, task, "left")
        self.assertEqual(self.joined_ready(engine), [])
        self.assertEqual(task.executions[-1].status, "waiting_start")
        self.assertEqual(task.status, "running")
        self.run_branch(engine, workflow, task, "right")
        ready = self.joined_ready(engine)
        self.assertEqual(len(ready), 1)
        self.assertIs(ready[0][0], task)
        self.assertEqual(
            {r["edge_id"] for r in ready[0][1].joined_from}, {"left-arrival", "right-arrival"}
        )
        self.assertEqual(engine.refresh_start_conditions(workflow), [])
        engine.run_execution(workflow, *ready[0])
        self.assertEqual(task.status, "completed")

    def test_join_all_legacy_field_now_enforced(self):
        workflow, engine = self.workflow(), self.engine()
        workflow.node("join").start_condition = None
        workflow.node("join").join = "all"
        task = Task.create(workflow)
        self.run_branch(engine, workflow, task, "left")
        self.assertEqual(self.joined_ready(engine), [])
        self.run_branch(engine, workflow, task, "right")
        self.assertEqual(len(self.joined_ready(engine)), 1)

    def test_no_condition_preserves_independent_execution(self):
        workflow, engine = self.workflow(), self.engine()
        workflow.node("join").start_condition = None
        task = Task.create(workflow)
        self.run_branch(engine, workflow, task, "left")
        self.run_branch(engine, workflow, task, "right")
        self.assertEqual(len(self.joined_ready(engine)), 2)

    def test_deterministically_skipped_predecessor_resolves_join(self):
        for blocker in ("when", "check"):
            workflow, engine = self.workflow(), self.engine()
            if blocker == "when":
                workflow.edges[1].when = {"eq": [1, 2]}
            else:
                workflow.edges[1].entry_check = "deny"
                engine.register_entry_check("deny", lambda *args: False)
            task = Task.create(workflow)
            self.run_branch(engine, workflow, task, "left")
            self.run_branch(engine, workflow, task, "right")
            self.assertEqual(task.completed_nodes(), {"left", "right"})
            self.assertEqual(len(self.joined_ready(engine)), 1)
            receipts = self.joined_ready(engine)[0][1].joined_from
            self.assertEqual(sum(r.get("resolution") == "skipped" for r in receipts), 1)
            self.assertEqual(task.status, "running")

    def test_duplicate_scheduling_cannot_fake_an_arrival(self):
        workflow, engine = self.workflow(), self.engine()
        task = Task.create(workflow)
        self.run_branch(engine, workflow, task, "left")
        source = task.executions[0]
        self.assertEqual(engine.schedule_after(workflow, task, source), [])
        self.assertEqual(len([e for e in task.executions if e.status == "waiting_start"]), 1)
        self.assertEqual(self.joined_ready(engine), [])

    def test_task_scope_never_combines_separate_tasks(self):
        workflow, engine = self.workflow(), self.engine()
        left = self.single_branch(workflow, "left", {})
        right = self.single_branch(workflow, "right", {})
        self.run_branch(engine, workflow, left, "left")
        self.run_branch(engine, workflow, right, "right")
        self.assertEqual(self.joined_ready(engine), [])

    def test_separate_tasks_merge_nonconflicting_nested_data(self):
        workflow, engine = self.workflow("group"), self.engine()
        left = self.single_branch(workflow, "left", {"batch": "a", "results": {"left": 1}})
        right = self.single_branch(workflow, "right", {"batch": "a", "results": {"right": 2}})
        self.run_branch(engine, workflow, left, "left")
        self.run_branch(engine, workflow, right, "right")
        owner, joined = self.joined_ready(engine)[0]
        self.assertNotIn(owner.id, [left.id, right.id])
        self.assertEqual(owner.data, {"batch": "a", "results": {"left": 1, "right": 2}})
        self.assertEqual(left.status, "merged")
        self.assertEqual(right.status, "merged")
        self.assertEqual(set(owner.metadata["joined_task_ids"]), {left.id, right.id})
        engine.run_execution(workflow, owner, joined)
        self.assertEqual(owner.status, "completed")

    def test_conflict_fails_join_without_overwriting_sources(self):
        workflow, engine = self.workflow("group"), self.engine()
        left = self.single_branch(workflow, "left", {"batch": "a", "result": {"value": 1}})
        right = self.single_branch(workflow, "right", {"batch": "a", "result": {"value": 2}})
        self.run_branch(engine, workflow, left, "left")
        self.run_branch(engine, workflow, right, "right")
        self.assertEqual(self.joined_ready(engine), [])
        self.assertEqual(left.status, "failed")
        self.assertEqual(right.status, "failed")
        self.assertIn("/result/value", left.executions[-1].error)
        self.assertEqual(left.data["result"]["value"], 1)
        self.assertEqual(right.data["result"]["value"], 2)

    def test_groups_and_generations_are_isolated(self):
        workflow, engine = self.workflow("group", generation_by="/round"), self.engine()
        for branch, batch, round_id in (("left", "a", 1), ("right", "b", 1), ("right", "a", 2)):
            task = self.single_branch(workflow, branch, {"batch": batch, "round": round_id})
            self.run_branch(engine, workflow, task, branch)
        self.assertEqual(self.joined_ready(engine), [])
        task = self.single_branch(workflow, "right", {"batch": "a", "round": 1})
        self.run_branch(engine, workflow, task, "right")
        self.assertEqual(len(self.joined_ready(engine)), 1)

    def test_consumed_arrivals_not_reused_in_next_round(self):
        workflow, engine = self.workflow(), self.engine()
        task = Task.create(workflow)
        self.run_branch(engine, workflow, task, "left")
        self.run_branch(engine, workflow, task, "right")
        engine.run_execution(workflow, *self.joined_ready(engine)[0])
        task.executions.append(TaskExecution.create("left"))
        task.status = "running"
        self.run_branch(engine, workflow, task, "left")
        self.assertEqual(self.joined_ready(engine), [])

    def test_save_reload_pool_resumes_cross_task_join(self):
        workflow, engine = self.workflow("group"), self.engine()
        left = self.single_branch(workflow, "left", {"batch": "a", "left": 1})
        self.run_branch(engine, workflow, left, "left")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.json"
            engine.task_pool.save(path)
            engine = self.engine(TaskPool.load(path))
        right = self.single_branch(workflow, "right", {"batch": "a", "right": 2})
        self.run_branch(engine, workflow, right, "right")
        self.assertEqual(len(self.joined_ready(engine)), 1)
        self.assertEqual(engine.refresh_start_conditions(workflow), [])

    def test_custom_check_can_wait_then_release(self):
        workflow, engine = self.workflow(check="gate", config={"flag": "approved"}), self.engine()
        engine.register_start_check(
            "gate", lambda context, config: context.tasks[0].metadata.get(config["flag"], False)
        )
        task = Task.create(workflow)
        self.run_branch(engine, workflow, task, "left")
        self.run_branch(engine, workflow, task, "right")
        self.assertEqual(self.joined_ready(engine), [])
        task.metadata["approved"] = True
        self.assertEqual(len(engine.refresh_start_conditions(workflow)), 1)

    def test_unknown_or_invalid_custom_check_fails_join(self):
        for callback in (None, lambda *args: "yes"):
            workflow, engine = self.workflow(check="gate"), self.engine()
            if callback:
                engine.register_start_check("gate", callback)
            task = Task.create(workflow)
            self.run_branch(engine, workflow, task, "left")
            self.run_branch(engine, workflow, task, "right")
            self.assertEqual(self.joined_ready(engine), [])
            self.assertEqual(task.status, "failed")

    def test_human_approval_precedes_edge_arrival(self):
        workflow, engine = self.workflow(), self.engine()
        workflow.node("left").post_run_human_check = True
        task = Task.create(workflow)
        self.run_branch(engine, workflow, task, "left")
        self.run_branch(engine, workflow, task, "right")
        self.assertEqual(self.joined_ready(engine), [])
        request = pending_human_checks(task)[0]
        resolve_human_check(task, request["id"], True, reviewer="tester")
        self.run_branch(engine, workflow, task, "left")
        self.assertEqual(len(self.joined_ready(engine)), 1)

    def test_prompt_includes_start_rule_and_join_provenance(self):
        workflow, engine = self.workflow(), self.engine()
        captured = []
        engine.register_llm(
            "join",
            lambda messages: captured.append(json.loads(messages[1]["content"])) or NodeResult(),
            memory_writer=lambda messages: "Join memory",
        )
        task = Task.create(workflow)
        self.run_branch(engine, workflow, task, "left")
        self.run_branch(engine, workflow, task, "right")
        engine.run_execution(workflow, *self.joined_ready(engine)[0])
        self.assertEqual(
            captured[0]["start_condition"]["required_edges"], ["left-arrival", "right-arrival"]
        )
        self.assertEqual(len(captured[0]["joined_from"]), 2)

    def test_schema_validation_and_workflow_round_trip(self):
        workflow = self.workflow("group", generation_by="/round")
        self.assertEqual(
            WorkflowDefinition.from_dict(workflow.to_dict()).to_dict(), workflow.to_dict()
        )
        bad_rules = [
            False,
            [],
            {"scope": "other"},
            {"scope": "group"},
            {"required_edges": []},
            {"required_edges": ["missing"]},
            {"generation_by": "round"},
            {"check": ""},
            {"config": []},
        ]
        for rule in bad_rules:
            workflow.node("join").start_condition = rule
            with self.subTest(rule=rule), self.assertRaises(ValueError):
                workflow.assert_valid()

    def test_missing_group_key_records_no_partial_arrivals(self):
        workflow, engine = self.workflow("group"), self.engine()
        task = self.single_branch(workflow, "left", {})
        result = self.run_branch(engine, workflow, task, "left")
        self.assertIn("Scheduling: KeyError", result.error)
        self.assertEqual(len(task.executions), 1)
        self.assertFalse(task.executions[0].scheduled)

    def test_strict_merge_rejects_values_types_and_lists(self):
        for left, right in ((1, 2), (True, 1), ([1], [2]), ({}, [])):
            with self.subTest(left=left, right=right), self.assertRaises(JoinConflict):
                merge_task_data([{"value": left}, {"value": right}])
        self.assertEqual(merge_task_data([{"a": [1]}, {"a": [1], "b": 2}]), {"a": [1], "b": 2})

    def test_parallel_edges_require_explicit_ids_and_record_both(self):
        workflow, engine = self.workflow(), self.engine()
        workflow.edges[1].from_node = "left"
        workflow.start = ["left"]
        workflow.assert_valid()
        task = Task.create(workflow)
        self.run_branch(engine, workflow, task, "left")
        self.assertEqual(len(self.joined_ready(engine)), 1)
        for edge in workflow.edges:
            edge.id = ""
        workflow.node("join").start_condition = {}
        with self.assertRaisesRegex(ValueError, "unique IDs"):
            workflow.assert_valid()

    def test_rule_change_after_release_does_not_bypass_new_rule(self):
        workflow, engine = self.workflow(), self.engine()
        task = Task.create(workflow)
        self.run_branch(engine, workflow, task, "left")
        self.run_branch(engine, workflow, task, "right")
        owner, execution = self.joined_ready(engine)[0]
        workflow.node("join").start_condition["check"] = "new-rule"
        result = engine.run_execution(workflow, owner, execution)
        self.assertEqual(result.status, "failed")
        self.assertIn("changed", result.error)

    def test_bad_edge_id_is_validation_error(self):
        workflow = self.workflow()
        workflow.edges[0].id = ["invalid"]
        with self.assertRaises(ValueError):
            workflow.assert_valid()


if __name__ == "__main__":
    unittest.main()
