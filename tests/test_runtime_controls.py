import json
import unittest

from workflow_core import (
    AllowanceSet,
    EdgeDefinition,
    LocalToolModule,
    NodeDefinition,
    NodeResult,
    ObservedBrowser,
    ReActAction,
    Task,
    ToolDefinition,
    WorkflowDefinition,
    WorkflowEngine,
    pending_branch_selections,
    pending_human_checks,
    resolve_branch_selection,
    resolve_human_check,
    traversal_count,
)


class RuntimeControlTests(unittest.TestCase):
    def workflow(self, routing=None):
        return WorkflowDefinition(
            "controls",
            "Controls",
            [
                NodeDefinition("start", "agent", routing=routing or {}),
                NodeDefinition("left", "worker"),
                NodeDefinition("right", "worker"),
                NodeDefinition("join", "worker", join="all"),
                NodeDefinition("end", "worker"),
            ],
            [
                EdgeDefinition("start", "left", id="left"),
                EdgeDefinition("start", "right", id="right"),
                EdgeDefinition("left", "join", id="left-join"),
                EdgeDefinition("right", "join", id="right-join"),
                EdgeDefinition("join", "end", id="end"),
            ],
            ["start"],
        )

    def engine(self):
        engine = WorkflowEngine()
        engine.register("agent", lambda ctx: NodeResult())
        engine.register("worker", lambda ctx: NodeResult())
        return engine

    def finish(self, engine, workflow, limit=30):
        for _ in range(limit):
            ready = engine.task_pool.ready()
            if not ready:
                return
            for task, execution in ready:
                engine.run_execution(workflow, task, execution)
        self.fail("Scheduler failed to terminate")

    def test_unknown_fields_and_types_rejected_at_load(self):
        raw = self.workflow().to_dict()
        for location, field in (
            (raw, "nmae"),
            (raw["nodes"][0], "timeot_seconds"),
            (raw["edges"][0], "otherwize"),
            (raw["nodes"][0]["react"], "max_step"),
        ):
            location[field] = True
            with self.assertRaisesRegex(ValueError, field):
                WorkflowDefinition.from_dict(raw)
            del location[field]
        for value in ("false", 1, None):
            raw["edges"][0]["otherwise"] = value
            with self.assertRaises(ValueError):
                WorkflowDefinition.from_dict(raw)
        for ctor, raw in (
            (NodeDefinition.from_dict, {"id": "a", "type": "b", "typo": 1}),
            (EdgeDefinition.from_dict, {"from": "a", "to": "b", "otherwize": True}),
            (AllowanceSet.from_dict, {"id": "a", "actons": []}),
            (ToolDefinition.from_dict, {"name": "a", "risk_level": "low", "human_chek": True}),
        ):
            with self.assertRaises(ValueError):
                ctor(raw)

    def test_freeform_config_and_new_fields_roundtrip(self):
        w = self.workflow({"mode": "default", "default_edges": ["left", "right"]})
        w.nodes[0].config = {"custom": {"anything": 1}}
        w.nodes[0].memory_checkpoints = ["review"]
        w.nodes[0].timeout_seconds = 10
        w.edges[0].max_traversals = 3
        self.assertEqual(WorkflowDefinition.from_dict(w.to_dict()).to_dict(), w.to_dict())
        for value in (True, -1, 1.5):
            w.edges[0].max_traversals = value
            with self.assertRaises(ValueError):
                w.assert_valid()

    def test_default_single_branch_skips_other_and_join_releases(self):
        w, engine = self.workflow({"mode": "default", "default_edges": ["left"]}), self.engine()
        task = Task.create(w)
        engine.run_execution(w, task, task.executions[0])
        self.assertTrue(
            any(e.node_id == "right" and e.status == "skipped" for e in task.executions)
        )
        self.finish(engine, w)
        joined = next(e for e in task.executions if e.node_id == "join" and e.joined_from)
        self.assertEqual(
            [r.get("resolution", "entered") for r in joined.joined_from], ["entered", "skipped"]
        )
        self.assertEqual(joined.status, "completed")
        self.assertEqual(task.status, "completed")

    def test_all_skipped_never_runs_join_or_descendants(self):
        w, engine = self.workflow({"mode": "default", "default_edges": []}), self.engine()
        engine.register("worker", lambda ctx: self.fail("Skipped handler ran"))
        task = Task.create(w)
        engine.run_execution(w, task, task.executions[0])
        self.assertFalse(task.active_executions())
        self.assertEqual(task.status, "completed")
        self.assertTrue(any(e.node_id == "end" and e.status == "skipped" for e in task.executions))

    def test_multiple_defaults_execute_both(self):
        w, engine = (
            self.workflow({"mode": "default", "default_edges": ["left", "right"]}),
            self.engine(),
        )
        task = Task.create(w)
        engine.run_execution(w, task, task.executions[0])
        self.finish(engine, w)
        self.assertEqual(
            {e.node_id for e in task.executions if e.status == "completed"},
            {"start", "left", "right", "join", "end"},
        )

    def test_human_choice_restores_without_replaying_handler(self):
        w, engine = self.workflow({"mode": "human"}), self.engine()
        calls = []
        engine.register("agent", lambda ctx: calls.append("run") or NodeResult())
        task = Task.create(w)
        self.assertEqual(engine.run_execution(w, task, task.executions[0]).status, "waiting")
        self.assertEqual(len(task.executions), 1)
        task = Task.from_dict(json.loads(json.dumps(task.to_dict())))
        request = pending_branch_selections(task)[0]
        with self.assertRaises(ValueError):
            resolve_branch_selection(task, request["id"], ["unknown"], reviewer="human")
        resolve_branch_selection(task, request["id"], ["right"], reviewer="human")
        engine.run_execution(w, task, task.executions[0])
        self.assertEqual(calls, ["run"])
        self.assertEqual(traversal_count(task, "right"), 1)
        self.assertEqual(traversal_count(task, "left"), 0)

    def test_human_choice_rechecked_when_conditions_change(self):
        w, engine = self.workflow({"mode": "human"}), self.engine()
        task = Task.create(w)
        engine.run_execution(w, task, task.executions[0])
        resolve_branch_selection(
            task, pending_branch_selections(task)[0]["id"], ["right"], reviewer="human"
        )
        w.edges[1].entry_check = "deny"
        engine.register_entry_check("deny", lambda *args: False)
        engine.run_execution(w, task, task.executions[0])
        self.assertEqual(traversal_count(task, "right"), 0)

    def test_llm_selection_requires_valid_edges_and_reason(self):
        for response in (
            {"edges": ["unknown"], "reason": "because"},
            {"edges": ["left"], "reason": ""},
            {"edges": ["left", "left"], "reason": "why"},
        ):
            w, engine = self.workflow({"mode": "llm"}), self.engine()
            engine.register_branch_selector("agent", lambda messages: response)
            task = Task.create(w)
            result = engine.run_execution(w, task, task.executions[0])
            self.assertIn("Scheduling: ValueError", result.error)
            self.assertEqual(len(task.executions), 1)
        engine.register_branch_selector(
            "agent", lambda messages: {"edges": ["left"], "reason": "Left has the required input."}
        )
        engine.retry_execution(task, task.executions[0])
        engine.run_execution(w, task, task.executions[0])
        self.assertEqual(task.active_memory["current"][-1]["kind"], "decision")
        self.assertTrue(task.active_memory["current"][-1]["text"])

    def test_loop_budget_survives_restore_and_skip_cycles_terminate(self):
        w = WorkflowDefinition(
            "loop",
            "Loop",
            [NodeDefinition("a", "worker")],
            [EdgeDefinition("a", "a", id="loop", max_traversals=2)],
            ["a"],
        )
        engine, task = self.engine(), Task.create(w)
        engine.run_execution(w, task, task.executions[0])
        task = Task.from_dict(task.to_dict())
        engine.task_pool.add(task)
        self.finish(engine, w)
        self.assertEqual(traversal_count(task, "loop"), 2)
        self.assertEqual(sum(e.status == "completed" for e in task.executions), 3)
        self.assertEqual(task.status, "completed")
        self.assertLess(len(task.executions), 7)

    def test_browser_records_actual_redirect_and_failures(self):
        w, engine = self.workflow(), self.engine()
        task = Task.create(w)
        context = engine.build_context(w, task, task.executions[0])
        browser = ObservedBrowser(
            lambda url: {"url": "https://example.org/final", "title": "Final", "status": 200}
        )
        browser.open(context, "https://example.org/start")
        opened = next(e for e in task.events if e["type"] == "browser.opened")
        self.assertEqual(opened["details"]["url"], "https://example.org/final")
        browser.navigate = lambda url: (_ for _ in ()).throw(OSError("offline"))
        with self.assertRaises(OSError):
            browser.open(context, "https://example.org/")
        self.assertEqual(task.events[-1]["type"], "browser.open_failed")

    def test_checkpoint_notes_and_summary_append_and_persist(self):
        w, engine = self.workflow(), self.engine()
        w.nodes[0].memory_checkpoints = ["review"]
        requests = []

        def writer(messages):
            payload = json.loads(messages[1]["content"])
            requests.append(payload)
            return "Memory: " + payload["checkpoint"]["name"]

        engine.register_memory_writer("agent", writer)
        tools = LocalToolModule()
        tools.register(ToolDefinition("echo", "low"), lambda *args: "observed")
        engine.attach_tools(tools)

        def handler(ctx):
            ctx.use_tool("echo")
            ctx.checkpoint("review", {"ready": True})
            return NodeResult()

        engine.register("agent", handler)
        task = Task.create(w)
        engine.run_execution(w, task, task.executions[0])
        notes = task.active_memory
        self.assertEqual(
            {n["checkpoint"] for n in notes["current"] + notes["archive"]},
            {"node_start", "tool", "review", "node_summary"},
        )
        self.assertEqual(notes["current"][-1]["kind"], "summary")
        self.assertEqual(len(requests[-1]["node_notes"]), 3)
        restored = Task.from_dict(task.to_dict())
        self.assertEqual(restored.active_memory, notes)
        with self.assertRaises(ValueError):
            engine.build_context(w, task, task.executions[0]).checkpoint("typo")

    def test_default_llm_adapter_handles_memory_request_type(self):
        w, engine = self.workflow(), self.engine()
        request_types = []

        def llm(messages):
            payload = json.loads(messages[1]["content"])
            kind = payload.get("request_type", "decision")
            request_types.append(kind)
            return "Observed current state." if kind == "active_memory" else NodeResult()

        engine.register_llm("agent", llm)
        task = Task.create(w)
        engine.run_execution(w, task, task.executions[0])
        self.assertEqual(request_types, ["active_memory", "decision", "active_memory"])

    def test_writer_failure_does_not_replay_tool(self):
        w, engine = self.workflow(), self.engine()
        effects = []
        tools = LocalToolModule()
        tools.register(ToolDefinition("effect", "low"), lambda *args: effects.append(1) or "ok")
        engine.attach_tools(tools)

        def writer(messages):
            payload = json.loads(messages[1]["content"])
            if payload["checkpoint"]["name"] == "tool":
                raise RuntimeError("model unavailable")
            return "note"

        engine.register_memory_writer("agent", writer)
        engine.register_react(
            "agent", lambda ctx, history: NodeResult() if history else ReActAction("tool:effect")
        )
        task = Task.create(w)
        self.assertEqual(engine.run_execution(w, task, task.executions[0]).status, "failed")
        self.assertEqual(effects, [1])
        engine.register_memory_writer("agent", lambda messages: "recovered")
        task.executions[0].status = "ready"
        engine.run_execution(w, task, task.executions[0])
        self.assertEqual(effects, [1])

    def test_timeout_defaults_to_human_and_disabled_fails(self):
        for human in (True, False):
            w, engine = self.workflow(), self.engine()
            w.nodes[0].timeout_seconds = 10
            w.nodes[0].timeout_human_check = human
            task = Task.create(w)
            engine.task_pool.add(task)
            engine.check_timeouts(w, now=task.executions[0].timing["deadline"] + 1)
            self.assertEqual(task.executions[0].status, "waiting" if human else "failed")
            if human:
                task = Task.from_dict(task.to_dict())
                resolve_human_check(
                    task, pending_human_checks(task)[0]["id"], True, reviewer="human"
                )
                self.assertEqual(task.executions[0].status, "ready")
                self.assertEqual(
                    engine.run_execution(w, task, task.executions[0]).status, "completed"
                )

    def test_slow_handler_result_waits_without_reexecution(self):
        w, engine = self.workflow(), self.engine()
        w.nodes[0].timeout_seconds = 10
        effects = []

        def handler(ctx):
            effects.append(1)
            ctx.execution.timing["deadline"] = 0
            return NodeResult()

        engine.register("agent", handler)
        task = Task.create(w)
        self.assertEqual(engine.run_execution(w, task, task.executions[0]).status, "waiting")
        self.assertEqual(len(task.executions), 1)
        resolve_human_check(task, pending_human_checks(task)[0]["id"], True, reviewer="human")
        engine.run_execution(w, task, task.executions[0])
        self.assertEqual(effects, [1])

    def test_timeout_waiting_join_keeps_arrivals_and_stays_gated(self):
        w, engine = self.workflow(), self.engine()
        w.node("join").timeout_seconds = 1
        task = Task.create(w)
        engine.run_execution(w, task, task.executions[0])
        left = next(e for e in task.executions if e.node_id == "left")
        engine.run_execution(w, task, left)
        arrival = next(e for e in task.executions if e.node_id == "join")
        engine.check_timeouts(w, now=arrival.timing["deadline"] + 1)
        self.assertEqual(arrival.status, "waiting")
        self.assertEqual(engine.refresh_start_conditions(w), [])
        resolve_human_check(task, pending_human_checks(task)[0]["id"], True, reviewer="human")
        self.assertEqual(arrival.status, "waiting_start")

    def test_cross_task_skips_do_not_prematurely_release_group(self):
        w, engine = self.workflow({"mode": "default", "default_edges": ["left"]}), self.engine()
        w.node("join").start_condition = {"scope": "group", "group_by": "/batch"}
        left = Task.create(w, {"batch": "same"})
        engine.run_execution(w, left, left.executions[0])
        self.finish(engine, w)
        self.assertFalse(any(e.joined_from for e in left.executions))
        w.nodes[0].routing["default_edges"] = ["right"]
        right = Task.create(w, {"batch": "same"})
        engine.run_execution(w, right, right.executions[0])
        self.finish(engine, w)
        joined = [t for t in engine.task_pool.tasks.values() if t.metadata.get("joined_task_ids")]
        self.assertEqual(len(joined), 1)
        self.assertEqual(set(joined[0].metadata["joined_task_ids"]), {left.id, right.id})

    def test_group_opt_in_accepts_skip_without_forging_output(self):
        w, engine = self.workflow({"mode": "default", "default_edges": ["left"]}), self.engine()
        w.node("join").start_condition = {
            "scope": "group",
            "group_by": "/batch",
            "accept_skipped": True,
        }
        task = Task.create(w, {"batch": "same"})
        engine.run_execution(w, task, task.executions[0])
        self.finish(engine, w)
        joined = next(e for e in task.executions if e.joined_from)
        self.assertTrue(any(r["resolution"] == "skipped" for r in joined.joined_from))
        self.assertEqual(task.data, {"batch": "same"})

    def test_expired_join_cannot_release_before_human_review(self):
        w, engine = self.workflow(), self.engine()
        w.node("join").timeout_seconds = 10
        task = Task.create(w)
        engine.run_execution(w, task, task.executions[0])
        engine.run_execution(w, task, next(e for e in task.executions if e.node_id == "left"))
        arrival = next(e for e in task.executions if e.node_id == "join")
        arrival.timing["deadline"] = 0
        engine.run_execution(w, task, next(e for e in task.executions if e.node_id == "right"))
        self.assertFalse(any(e.joined_from for e in task.executions))
        resolve_human_check(task, pending_human_checks(task)[0]["id"], True, reviewer="human")
        ready = engine.refresh_start_conditions(w)
        self.assertEqual(len(ready), 1)
        self.assertIn("deadline", ready[0][1].timing)

    def test_post_review_then_branch_review_each_happen_once(self):
        w, engine = self.workflow({"mode": "human"}), self.engine()
        w.nodes[0].post_run_human_check = True
        task = Task.create(w)
        engine.run_execution(w, task, task.executions[0])
        resolve_human_check(task, pending_human_checks(task)[0]["id"], True, reviewer="human")
        self.assertEqual(engine.run_execution(w, task, task.executions[0]).status, "waiting")
        resolve_branch_selection(
            task, pending_branch_selections(task)[0]["id"], ["left"], reviewer="human"
        )
        self.assertEqual(engine.run_execution(w, task, task.executions[0]).status, "completed")
        self.assertEqual(len(task.executions[0].human_checks), 1)

    def test_runtime_json_unknown_fields_fail(self):
        raw = Task.create(self.workflow()).to_dict()
        raw["active_memroy"] = []
        with self.assertRaisesRegex(ValueError, "active_memroy"):
            Task.from_dict(raw)
        del raw["active_memroy"]
        raw["executions"][0]["routing_sate"] = {}
        with self.assertRaisesRegex(ValueError, "routing_sate"):
            Task.from_dict(raw)

    def test_downstream_skip_validation_precedes_any_arrival(self):
        w, engine = self.workflow({"mode": "default", "default_edges": ["left"]}), self.engine()
        w.node("join").start_condition = {"scope": "group", "group_by": "/missing"}
        task = Task.create(w)
        result = engine.run_execution(w, task, task.executions[0])
        self.assertIn("Scheduling: KeyError", result.error)
        self.assertEqual(len(task.executions), 1)
        self.assertEqual(traversal_count(task, "left"), 0)

    def test_budget_and_active_memory_preserved_on_task_merge(self):
        w, engine = self.workflow(), self.engine()
        w.node("join").start_condition = {"scope": "group", "group_by": "/batch"}
        engine.register_memory_writer("worker", lambda messages: "Recorded branch outcome")
        sources = []
        for name in ("left", "right"):
            task = Task.create(w, {"batch": "same"})
            from workflow_core import TaskExecution

            task.executions = [TaskExecution.create(name)]
            sources.append(task)
            engine.run_execution(w, task, task.executions[0])
        owner = next(
            t for t in engine.task_pool.tasks.values() if t.metadata.get("joined_task_ids")
        )
        self.assertEqual(
            {n["task_id"] for n in owner.active_memory["current"]}, {t.id for t in sources}
        )
        self.assertEqual(traversal_count(owner, "left-join"), 1)
        self.assertEqual(traversal_count(owner, "right-join"), 1)


if __name__ == "__main__":
    unittest.main()
