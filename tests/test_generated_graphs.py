import json
import tempfile
import unittest
from pathlib import Path

from workflow_core import (
    AllowanceSet,
    DirectedGraph,
    LocalToolModule,
    NodeDefinition,
    NodeResult,
    ReActAction,
    SkillGraphInterface,
    Task,
    ToolDefinition,
    WorkflowDefinition,
    WorkflowEngine,
    pending_human_checks,
    register_skill,
    resolve_human_check,
)


class GeneratedGraphTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.settings = self.root / "settings.json"
        self.config = {
            "react_rounds": 5,
            "token_limit": 8192,
            "graph_max_depth": 2,
            "graph_creation_human_check": True,
            "graph_save_human_check": True,
        }
        self.settings.write_text(json.dumps(self.config))
        self.module = LocalToolModule()
        self.engine = WorkflowEngine(tools=self.module, memory_settings_path=self.settings)
        self.workflow = WorkflowDefinition(
            "w",
            "W",
            [
                NodeDefinition(
                    "parent",
                    "parent",
                    allow_create_graph=True,
                    react={"on_action_error": "fail"},
                )
            ],
            [],
            ["parent"],
        )
        self.task = Task.create(self.workflow)
        self.calls = []
        self.engine.register(
            "worker",
            lambda ctx: (
                self.calls.append(ctx.task.metadata["_graph_depth"]),
                NodeResult(values={"value": 42}),
            )[1],
            generated_graphs=True,
        )
        self.action = "tool:graph.create"
        self.args = {
            "nodes": [{"id": "a", "type": "worker", "outputs": {"value": "/answer"}}],
            "edges": [],
            "start": ["a"],
            "explanation": "Run one worker to compute the result.",
        }
        self.engine.register_react(
            "parent",
            lambda ctx, history: NodeResult() if history else ReActAction(self.action, self.args),
        )

    def run_parent(self):
        return self.engine.run_execution(self.workflow, self.task, self.task.executions[0])

    def approve(self):
        (request,) = pending_human_checks(self.task)
        resolve_human_check(self.task, request["id"], True, reviewer="reviewer")
        return request

    def test_review_then_execute_and_roundtrip(self):
        self.assertEqual(self.run_parent().status, "waiting")
        self.assertEqual(self.calls, [])
        request = self.approve()
        self.assertEqual(request["details"]["explanation"], self.args["explanation"])
        checkpoint = self.root / "task.json"
        self.task.save(checkpoint)
        self.task = Task.load(checkpoint)
        self.assertEqual(self.run_parent().status, "completed")
        self.assertEqual(self.calls, [1])
        (record,) = self.task.executions[0].memory_state["graphs"].values()
        self.assertEqual(record["result"]["result"]["answer"], 42)
        graph = DirectedGraph.from_dict(record["graph"])
        self.assertEqual((graph.depth, graph.node_count, graph.edge_count), (1, 1, 0))
        path = self.root / "graph.json"
        graph.save(path)
        self.assertEqual(DirectedGraph.load(path).to_dict(), graph.to_dict())
        self.assertIn("graph.completed", [e["type"] for e in self.task.events])
        child_events = [e for e in self.task.events if e["task_id"] == record["task"]["id"]]
        self.assertTrue(child_events)

    def test_run_observers_see_child_graph_executions(self):
        seen = []
        self.engine.add_run_observer(lambda task, execution: seen.append((task, execution.node_id)))

        def broken(task, execution):
            raise RuntimeError("display down")

        self.engine.add_run_observer(broken)
        self.assertEqual(self.run_parent().status, "waiting")
        self.approve()
        self.assertEqual(self.run_parent().status, "completed")
        self.assertEqual((seen[0][0].id, seen[0][1]), (self.task.id, "parent"))
        children = [(task, node) for task, node in seen if task.id != self.task.id]
        self.assertEqual([node for _, node in children], ["a"])
        child = children[0][0]
        self.assertEqual(child.metadata["parent_task_id"], self.task.id)
        self.assertEqual(child.status, "completed")
        failures = [e for e in self.task.events if e["type"] == "observer.failed"]
        self.assertTrue(failures)
        self.assertEqual(failures[0]["details"]["error"], "display down")
        with self.assertRaises(TypeError):
            self.engine.add_run_observer("not callable")

    def test_child_approval_bridged_and_completed_nodes_not_replayed(self):
        self.module.register(
            ToolDefinition("reviewed", "low", human_check=True), lambda *args: "ok"
        )
        self.engine.register_react(
            "reviewed",
            lambda ctx, history: NodeResult() if history else ReActAction("tool:reviewed"),
            generated_graphs=True,
        )
        self.args["nodes"].append({"id": "b", "type": "reviewed"})
        self.args["edges"] = [{"from": "a", "to": "b"}]
        self.assertEqual(self.run_parent().status, "waiting")
        self.approve()
        self.assertEqual(self.run_parent().status, "waiting")
        self.assertEqual(self.calls, [1])
        self.assertEqual(set(self.engine.task_pool.tasks), {self.task.id})
        request = self.approve()
        self.assertEqual(request["details"]["kind"], "child_graph_review")
        checkpoint = self.root / "task.json"
        self.task.save(checkpoint)
        self.task = Task.load(checkpoint)
        self.assertEqual(self.run_parent().status, "completed")
        self.assertEqual(self.calls, [1])

    def test_flag_depth_and_permission_subset(self):
        self.workflow.nodes[0].allow_create_graph = False
        result = self.run_parent()
        self.assertEqual(result.status, "failed")
        self.assertEqual(self.calls, [])
        self.workflow.nodes[0].allow_create_graph = True
        self.task = Task.create(self.workflow)
        self.task.metadata["_graph_depth"] = 2
        self.assertIn("depth limit", self.run_parent().error)
        self.task = Task.create(self.workflow)
        self.engine.allowances.register(
            AllowanceSet("parent", ["tool:graph.create"], max_tool_risk="high")
        )
        self.engine.allowances.register(
            AllowanceSet("wide", ["tool:graph.create", "tool:extra"], max_tool_risk="high")
        )
        self.workflow.nodes[0].allowance = "parent"
        self.args["nodes"][0]["allowance"] = "wide"
        self.assertIn("exceeds parent", self.run_parent().error)

    def test_actual_nested_creation_obeys_live_depth_limit(self):
        self.config["graph_creation_human_check"] = False
        self.settings.write_text(json.dumps(self.config))
        leaf = {
            "nodes": [{"id": "leaf", "type": "worker", "outputs": {"value": "/answer"}}],
            "edges": [],
            "start": ["leaf"],
            "explanation": "Compute inside the nested graph.",
        }
        self.engine.register_react(
            "nest",
            lambda ctx, history: (
                NodeResult() if history else ReActAction("tool:graph.create", leaf)
            ),
            generated_graphs=True,
        )
        self.args["nodes"] = [
            {
                "id": "a",
                "type": "nest",
                "allow_create_graph": True,
                "react": {"on_action_error": "fail"},
            }
        ]
        self.assertEqual(self.run_parent().status, "completed")
        self.assertEqual(self.calls, [2])
        self.task = Task.create(self.workflow)
        self.config["graph_max_depth"] = 1
        self.settings.write_text(json.dumps(self.config))
        self.assertEqual(self.run_parent().status, "failed")
        self.assertEqual(self.calls, [2])

    def test_skill_persistence_has_separate_review(self):
        package = self.root / "review"
        package.mkdir()
        (package / "SKILL.md").write_text(
            "---\nname: review\ndescription: Review work\n---\nCompute a result."
        )
        register_skill(self.module, package)
        ctx = self.engine.build_context(self.workflow, self.task, self.task.executions[0])
        invoke = ctx.use_tool("skill.review.invoke")["invoke_id"]
        self.args.update(
            skill_invoke_id=invoke,
            save_to_skill=True,
            usage_scenario="""# Usage scenario
For tasks that need a single computed result.
The input is the current task data.""",
        )
        self.assertEqual(self.run_parent().status, "waiting")
        self.approve()
        # The graph runs first; saving is reviewed only after it succeeded.
        self.assertEqual(self.run_parent().status, "waiting")
        self.assertEqual(self.calls, [1])
        self.assertFalse((package / "graphs").exists())
        request = self.approve()
        self.assertEqual(request["details"]["kind"], "skill_graph_save")
        self.assertEqual(request["details"]["usage_scenario"], self.args["usage_scenario"])
        self.assertEqual(self.run_parent().status, "completed")
        self.assertEqual(self.calls, [1])
        (artifact,) = [
            p for p in (package / "graphs").glob("*.json") if not p.name.endswith(".meta.json")
        ]
        self.assertEqual(json.loads(artifact.read_text())["graph"]["depth"], 1)
        self.assertEqual(
            artifact.with_suffix(".md").read_text(encoding="utf8"), self.args["usage_scenario"]
        )
        self.assertIn("skill.graph.saved", [e["type"] for e in self.task.events])

    # -- graphs saved in skills: listing and reuse ----------------------------

    def skill(self):
        package = self.root / "review"
        package.mkdir(exist_ok=True)
        (package / "SKILL.md").write_text("---\nname: review\ndescription: Review work\n---\nWork.")
        register_skill(self.module, package)
        return package

    def open_skill(self):
        ctx = self.engine.build_context(self.workflow, self.task, self.task.executions[0])
        return ctx.use_tool("skill.review.invoke")

    def save_graph(self, scenario="# Single result\nFor one computed value."):
        self.args.update(skill_invoke_id=self.open_skill()["invoke_id"], save_to_skill=True)
        self.args["usage_scenario"] = scenario
        self.run_parent()
        self.approve()
        self.run_parent()
        self.approve()
        self.assertEqual(self.run_parent().status, "completed")
        (record,) = self.task.executions[0].memory_state["graphs"].values()
        return record["id"]

    def test_invoke_lists_saved_graphs_and_flags_tampered_ones(self):
        package = self.skill()
        graph_id = self.save_graph()
        self.task = Task.create(self.workflow)
        (listed,) = self.open_skill()["graphs"]
        self.assertEqual(listed["id"], graph_id)
        self.assertEqual(listed["title"], "Single result")
        self.assertEqual(listed["nodes"], [{"id": "a", "name": ""}])
        self.assertEqual(listed["explanation"], self.args["explanation"])

        (package / "graphs" / f"{graph_id}.md").write_text("# Changed")
        self.task = Task.create(self.workflow)
        (listed,) = self.open_skill()["graphs"]
        self.assertIn("does not match", listed["invalid"])
        self.assertNotIn("nodes", listed)

    def run_saved(self, graph_id, invoke=None):
        self.action = "tool:skill.review.run_graph"
        self.args = {"invoke_id": invoke or self.open_skill()["invoke_id"], "graph_id": graph_id}
        return self.run_parent()

    def test_saved_graph_runs_through_the_skill_hook_after_review(self):
        self.skill()
        graph_id = self.save_graph()
        self.calls.clear()
        self.task = Task.create(self.workflow)
        # The skill's run_graph action is the permission; graph creation is not needed.
        self.workflow.nodes[0].allow_create_graph = False
        self.assertEqual(self.run_saved(graph_id).status, "waiting")
        request = self.approve()
        self.assertEqual(request["details"]["skill_graph"], graph_id)
        self.assertEqual(request["details"]["graph"]["nodes"][0]["id"], "a")
        self.assertEqual(self.run_parent().status, "completed")
        self.assertEqual(self.calls, [1])
        (record,) = self.task.executions[0].memory_state["graphs"].values()
        self.assertEqual(record["skill_graph"], graph_id)
        self.assertNotEqual(record["id"], graph_id)
        types = [e["type"] for e in self.task.events]
        self.assertIn("skill.run_graph.completed", types)
        self.assertNotIn("skill.run_graph.failed", types)  # the review pause is not a failure

    def test_saved_graph_requests_are_checked(self):
        self.skill()
        graph_id = self.save_graph()
        for graph, invoke, expected in [
            ("../escape", None, "skill graph id"),
            ("missing", None, "FileNotFoundError"),
            (graph_id, "invoke_closed", "Unknown, closed, or foreign"),
        ]:
            with self.subTest(graph=graph):
                self.task = Task.create(self.workflow)
                result = self.run_saved(graph, invoke)
                self.assertEqual(result.status, "failed")
                self.assertIn(expected, result.error)
        self.task = Task.create(self.workflow)
        self.config["graph_max_depth"] = 0
        self.settings.write_text(json.dumps(self.config))
        self.assertIn("depth", self.run_saved(graph_id).error)

    def test_child_allowances_still_bound_a_saved_graph(self):
        self.skill()
        graph_id = self.save_graph()
        self.engine.allowances.register(
            AllowanceSet(
                "parent",
                ["tool:skill.review.invoke", "tool:skill.review.run_graph"],
                max_tool_risk="high",
            )
        )
        self.engine.allowances.register(AllowanceSet("wide", ["tool:anything"]))
        path = self.root / "review" / "graphs" / f"{graph_id}.json"
        document = json.loads(path.read_text())
        document["graph"]["nodes"][0]["allowance"] = "wide"
        path.write_text(json.dumps(document))
        self.task = Task.create(self.workflow)
        self.workflow.nodes[0].allowance = "parent"
        result = self.run_saved(graph_id)
        self.assertEqual(result.status, "failed")
        self.assertIn("exceeds", result.error)

    def test_hosts_can_replace_or_remove_the_graph_runner(self):
        package = self.root / "custom"
        package.mkdir()
        (package / "SKILL.md").write_text("---\nname: custom\ndescription: Custom\n---\nWork.")
        seen = []

        class Runner(SkillGraphInterface):
            def execute_graph(self, context, invoke_id, graph_id):
                seen.append((invoke_id, graph_id))
                return {"ran": graph_id}

        register_skill(self.module, package, graph_runner=Runner())
        ctx = self.engine.build_context(self.workflow, self.task, self.task.executions[0])
        invoke = ctx.use_tool("skill.custom.invoke")["invoke_id"]
        self.assertEqual(
            ctx.use_tool("skill.custom.run_graph", {"invoke_id": invoke, "graph_id": "g1"}),
            {"ran": "g1"},
        )
        self.assertEqual(seen, [(invoke, "g1")])

        bare = self.root / "bare"
        bare.mkdir()
        (bare / "SKILL.md").write_text("---\nname: bare\ndescription: Bare\n---\nWork.")
        register_skill(self.module, bare, graph_runner=None)
        names = {tool.name for tool in self.module.list_tools()}
        self.assertIn("skill.custom.run_graph", names)
        self.assertNotIn("skill.bare.run_graph", names)
        with self.assertRaises(TypeError):
            register_skill(self.module, bare, graph_runner=object())

    def test_declining_the_save_keeps_the_result(self):
        package = self.skill()
        self.args.update(
            skill_invoke_id=self.open_skill()["invoke_id"],
            save_to_skill=True,
            usage_scenario="# Usage\nOne value.",
        )
        self.run_parent()
        self.approve()
        self.assertEqual(self.run_parent().status, "waiting")
        (request,) = pending_human_checks(self.task)
        resolve_human_check(self.task, request["id"], False, reviewer="reviewer")
        self.assertEqual(self.task.executions[0].status, "ready")
        self.assertEqual(self.run_parent().status, "completed")
        self.assertEqual(self.calls, [1])
        self.assertFalse((package / "graphs").exists())
        (record,) = self.task.executions[0].memory_state["graphs"].values()
        self.assertEqual(
            (record["save_outcome"], record["result"]["saved_path"]), ("declined", None)
        )
        self.assertIn("skill.graph.save_declined", [e["type"] for e in self.task.events])

    def test_failed_graphs_are_not_offered_for_saving(self):
        self.skill()
        self.engine.register(
            "worker", lambda ctx: NodeResult(status="failed", error="boom"), generated_graphs=True
        )
        self.args.update(
            skill_invoke_id=self.open_skill()["invoke_id"],
            save_to_skill=True,
            usage_scenario="# Usage\nOne value.",
        )
        self.run_parent()
        self.approve()
        result = self.run_parent()
        self.assertEqual(result.status, "failed")
        self.assertEqual(pending_human_checks(self.task), [])

    # -- environment and run history of saved graphs ----------------------------

    def meta(self, graph_id):
        return json.loads((self.root / "review" / "graphs" / f"{graph_id}.meta.json").read_text())

    def listed(self, graph_id):
        self.task = Task.create(self.workflow)
        return next(g for g in self.open_skill()["graphs"] if g["id"] == graph_id)

    def test_saving_records_the_environment_and_the_first_run(self):
        self.skill()
        self.engine.host_environment["model"] = "provider/model-a"
        graph_id = self.save_graph()
        meta = self.meta(graph_id)
        self.assertEqual(meta["runs"], {"total": 1, "completed": 1, "failed": 0})
        self.assertEqual(meta["last_run"]["outcome"], "completed")
        environment = meta["saved"]["environment"]
        self.assertEqual(environment["node_types"], {"worker": True})
        self.assertEqual(environment["host"], {"model": "provider/model-a"})
        listed = self.listed(graph_id)  # the metadata file itself is not listed as a graph
        self.assertEqual(len(self.open_skill()["graphs"]), 1)
        self.assertEqual(listed["environment"]["status"], "same")
        self.assertEqual(listed["environment"]["compared_with"], "last successful run")
        self.assertEqual(listed["history"]["runs"]["completed"], 1)
        self.assertIn("safe to reuse", listed["hint"])

    def test_changes_in_the_environment_are_reported(self):
        self.skill()
        self.engine.allowances.register(AllowanceSet("child", ["tool:echo"]))
        self.module.register(ToolDefinition("echo", "low", description="v1"), lambda *a: "ok")
        self.args["nodes"][0]["allowance"] = "child"
        self.engine.host_environment["model"] = "provider/model-a"
        graph_id = self.save_graph()

        self.engine.host_environment["model"] = "provider/model-b"
        listed = self.listed(graph_id)
        self.assertEqual(listed["environment"]["status"], "changed")
        self.assertIn("model: provider/model-a → provider/model-b", listed["hint"])

        tools = LocalToolModule()
        tools.register(ToolDefinition("echo", "low", description="v2"), lambda *a: "ok")
        register_skill(tools, self.root / "review")
        self.engine.attach_tools(tools)
        self.assertIn("tool echo changed", self.listed(graph_id)["hint"])

        del self.engine.handlers["worker"]
        listed = self.listed(graph_id)
        self.assertEqual(listed["environment"]["status"], "incompatible")
        self.assertTrue(listed["hint"].startswith("Do not reuse this graph"))
        self.assertIn("node type worker is no longer registered", listed["hint"])

    def test_each_run_of_a_saved_graph_is_recorded_once(self):
        self.skill()
        graph_id = self.save_graph()
        self.task = Task.create(self.workflow)
        self.assertEqual(self.run_saved(graph_id).status, "waiting")  # review pause
        self.approve()
        self.assertEqual(self.run_parent().status, "completed")
        meta = self.meta(graph_id)
        self.assertEqual(meta["runs"], {"total": 2, "completed": 2, "failed": 0})
        self.assertIsNotNone(meta["last_run"]["child_task_id"])
        self.assertEqual(meta["last_run"]["steps"], 1)

        self.engine.register(
            "worker", lambda ctx: NodeResult(status="failed", error="boom"), generated_graphs=True
        )
        self.task = Task.create(self.workflow)
        self.run_saved(graph_id)
        self.approve()
        self.assertEqual(self.run_parent().status, "failed")
        meta = self.meta(graph_id)
        self.assertEqual(meta["runs"], {"total": 3, "completed": 2, "failed": 1})
        self.assertIn("boom", meta["last_run"]["error"])
        listed = self.listed(graph_id)
        self.assertEqual(listed["environment"]["status"], "same")
        self.assertIn("last run failed", listed["hint"])

    def test_metadata_for_another_version_of_the_graph_is_ignored(self):
        self.skill()
        graph_id = self.save_graph()
        path = self.root / "review" / "graphs" / f"{graph_id}.json"
        document = json.loads(path.read_text())
        document["explanation"] = "Edited by hand."
        path.write_text(json.dumps(document))
        listed = self.listed(graph_id)
        self.assertEqual(listed["environment"]["status"], "unknown")
        self.assertEqual(listed["history"]["runs"]["total"], 0)

    def test_child_argument_policy_is_bounded_by_creator(self):
        self.config["graph_creation_human_check"] = False
        self.settings.write_text(json.dumps(self.config))
        self.engine.allowances.register(
            AllowanceSet("parent", ["tool:graph.create", "tool:echo"], config={"path": "public"})
        )
        self.engine.allowances.register(AllowanceSet("child", ["tool:echo"], config={}))
        self.workflow.nodes[0].allowance = "parent"
        self.args["nodes"][0]["allowance"] = "child"
        self.module.register(
            ToolDefinition("echo", "low"), lambda *args: self.calls.append("executed")
        )
        self.engine.allowance_check = lambda ctx, policy, action, args: (
            action != "tool:echo" or not policy.config or args.get("path") == policy.config["path"]
        )
        self.engine.register_react(
            "worker", lambda ctx, history: ReActAction("tool:echo", {"path": "private"})
        )
        self.assertEqual(self.run_parent().status, "failed")
        self.assertEqual(self.calls, [])

    def test_parent_allowance_expansion_cannot_expand_existing_child(self):
        self.config["graph_creation_human_check"] = False
        self.settings.write_text(json.dumps(self.config))
        self.engine.allowances.register(AllowanceSet("parent", ["tool:graph.create"]))
        self.workflow.nodes[0].allowance = "parent"
        self.module.register(
            ToolDefinition("echo", "low"), lambda *args: self.calls.append("executed")
        )

        def child_handler(ctx):
            self.engine.allowances.register(
                AllowanceSet("parent", ["tool:graph.create", "tool:echo"])
            )
            self.assertNotIn("echo", [t["name"] for t in ctx.prompt_context["available_tools"]])
            ctx.use_tool("echo")
            return NodeResult()

        self.engine.register("worker", child_handler)
        self.assertEqual(self.run_parent().status, "failed")
        self.assertEqual(self.calls, [])

    def test_skill_graph_requires_scenario_before_review(self):
        package = self.root / "review"
        package.mkdir()
        (package / "SKILL.md").write_text("---\nname: review\ndescription: Review work\n---\nWork.")
        register_skill(self.module, package)
        ctx = self.engine.build_context(self.workflow, self.task, self.task.executions[0])
        invoke = ctx.use_tool("skill.review.invoke")["invoke_id"]
        self.args.update(skill_invoke_id=invoke, save_to_skill=True)
        result = self.run_parent()
        self.assertEqual(result.status, "failed")
        self.assertIn("usage_scenario", result.error)
        self.assertEqual(pending_human_checks(self.task), [])
        self.assertFalse((package / "graphs").exists())
