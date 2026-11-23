import unittest

from workflow_core import (
    AllowanceSet,
    HumanCheckRequired,
    LocalToolModule,
    MCPToolModule,
    NodeDefinition,
    NodeResult,
    Task,
    ToolDefinition,
    WorkflowDefinition,
    WorkflowEngine,
    resolve_human_check,
)


class Server(MCPToolModule, LocalToolModule):
    connect = lambda self, config: None
    close = lambda self: None
    get_tool = LocalToolModule.get_tool
    list_tools = LocalToolModule.list_tools
    execute = LocalToolModule.execute


class MCPAllowanceTests(unittest.TestCase):
    def setUp(self):
        node = NodeDefinition.from_dict(
            {
                "id": "n",
                "type": "agent",
                "allowance": "read",
                "requires": {"mcp_servers": ["docs", "tickets"]},
            }
        )
        self.workflow = WorkflowDefinition("w", "w", [node], [], ["n"])
        self.task = Task.create(self.workflow)
        local = LocalToolModule()
        local.register(ToolDefinition("search", "low"), lambda *args: "local")
        self.engine = WorkflowEngine(tools=local)
        for name in ["docs", "tickets", "other"]:
            server = Server()
            server.register(ToolDefinition("search", "low"), lambda *args, name=name: name)
            server.register(ToolDefinition("write", "high"), lambda *args: "written")
            self.engine.attach_mcp_server(name, server)
        self.engine.allowances.register(
            AllowanceSet(
                "read",
                [
                    "tool:search",
                    "tool:mcp:docs:search",
                    "tool:mcp:docs:write",
                    "tool:mcp:other:search",
                ],
                max_tool_risk="low",
            )
        )
        self.ctx = self.engine.build_context(self.workflow, self.task, self.task.executions[0])

    def test_catalog_and_execution_share_exact_permissions(self):
        names = {t["action"] for t in self.ctx.prompt_context["available_tools"]}
        self.assertEqual(names, {"tool:search", "tool:mcp:docs:search"})
        self.assertEqual(self.ctx.use_tool("search"), "local")
        self.assertEqual(self.ctx.use_tool("mcp:docs:search"), "docs")
        for name in ["mcp:tickets:search", "mcp:docs:write", "mcp:other:search"]:
            with self.assertRaises(PermissionError):
                self.ctx.use_tool(name)

    def test_required_session_and_scoped_approvals(self):
        self.ctx.node.allowance = None
        for name in ["docs", "tickets"]:
            self.engine.mcp_servers[name].set_human_check("search")
        with self.assertRaises(HumanCheckRequired) as first:
            self.ctx.use_tool("mcp:docs:search")
        self.task.executions[0].status = "waiting"
        resolve_human_check(self.task, first.exception.request_id, True, reviewer="tester")
        with self.assertRaises(HumanCheckRequired):
            self.ctx.use_tool("mcp:tickets:search")
        self.assertEqual(self.ctx.use_tool("mcp:docs:search"), "docs")
        self.engine.attach_mcp_server("docs", None)
        self.engine.register("agent", lambda ctx: NodeResult())
        self.task.executions[0].status = "ready"
        result = self.engine.run_execution(self.workflow, self.task, self.task.executions[0])
        self.assertEqual(result.status, "failed")
        self.assertIn("docs", result.error)
