import copy
import hashlib
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workflow_core import (
    AllowanceSet,
    BuiltinPrivateToolModule,
    HumanCheckRequired,
    NodeDefinition,
    Task,
    WorkflowDefinition,
    WorkflowEngine,
    resolve_human_check,
)


class BasicAndTemplateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tools = BuiltinPrivateToolModule(
            self.root, http_allowed_origins=["https://api.example.com"]
        )
        self.engine = WorkflowEngine(tools=self.tools)
        self.workflow = WorkflowDefinition("w", "W", [NodeDefinition("n", "work")], [], ["n"])
        self.task = Task.create(self.workflow)
        self.ctx = self.engine.build_context(self.workflow, self.task, self.task.executions[0])

    def call(self, name, **args):
        return self.ctx.use_tool(name, args)

    def definition(self):
        return {
            "name": "decl.fetch",
            "description": "Fetch and parse JSON",
            "parameters": {"accept": "string"},
            "steps": [
                {
                    "id": "fetch",
                    "tool": "http.request",
                    "arguments": {
                        "url": "https://api.example.com/data",
                        "method": "GET",
                        "headers": {"Accept": {"$input": "accept"}},
                    },
                },
                {
                    "id": "parse",
                    "tool": "json.parse",
                    "arguments": {"text": {"$step": "fetch", "path": "/body"}},
                },
            ],
        }

    def test_twenty_tools_and_data_operations(self):
        self.assertEqual(len(self.tools.list_tools()), 20)
        cases = [
            ("text.search", {"text": "abcabc", "query": "bc"}, "positions", [1, 4]),
            ("text.replace", {"text": "abc", "old": "b", "new": "x"}, "text", "axc"),
            ("text.split", {"text": "a,b", "separator": ","}, "items", ["a", "b"]),
            ("text.join", {"items": ["a", "b"], "separator": ","}, "text", "a,b"),
            ("json.parse", {"text": '{"x": 2}'}, "data", {"x": 2}),
            ("json.stringify", {"data": {"x": 2}}, "text", '{"x": 2}'),
            ("json.get", {"data": {"x": [2]}, "pointer": "/x/0"}, "value", 2),
            ("csv.parse", {"text": 'a,"b,c"\n'}, "rows", [["a", "b,c"]]),
            ("csv.format", {"rows": [["a", "b,c"]]}, "text", 'a,"b,c"\r\n'),
            ("hash.sha256", {"text": "hello"}, "sha256", hashlib.sha256(b"hello").hexdigest()),
            (
                "url.build",
                {"url": "https://api.example.com", "query": {"q": "a b"}},
                "url",
                "https://api.example.com?q=a+b",
            ),
        ]
        for name, args, key, expected in cases:
            self.assertEqual(self.call(name, **args)[key], expected)
        with self.assertRaises(ValueError):
            self.call("json.parse", text="NaN")

    def test_template_persistence_and_inner_authorization(self):
        definition = self.definition()
        self.call("tools.define", definition=definition)
        self.assertEqual(self.tools.get_tool("decl.fetch").risk_level, "high")
        self.tools = BuiltinPrivateToolModule(
            self.root, http_allowed_origins=["https://api.example.com"]
        )
        self.engine.attach_tools(self.tools)
        self.workflow.nodes[0].allowance = "allowed"
        self.engine.allowances.register(AllowanceSet("allowed", ["tool:decl.fetch"]))
        with patch.object(self.tools._http, "request") as request:
            with self.assertRaises(PermissionError):
                self.call("decl.fetch", accept="application/json")
            request.assert_not_called()
        self.assertNotIn("declarative_runs", self.ctx.execution.memory_state)
        self.engine.allowances.register(
            AllowanceSet("allowed", ["tool:decl.fetch", "tool:http.request", "tool:json.parse"])
        )
        with patch.object(
            self.tools._http, "request", return_value={"body": '{"ok": true}'}
        ) as request:
            result = self.call("decl.fetch", accept="application/json")
        self.assertEqual(result["result"], {"data": {"ok": True}})
        self.assertEqual(request.call_args.args[1]["headers"], {"Accept": "application/json"})
        self.assertTrue(any(e["type"] == "tool.added" for e in self.task.events))

    def test_dynamic_checks_and_invalid_edit_are_atomic(self):
        definition = self.definition()
        self.call("tools.define", definition=definition)
        before = self.tools.get_tool("decl.fetch")
        bad = copy.deepcopy(definition)
        bad["steps"][0]["arguments"]["url"] = "https://evil.example/data"
        with self.assertRaises(PermissionError):
            self.call("tools.define", definition=bad, overwrite=True)
        self.assertEqual(before, self.tools.get_tool("decl.fetch"))
        with self.assertRaises(ValueError):
            self.call("decl.fetch", accept="application/json\r\nHost: evil")
        with self.assertRaises(PermissionError):
            self.call(
                "fs.write_text", path=".dgs-tools/definitions.json", content="{}", overwrite=True
            )
        bad = copy.deepcopy(definition)
        bad["steps"][0]["arguments"]["headers"] = {"$step": "parse"}
        with self.assertRaises(ValueError):
            self.call("tools.define", definition=bad, overwrite=True)

    def test_human_resume_preserves_completed_steps(self):
        (self.root / "delete.txt").write_text("delete")
        definition = {
            "name": "decl.cleanup",
            "description": "Write then delete",
            "parameters": {},
            "steps": [
                {
                    "id": "write",
                    "tool": "fs.write_text",
                    "arguments": {"path": "keep.txt", "content": "kept"},
                },
                {"id": "delete", "tool": "fs.delete_file", "arguments": {"path": "delete.txt"}},
            ],
        }
        self.call("tools.define", definition=definition)
        with self.assertWarns(UserWarning), self.assertRaises(HumanCheckRequired) as waiting:
            self.call("decl.cleanup")
        self.assertEqual((self.root / "keep.txt").read_text(), "kept")
        # A replay of step one would fail with FileExistsError.
        self.task.executions[0].status = "waiting"
        resolve_human_check(self.task, waiting.exception.request_id, True, reviewer="test")
        self.task = Task.from_dict(self.task.to_dict())
        self.ctx = self.engine.build_context(self.workflow, self.task, self.task.executions[0])
        self.call("decl.cleanup")
        self.assertFalse((self.root / "delete.txt").exists())

    def test_post_needs_approval_even_for_direct_backend_execution(self):
        args = {"url": "https://api.example.com/data", "method": "POST", "body": "{}"}
        with self.assertRaises(HumanCheckRequired):
            self.call("http.request", **args)
        with self.assertRaises(PermissionError):
            self.tools.execute(self.ctx, self.tools.get_tool("http.request"), args)

    def test_http_private_dns_and_header_injection(self):
        addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]
        with (
            patch("workflow_http_tools.socket.getaddrinfo", return_value=addresses),
            patch("workflow_http_tools.socket.socket") as connect,
        ):
            with self.assertRaises(PermissionError):
                self.call("http.request", url="https://api.example.com/data", method="GET")
            connect.assert_not_called()
        for headers in [{"Host": "evil"}, {"X-Data": "x\r\ny"}, {"TE": "chunked"}]:
            with self.assertRaises(ValueError):
                self.call(
                    "http.request",
                    url="https://api.example.com/data",
                    method="GET",
                    headers=headers,
                )
        with self.assertRaises(PermissionError):
            self.call("http.request", url="https://example.org", method="GET")

    def test_http_pins_address_and_does_not_follow_redirect(self):
        addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        with (
            patch("workflow_http_tools.socket.getaddrinfo", return_value=addresses) as dns,
            patch("workflow_http_tools.socket.socket") as socket_factory,
            patch("workflow_http_tools.ssl.create_default_context") as tls,
            patch("workflow_http_tools.http.client.HTTPConnection") as connection,
        ):
            response = connection.return_value.getresponse.return_value
            response.status = 302
            response.getheaders.return_value = [("Location", "https://127.0.0.1")]
            response.read.return_value = b"redirect"
            result = self.call(
                "http.request", url="https://api.example.com/data", method="GET", query={"q": "a b"}
            )
            self.assertEqual(result["status"], 302)
            self.assertEqual(connection.return_value.request.call_args.args[1], "/data?q=a+b")
            socket_factory.return_value.connect.assert_called_once_with(("93.184.216.34", 443))
            self.assertEqual(
                tls.return_value.wrap_socket.call_args.kwargs["server_hostname"], "api.example.com"
            )
            self.assertEqual(dns.call_count, 1)
            self.assertTrue(any(e["type"] == "http.response" for e in self.task.events))

    def test_static_schema_ast_and_type_rejections(self):
        definition = self.definition()
        invalid = []
        unknown = copy.deepcopy(definition)
        unknown["execute_python"] = "print(1)"
        invalid.append(unknown)
        forward = copy.deepcopy(definition)
        forward["steps"][0]["arguments"]["headers"] = {"$step": "parse"}
        invalid.append(forward)
        scalar = copy.deepcopy(definition)
        scalar["steps"][1]["arguments"]["text"]["path"] = "/status"
        invalid.append(scalar)
        unknown_field = copy.deepcopy(definition)
        unknown_field["steps"][1]["arguments"]["text"]["path"] = "/missing"
        invalid.append(unknown_field)
        duplicate_header = copy.deepcopy(definition)
        duplicate_header["steps"][0]["arguments"]["headers"] = {"Accept": "a", "accept": "b"}
        invalid.append(duplicate_header)
        operator = copy.deepcopy(definition)
        operator["steps"][0]["arguments"]["headers"] = {"$eval": "anything"}
        invalid.append(operator)
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                self.call("tools.define", definition=item)
        self.assertFalse((self.root / ".dgs-tools").exists())

    def test_static_permission_subset_and_host_whitelist(self):
        self.workflow.nodes[0].allowance = "author"
        self.engine.allowances.register(
            AllowanceSet("author", ["tool:tools.define", "tool:http.request"])
        )
        with self.assertRaises(PermissionError):
            self.call("tools.define", definition=self.definition())
        self.assertFalse((self.root / ".dgs-tools").exists())
        self.assertIn("tool.template.rejected", [event["type"] for event in self.task.events])
        self.workflow.nodes[0].allowance = None
        self.tools.configure(
            {
                "workspace_root": str(self.root),
                "declarative_allowed_tools": ["json.parse"],
                "http_allowed_origins": ["https://api.example.com"],
            }
        )
        with self.assertRaises(ValueError):
            self.call("tools.define", definition=self.definition())

    def test_static_ast_depth_and_literal_path_checks(self):
        definition = self.definition()
        value = "text"
        for _ in range(35):
            value = {"x": value}
        definition["steps"][0]["arguments"]["headers"] = value
        with self.assertRaisesRegex(ValueError, "depth"):
            self.call("tools.define", definition=definition)
        definition = {
            "name": "decl.escape",
            "description": "invalid path",
            "parameters": {},
            "steps": [{"id": "read", "tool": "fs.read_text", "arguments": {"path": "../outside"}}],
        }
        with self.assertRaises(PermissionError):
            self.call("tools.define", definition=definition)
        self.assertFalse((self.root / ".dgs-tools").exists())
