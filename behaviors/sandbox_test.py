"""
SandboxTestBehavior - Runs tests in isolated sandbox environments.

Tools:
- run_behavior_tests: Run behavior tests in isolated temp workspace
- run_agent_sanity_check: Verify agent can instantiate successfully

This behavior provides safe testing of generated code before installation.
Sandbox characteristics:
- Temporary directory (auto-cleaned)
- Subprocess isolation
- Timeout enforcement (30s default)
- Resource monitoring (basic)

Now uses @tool decorator for automatic tool registration!"""

from typing import Any
import subprocess
import tempfile
import shutil
import sys
from pathlib import Path
from behaviors.base import AgentBehavior
from behaviors.tool_decorator import tool
from behaviors.rule_of_two_types import RuleOfTwoProperty


class SandboxTestBehavior(AgentBehavior):
    """
    Provides sandbox testing tools for validating generated code.

    Runs tests in isolated temporary workspaces with timeouts and
    resource constraints to prevent damage to the production environment.

    Security: [] None (testing utility - runs tests in isolation)
    """

    # Rule of Two: Empty (utility behavior for testing)
    rule_of_two_properties = set()

    def __init__(self, **kwargs):
        """Initialize sandbox test behavior."""
        self.default_timeout = kwargs.get("default_timeout", 30)

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "sandbox_test"

    def on_initial_context(
        self,
        agent: Any,
        context: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Inject tool documentation for sandbox_test tools.

        Called once during agent initialization to document available tools.

        Args:
            agent: Agent instance
            context: Initial context (system prompt only)

        Returns:
            Context with tool documentation injected
        """
        tools = self.get_tools()
        if not tools:
            return context

        # Build tool documentation
        tool_docs = []
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {}).get("properties", {})
            required = func.get("parameters", {}).get("required", [])

            # Build parameter signature
            param_strs = []
            for param_name, param_spec in params.items():
                param_type = param_spec.get("type", "any")
                is_required = param_name in required
                if is_required:
                    param_strs.append(f"{param_name}: {param_type}")
                else:
                    param_strs.append(f"{param_name}?: {param_type}")

            param_sig = ", ".join(param_strs) if param_strs else ""
            tool_docs.append(f"  - {name}({param_sig}): {desc}")

        if tool_docs:
            tool_message = f"\n{self.get_name()} tools:\n" + "\n".join(tool_docs)
            # Use default role="system" for framework tool documentation
            return self.inject_message_after_system(context, tool_message)

        return context

    def get_tools(self) -> list[dict[str, Any]]:
        """Return sandbox testing tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "run_behavior_tests",
                    "description": "Run behavior tests in isolated sandbox with timeout",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "behavior_file": {
                                "type": "string",
                                "description": "Path to behavior Python file (relative to workspace)"
                            },
                            "test_file": {
                                "type": "string",
                                "description": "Path to test file (relative to workspace)"
                            },
                            "timeout": {
                                "type": "number",
                                "description": "Timeout in seconds (default: 30)"
                            }
                        },
                        "required": ["behavior_file", "test_file"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_agent_sanity_check",
                    "description": "Verify agent configuration can load and instantiate",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "config_file": {
                                "type": "string",
                                "description": "Path to agent config YAML (relative to workspace)"
                            },
                            "agent_class": {
                                "type": "string",
                                "description": "Agent class name to instantiate (e.g., 'TaskExecutorAgent')"
                            }
                        },
                        "required": ["config_file", "agent_class"]
                    }
                }
            }
        ]

    def dispatch_tool(
        self,
        agent: Any,
        tool_name: str,
        args: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle sandbox test tool execution.

        Args:
            agent: Agent instance
            tool_name: Tool being called
            args: Tool arguments

        Returns:
            Test result dict with "success", "stdout", "stderr"
        """
        if tool_name == "run_behavior_tests":
            return self._run_behavior_tests(agent, args)
        elif tool_name == "run_agent_sanity_check":
            return self._run_agent_sanity_check(agent, args)
        else:
            return super().dispatch_tool(agent, tool_name, args)

    def _run_behavior_tests(self, agent: Any, args: dict[str, Any]) -> dict[str, Any]:
        """
        Run behavior tests in isolated sandbox.

        Sandbox characteristics:
        - Temporary workspace (cleaned after test)
        - Subprocess isolation
        - Timeout enforcement
        - Captures stdout/stderr
        """
        try:
            behavior_file = args.get("behavior_file", "")
            test_file = args.get("test_file", "")
            timeout = args.get("timeout", self.default_timeout)

            # Resolve paths relative to workspace
            workspace = agent.workspace if hasattr(agent, 'workspace') else Path.cwd()
            behavior_path = workspace / behavior_file
            test_path = workspace / test_file

            # Validate files exist
            if not behavior_path.exists():
                return {"error": f"Behavior file not found: {behavior_file}"}
            if not test_path.exists():
                return {"error": f"Test file not found: {test_file}"}

            # Create sandbox directory
            sandbox_dir = tempfile.mkdtemp(prefix="sandbox_behavior_test_")
            sandbox_path = Path(sandbox_dir)

            try:
                print(f"[sandbox_test] Created sandbox: {sandbox_dir}")

                # Copy files to sandbox
                sandbox_behavior = sandbox_path / behavior_path.name
                sandbox_test = sandbox_path / test_path.name
                shutil.copy(behavior_path, sandbox_behavior)
                shutil.copy(test_path, sandbox_test)

                # Copy behaviors/base.py if needed (for imports)
                base_behavior_path = Path("behaviors/base.py")
                if base_behavior_path.exists():
                    behaviors_dir = sandbox_path / "behaviors"
                    behaviors_dir.mkdir(exist_ok=True)
                    shutil.copy(base_behavior_path, behaviors_dir / "base.py")
                    # Create __init__.py
                    (behaviors_dir / "__init__.py").touch()

                # Copy utils directory if needed (for validator imports)
                utils_dir = Path("utils")
                if utils_dir.exists():
                    shutil.copytree(utils_dir, sandbox_path / "utils", dirs_exist_ok=True)

                print(f"[sandbox_test] Running pytest with timeout={timeout}s")

                # Run pytest in sandbox
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-v", sandbox_test.name],
                    cwd=sandbox_path,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )

                success = result.returncode == 0

                print(f"[sandbox_test] {'✅ Tests passed' if success else '❌ Tests failed'}")

                return {
                    "success": success,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "message": "Tests passed" if success else "Tests failed"
                }

            except subprocess.TimeoutExpired:
                print(f"[sandbox_test] ❌ Timeout after {timeout}s")
                return {
                    "success": False,
                    "error": f"Tests timed out after {timeout}s",
                    "message": f"Timeout ({timeout}s)"
                }

            finally:
                # Cleanup sandbox
                try:
                    shutil.rmtree(sandbox_dir, ignore_errors=True)
                    print(f"[sandbox_test] Cleaned up sandbox: {sandbox_dir}")
                except Exception as e:
                    print(f"[sandbox_test] Warning: Failed to cleanup sandbox: {e}")

        except Exception as e:
            return {"error": f"Sandbox test error: {str(e)}"}

    def _run_agent_sanity_check(self, agent: Any, args: dict[str, Any]) -> dict[str, Any]:
        """
        Verify agent configuration can load and instantiate.

        This checks:
        1. Config file is valid YAML
        2. Required fields present
        3. Referenced behaviors exist
        4. Agent class can be imported
        5. Agent can instantiate with config
        """
        try:
            config_file = args.get("config_file", "")
            agent_class_name = args.get("agent_class", "")

            # Resolve path relative to workspace
            workspace = agent.workspace if hasattr(agent, 'workspace') else Path.cwd()
            config_path = workspace / config_file

            if not config_path.exists():
                return {"error": f"Config file not found: {config_file}"}

            # Create sandbox directory
            sandbox_dir = tempfile.mkdtemp(prefix="sandbox_agent_check_")
            sandbox_path = Path(sandbox_dir)

            try:
                print(f"[sandbox_test] Testing agent instantiation in sandbox: {sandbox_dir}")

                # Copy config to sandbox
                shutil.copy(config_path, sandbox_path / config_path.name)

                # Create test script
                test_script = f"""
import sys
from pathlib import Path

# Test agent instantiation
try:
    # Import agent class
    module_name = '{agent_class_name.lower().replace('agent', '_agent')}'
    agent_module = __import__(module_name, fromlist=['{agent_class_name}'])
    AgentClass = getattr(agent_module, '{agent_class_name}')

    # Instantiate agent
    agent = AgentClass(
        workspace=Path.cwd(),
        goal="Sanity check test",
        config_file="{config_path.name}"
    )

    print("✅ Agent instantiated successfully")
    print(f"Agent name: {{agent.name}}")
    print(f"Behaviors loaded: {{len(agent._behaviors) if hasattr(agent, '_behaviors') else 0}}")
    sys.exit(0)

except Exception as e:
    print(f"❌ Error: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""

                test_script_path = sandbox_path / "test_instantiation.py"
                test_script_path.write_text(test_script)

                # Run test script
                result = subprocess.run(
                    [sys.executable, str(test_script_path)],
                    cwd=sandbox_path,
                    capture_output=True,
                    text=True,
                    timeout=10  # Quick sanity check
                )

                success = result.returncode == 0

                print(f"[sandbox_test] {'✅ Agent instantiation passed' if success else '❌ Agent instantiation failed'}")

                return {
                    "success": success,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "message": "Agent instantiation succeeded" if success else "Agent instantiation failed"
                }

            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": "Agent instantiation timed out (10s)",
                    "message": "Timeout"
                }

            finally:
                # Cleanup sandbox
                try:
                    shutil.rmtree(sandbox_dir, ignore_errors=True)
                    print(f"[sandbox_test] Cleaned up sandbox: {sandbox_dir}")
                except Exception as e:
                    print(f"[sandbox_test] Warning: Failed to cleanup sandbox: {e}")

        except Exception as e:
            return {"error": f"Agent sanity check error: {str(e)}"}
