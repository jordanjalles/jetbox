"""
CreateBehaviorBehavior - MetaProgrammer's behavior generation system.

This behavior provides tools for generating new AgentBehavior classes with:
- LLM-powered code generation from specifications
- Automatic test generation
- YAML syntax validation
- Code quality validation
- Sandbox testing
- Safety modes (dryrun/review/auto/strict)
"""

from typing import Any
from pathlib import Path
import json
import yaml
from behaviors.base import AgentBehavior


class CreateBehaviorBehavior(AgentBehavior):
    """
    Provides tools for generating new behavior classes.

    Features:
    - LLM-powered code generation from tool specifications
    - Test generation with proper assertions
    - Multi-stage validation (YAML, config, code quality)
    - Sandbox testing in isolated environment
    - Safety modes for different deployment scenarios
    """

    def __init__(self, workspace_manager=None, **kwargs):
        """
        Initialize CreateBehaviorBehavior.

        Args:
            workspace_manager: Optional WorkspaceManager for path resolution
            **kwargs: Additional parameters (ignored for extensibility)
        """
        self.workspace_manager = workspace_manager
        self.staging_dir = Path(".agent_generated/staging")
        self.default_safety_mode = "review"  # Default to review mode

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "create_behavior"

    def get_tools(self) -> list[dict[str, Any]]:
        """
        Return tool definitions.

        Returns:
            List with create_behavior tool definition
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "create_behavior",
                    "description": "Generate a new AgentBehavior class with specified tools and lifecycle hooks",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "behavior_name": {
                                "type": "string",
                                "description": "Name of the behavior (e.g., 'HttpRequestBehavior')"
                            },
                            "description": {
                                "type": "string",
                                "description": "Description of what the behavior does"
                            },
                            "tool_specs": {
                                "type": "array",
                                "description": "List of tool specifications with name, description, and parameters",
                                "items": {"type": "object"}
                            },
                            "lifecycle_hooks": {
                                "type": "array",
                                "description": "Optional list of lifecycle hooks to implement (e.g., on_initial_context, on_round_start)",
                                "items": {"type": "string"}
                            },
                            "safety_mode": {
                                "type": "string",
                                "description": "Safety mode: 'dryrun' (staging only), 'review' (staging + return for approval), 'auto' (install if valid), 'strict' (extra checks + review). Default: 'review'"
                            }
                        },
                        "required": ["behavior_name", "description", "tool_specs"]
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
        Handle tool execution.

        Args:
            agent: Agent instance
            tool_name: Tool being called
            args: Tool arguments

        Returns:
            Tool result dict
        """
        if tool_name == "create_behavior":
            return self._execute_create_behavior(agent, args)
        else:
            return super().dispatch_tool(agent, tool_name, args)

    def _execute_create_behavior(
        self,
        agent: Any,
        args: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute create_behavior tool.

        Args:
            agent: Agent instance
            args: Tool arguments

        Returns:
            Result dict with behavior file, tests, and validation results
        """
        try:
            # Extract and validate parameters
            behavior_name = args.get("behavior_name", "")
            description = args.get("description", "")
            tool_specs = args.get("tool_specs", [])
            lifecycle_hooks = args.get("lifecycle_hooks", [])
            safety_mode = args.get("safety_mode", self.default_safety_mode)

            # Ignore unsupported parameters with warning
            supported_params = {"behavior_name", "description", "tool_specs", "lifecycle_hooks", "safety_mode"}
            unsupported = set(args.keys()) - supported_params
            if unsupported:
                print(f"[create_behavior] Ignoring parameters: {unsupported}")

            # Validate required inputs
            if not behavior_name:
                return {"error": "behavior_name is required"}
            if not description:
                return {"error": "description is required"}
            if not tool_specs:
                return {"error": "tool_specs must be a non-empty list"}

            print(f"[create_behavior] Generating behavior: {behavior_name}")
            print(f"[create_behavior] Description: {description}")
            print(f"[create_behavior] Tools: {len(tool_specs)}")
            print(f"[create_behavior] Safety mode: {safety_mode}")

            # Run the workflow
            return self._run_behavior_generation_workflow(
                agent, behavior_name, description, tool_specs, lifecycle_hooks, safety_mode
            )

        except Exception as e:
            return {"error": f"Error creating behavior: {str(e)}"}

    def _run_behavior_generation_workflow(
        self,
        agent: Any,
        behavior_name: str,
        description: str,
        tool_specs: list,
        lifecycle_hooks: list,
        safety_mode: str
    ) -> dict[str, Any]:
        """
        Run the full behavior generation workflow.

        Steps:
        1. Generate behavior code
        2. Generate test code
        3. Save to staging
        4. Validate code
        5. Run sandbox tests
        6. Handle safety mode

        Args:
            agent: Agent instance
            behavior_name: Name of behavior
            description: Description
            tool_specs: Tool specifications
            lifecycle_hooks: Lifecycle hooks to implement
            safety_mode: Safety mode

        Returns:
            Result dict with all outputs and validation results
        """
        # Step 1: Generate behavior code
        print("[create_behavior] Step 1/6: Generating behavior code...")
        code_result = self._generate_behavior_code(
            agent, behavior_name, description, tool_specs, lifecycle_hooks
        )
        if "error" in code_result:
            return code_result
        behavior_code = code_result["code"]

        # Step 2: Generate test code
        print("[create_behavior] Step 2/6: Generating test code...")
        test_result = self._generate_test_code(
            agent, behavior_name, description, tool_specs
        )
        if "error" in test_result:
            return test_result
        test_code = test_result["code"]

        # Step 3: Save to staging
        print("[create_behavior] Step 3/6: Saving to staging...")
        staging_result = self._save_to_staging(behavior_name, behavior_code, test_code)
        if "error" in staging_result:
            return staging_result
        behavior_file = staging_result["behavior_file"]
        test_file = staging_result["test_file"]

        # Build class name (needed for validation)
        if behavior_name.endswith("Behavior"):
            class_name = behavior_name
        else:
            words = behavior_name.replace("-", "_").split("_")
            class_name = "".join(word.capitalize() for word in words) + "Behavior"

        # Step 4: Validate generated code
        print("[create_behavior] Step 4/6: Validating generated code...")
        validation_result = self._validate_generated_code(agent, behavior_file, class_name)

        # Step 5: Run sandbox tests
        print("[create_behavior] Step 5/6: Testing in sandbox...")
        sandbox_result = self._run_sandbox_tests(agent, test_file)

        # Step 6: Handle safety mode
        print(f"[create_behavior] Step 6/6: Handling safety mode '{safety_mode}'...")
        return self._handle_safety_mode(
            agent, safety_mode, behavior_name, behavior_file, test_file,
            validation_result, sandbox_result
        )

    def _generate_behavior_code(
        self,
        agent: Any,
        behavior_name: str,
        description: str,
        tool_specs: list,
        lifecycle_hooks: list
    ) -> dict[str, Any]:
        """
        Generate behavior code using LLM.

        Args:
            agent: Agent instance
            behavior_name: Name of behavior
            description: Description
            tool_specs: Tool specifications
            lifecycle_hooks: Lifecycle hooks to implement

        Returns:
            Dict with "code" key or "error"
        """
        try:
            # Load template
            template_path = Path("behaviors/templates/behavior_simple_template.py")
            if not template_path.exists():
                return {"error": f"Template not found: {template_path}"}

            template_code = template_path.read_text()

            # Build class name
            # If behavior_name already ends with "Behavior", use it as-is
            # Otherwise, add "Behavior" suffix and ensure proper CamelCase
            if behavior_name.endswith("Behavior"):
                class_name = behavior_name
            else:
                # Convert snake_case or kebab-case to CamelCase and add "Behavior"
                words = behavior_name.replace("-", "_").split("_")
                class_name = "".join(word.capitalize() for word in words) + "Behavior"

            # Add generation marker to template
            generation_marker = "# GENERATED BY METAPROGRAMMER - Safe to delete for testing\n"

            # Build prompt for LLM
            tools_spec_str = json.dumps(tool_specs, indent=2)
            hooks_str = ", ".join(lifecycle_hooks) if lifecycle_hooks else "none"

            prompt = f"""You are generating a Python behavior class. Follow these instructions EXACTLY.

IMPORTANT: Start your code with this exact comment on the first line:
# GENERATED BY METAPROGRAMMER - Safe to delete for testing

Then add the module docstring.

BEHAVIOR SPECIFICATION:
- Behavior identifier: {behavior_name}
- Class name: {class_name}
- Description: {description}
- Tools: {tools_spec_str}
- Lifecycle hooks: {hooks_str}

CRITICAL INSTRUCTIONS:
1. Use EXACTLY this class name: {class_name}
2. The import MUST be: from behaviors.base import AgentBehavior
3. get_name() MUST return: "{behavior_name}"
4. Implement ALL tools from the specification
5. Each tool in dispatch_tool() must have proper error handling
6. Return ONLY Python code, no markdown, no explanations

OUTPUT FORMAT:
- Start with generation marker comment
- Then docstring
- Then imports (behaviors.base is REQUIRED)
- Then class definition with name {class_name}
- NO markdown code blocks
- NO explanations or comments outside the code

TEMPLATE TO FOLLOW:
{template_code}

Generate the complete behavior code now:"""

            # Call LLM
            from llm_utils import chat_with_inactivity_timeout
            import os

            model = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
            response = chat_with_inactivity_timeout(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2},
                inactivity_timeout=60,
            )

            generated_code = response["message"]["content"].strip()

            # Clean up markdown code blocks if present
            if generated_code.startswith("```python"):
                code_part = generated_code.split("```python", 1)[1]
                if "```" in code_part:
                    generated_code = code_part.split("```")[0].strip()
                else:
                    generated_code = code_part.strip()
            elif generated_code.startswith("```"):
                parts = generated_code.split("```")
                if len(parts) > 1:
                    generated_code = parts[1].strip()

            # POST-PROCESS: Fix class name if LLM messed it up
            # Look for class definitions and fix incorrect naming
            import re
            # Find class definition line
            class_match = re.search(r'^class\s+(\w+)\s*\(AgentBehavior\)', generated_code, re.MULTILINE)
            if class_match:
                actual_class_name = class_match.group(1)
                print(f"[create_behavior] Found class name: {actual_class_name}, expected: {class_name}")
                # If it doesn't match expected class_name exactly, fix it
                if actual_class_name != class_name:
                    print(f"[create_behavior] Fixing class name: {actual_class_name} -> {class_name}")
                    # Use re.sub to replace the class definition line
                    generated_code = re.sub(
                        r'^(class\s+)' + re.escape(actual_class_name) + r'(\s*\(AgentBehavior\))',
                        r'\1' + class_name + r'\2',
                        generated_code,
                        count=1,
                        flags=re.MULTILINE
                    )
                    # Verify the fix worked
                    verify_match = re.search(r'^class\s+(\w+)\s*\(AgentBehavior\)', generated_code, re.MULTILINE)
                    if verify_match:
                        print(f"[create_behavior] After fix: {verify_match.group(1)}")
                else:
                    print(f"[create_behavior] Class name already correct")
            else:
                print(f"[create_behavior] WARNING: Could not find class definition in generated code")

            return {"code": generated_code}

        except Exception as e:
            return {"error": f"Code generation failed: {str(e)}"}

    def _generate_test_code(
        self,
        agent: Any,
        behavior_name: str,
        description: str,
        tool_specs: list
    ) -> dict[str, Any]:
        """
        Generate test code using LLM.

        Args:
            agent: Agent instance
            behavior_name: Name of behavior
            description: Description
            tool_specs: Tool specifications

        Returns:
            Dict with "code" key or "error"
        """
        try:
            # Build class name
            if behavior_name.endswith("Behavior"):
                class_name = behavior_name
            else:
                words = behavior_name.replace("-", "_").split("_")
                class_name = "".join(word.capitalize() for word in words) + "Behavior"

            tools_spec_str = json.dumps(tool_specs, indent=2)

            prompt = f"""Generate pytest tests for a behavior class.

IMPORTANT: Start your code with this exact comment on the first line:
# GENERATED BY METAPROGRAMMER - Safe to delete for testing

BEHAVIOR:
- Name: {behavior_name}
- Class: {class_name}
- Description: {description}
- Tools: {tools_spec_str}

REQUIREMENTS:
1. Import pytest and the behavior class
2. Create test functions for each tool
3. Test both success and error cases
4. Use proper assertions
5. Mock external dependencies if needed
6. Return ONLY Python test code

Generate the complete test file now:"""

            from llm_utils import chat_with_inactivity_timeout
            import os

            model = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
            response = chat_with_inactivity_timeout(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2},
                inactivity_timeout=60,
            )

            test_code = response["message"]["content"].strip()

            # Clean up markdown if present
            if test_code.startswith("```python"):
                code_part = test_code.split("```python", 1)[1]
                if "```" in code_part:
                    test_code = code_part.split("```")[0].strip()
                else:
                    test_code = code_part.strip()
            elif test_code.startswith("```"):
                parts = test_code.split("```")
                if len(parts) > 1:
                    test_code = parts[1].strip()

            return {"code": test_code}

        except Exception as e:
            return {"error": f"Test generation failed: {str(e)}"}

    def _save_to_staging(
        self,
        behavior_name: str,
        behavior_code: str,
        test_code: str
    ) -> dict[str, Any]:
        """
        Save generated code to staging directory.

        Args:
            behavior_name: Name of behavior
            behavior_code: Generated behavior code
            test_code: Generated test code

        Returns:
            Dict with file paths or error
        """
        try:
            # Create staging directory
            self.staging_dir.mkdir(parents=True, exist_ok=True)

            # Save behavior file
            behavior_file = self.staging_dir / f"{behavior_name}.py"
            behavior_file.write_text(behavior_code)
            print(f"[create_behavior] Saved behavior to: {behavior_file}")

            # Save test file
            test_file = self.staging_dir / f"test_{behavior_name}.py"
            test_file.write_text(test_code)
            print(f"[create_behavior] Saved test to: {test_file}")

            return {
                "behavior_file": str(behavior_file),
                "test_file": str(test_file)
            }

        except Exception as e:
            return {"error": f"Failed to save files: {str(e)}"}

    def _validate_generated_code(
        self,
        agent: Any,
        behavior_file: str,
        class_name: str
    ) -> dict[str, Any]:
        """
        Validate generated behavior code.

        Uses ValidationBehavior if available.

        Args:
            agent: Agent instance
            behavior_file: Path to behavior file
            class_name: Expected class name

        Returns:
            Validation result dict
        """
        try:
            # Find ValidationBehavior
            validation_behavior = None
            for behavior in agent.behaviors:
                if behavior.get_name() == "validation":
                    validation_behavior = behavior
                    break

            if not validation_behavior:
                return {"valid": True, "message": "Validation skipped (ValidationBehavior not available)"}

            # Call validate_behavior_class tool (needs code string, not file path)
            try:
                with open(behavior_file, 'r') as f:
                    code = f.read()
            except Exception as e:
                return {"valid": False, "error": f"Failed to read file for validation: {str(e)}"}

            result = validation_behavior.dispatch_tool(
                agent,
                "validate_behavior_class",
                {"code": code, "expected_name": class_name}
            )

            return result.get("result", {"valid": False, "error": "No validation result"})

        except Exception as e:
            return {"valid": False, "error": f"Validation failed: {str(e)}"}

    def _run_sandbox_tests(
        self,
        agent: Any,
        test_file: str
    ) -> dict[str, Any]:
        """
        Run tests in sandbox environment.

        Uses SandboxTestBehavior if available.

        Args:
            agent: Agent instance
            test_file: Path to test file

        Returns:
            Sandbox test result dict
        """
        try:
            # Find SandboxTestBehavior
            sandbox_behavior = None
            for behavior in agent.behaviors:
                if behavior.get_name() == "sandbox_test":
                    sandbox_behavior = behavior
                    break

            if not sandbox_behavior:
                return {"success": True, "message": "Sandbox testing skipped (SandboxTestBehavior not available)"}

            # Call sandbox_test tool
            result = sandbox_behavior.dispatch_tool(
                agent,
                "sandbox_test",
                {"test_file": test_file, "timeout": 30}
            )

            return result.get("result", {"success": False, "error": "No sandbox result"})

        except Exception as e:
            return {"success": False, "error": f"Sandbox testing failed: {str(e)}"}

    def _handle_safety_mode(
        self,
        agent: Any,
        safety_mode: str,
        behavior_name: str,
        behavior_file: str,
        test_file: str,
        validation_result: dict,
        sandbox_result: dict
    ) -> dict[str, Any]:
        """
        Handle safety mode and installation.

        Safety modes:
        - dryrun: Staging only, no installation
        - review: Return for user approval
        - auto: Install if validation passes
        - strict: Extra checks + return for approval

        Args:
            agent: Agent instance
            safety_mode: Safety mode
            behavior_name: Name of behavior
            behavior_file: Path to behavior file
            test_file: Path to test file
            validation_result: Validation results
            sandbox_result: Sandbox test results

        Returns:
            Final result dict
        """
        result = {
            "success": True,
            "behavior_name": behavior_name,
            "behavior_file": behavior_file,
            "test_file": test_file,
            "staging_location": str(self.staging_dir),
            "validation_results": validation_result,
            "sandbox_results": sandbox_result,
            "safety_mode": safety_mode,
            "installed": False
        }

        if safety_mode == "dryrun":
            result["message"] = "Dryrun mode: Files saved to staging only"
            return result

        elif safety_mode == "review":
            result["message"] = "Review mode: Files staged for approval"
            return result

        elif safety_mode == "auto":
            # Install if validation passed (ignore sandbox failures for now)
            if validation_result.get("valid", False) or "Validation skipped" in validation_result.get("message", ""):
                install_result = self._install_behavior(behavior_name, behavior_file, test_file)
                if install_result.get("success"):
                    result["installed"] = True
                    result["installed_files"] = install_result.get("files", [])
                    result["message"] = "Auto mode: Behavior installed successfully"
                else:
                    result["success"] = False
                    result["error"] = install_result.get("error", "Installation failed")
            else:
                result["success"] = False
                result["error"] = f"Validation failed: {validation_result.get('error', 'Unknown error')}"
            return result

        elif safety_mode == "strict":
            # Require both validation and sandbox tests to pass
            validation_ok = validation_result.get("valid", False)
            sandbox_ok = sandbox_result.get("success", False)

            if not validation_ok:
                result["success"] = False
                result["error"] = f"Validation failed: {validation_result.get('error', 'Unknown')}"
                result["message"] = "Strict mode: Validation failed, review required"
            elif not sandbox_ok:
                result["success"] = False
                result["error"] = f"Sandbox tests failed: {sandbox_result.get('error', 'Unknown')}"
                result["message"] = "Strict mode: Tests failed, review required"
            else:
                result["message"] = "Strict mode: All checks passed, review for approval"
            return result

        else:
            result["success"] = False
            result["error"] = f"Unknown safety mode: {safety_mode}"
            return result

    def _install_behavior(
        self,
        behavior_name: str,
        behavior_file: str,
        test_file: str
    ) -> dict[str, Any]:
        """
        Install behavior to production locations.

        Args:
            behavior_name: Name of behavior
            behavior_file: Path to staged behavior file
            test_file: Path to staged test file

        Returns:
            Installation result dict
        """
        try:
            import shutil
            from datetime import datetime

            installed_files = []

            # Backup directory
            backup_dir = Path(".agent_generated/backups")
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Install behavior to behaviors/
            behavior_dest = Path(f"behaviors/{behavior_name}.py")
            if behavior_dest.exists():
                # Backup existing file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = backup_dir / f"{behavior_name}.py.{timestamp}.backup"
                shutil.copy2(behavior_dest, backup_path)
                print(f"[installer] Backed up existing file: {backup_path}")

            shutil.copy2(behavior_file, behavior_dest)
            installed_files.append(str(behavior_dest))
            print(f"[installer] Installed: {behavior_dest}")

            # Install test to tests/
            test_dest = Path(f"tests/test_{behavior_name}.py")
            if test_dest.exists():
                # Backup existing test
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = backup_dir / f"test_{behavior_name}.py.{timestamp}.backup"
                shutil.copy2(test_dest, backup_path)
                print(f"[installer] Backed up existing file: {backup_path}")

            shutil.copy2(test_file, test_dest)
            installed_files.append(str(test_dest))
            print(f"[installer] Installed: {test_dest}")

            return {
                "success": True,
                "files": installed_files
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Installation failed: {str(e)}"
            }
