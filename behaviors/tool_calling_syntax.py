"""
ToolCallingSyntaxBehavior - Manages tool calling format presentation and parsing.

This behavior handles how tool calls are presented to the LLM and how they're
parsed from responses. Supports multiple formats (JSON, XML, YAML) with fallback.

Key features:
- Injects format examples into context
- Parses tool calls from LLM responses
- Supports multiple formats with fallback
- Swappable syntax handlers

Event: on_initial_context() - Add format example after system prompt
Event: on_llm_response() - Parse tool calls from content if not natively parsed
"""

from typing import Any
import re
from behaviors.base import AgentBehavior


class ToolCallingSyntaxBehavior(AgentBehavior):
    """
    Manages tool calling format for LLM interactions.

    Responsibilities:
    - Inject format examples into context
    - Parse tool calls from LLM response content
    - Support multiple formats with fallback

    Swappable syntax handlers:
    - JsonToolCallingSyntax (default, Ollama native)
    - XmlToolCallingSyntax (Anthropic/Claude style)
    - YamlToolCallingSyntax (future)

    Security: [] None (utility behavior for format management)
    """

    # Rule of Two: Empty (utility behavior for format management)
    rule_of_two_properties = set()

    def __init__(
        self,
        syntax: str = "json",
        fallback_syntaxes: list[str] | None = None
    ):
        """
        Initialize tool calling syntax behavior.

        Args:
            syntax: Primary syntax ("json", "xml", "yaml")
            fallback_syntaxes: Try these if primary fails (e.g., ["xml", "json"])
        """
        super().__init__()
        self.primary_syntax = self._get_syntax_handler(syntax)
        self.fallback_syntaxes = [
            self._get_syntax_handler(s)
            for s in (fallback_syntaxes or [])
        ]

    def get_name(self) -> str:
        """Return behavior identifier."""
        return "tool_calling_syntax"

    def get_sequence_number(self) -> int:
        """Run early to inject examples, and late to parse responses."""
        # Early for inject_initial_context (10)
        # Late for on_llm_response (950, before context inspector at 999)
        return 10

    def _get_syntax_handler(self, syntax: str):
        """Get syntax handler for format name."""
        handlers = {
            "json": JsonToolCallingSyntax(),
            "xml": XmlToolCallingSyntax(),
        }
        if syntax not in handlers:
            raise ValueError(f"Unknown syntax: {syntax}. Available: {list(handlers.keys())}")
        return handlers[syntax]

    def on_initial_context(self, agent: Any, context: list) -> list:
        """Add format example after system prompt."""
        example = self.primary_syntax.get_example()

        # Insert after system prompt (position 1)
        context.insert(1, {
            "role": "user",
            "content": example
        })

        return context

    def on_llm_response(self, agent: Any, response: dict) -> dict:
        """
        Parse tool calls from response content if not natively parsed by Ollama.

        This runs after LLM returns but before tool dispatch. If Ollama already
        parsed tool_calls, we're done. Otherwise, try to extract tool calls from
        content using primary and fallback parsers.

        Args:
            agent: Agent instance
            response: LLM response dict

        Returns:
            Modified response with tool_calls injected if found
        """
        # If already parsed, we're done
        if response.get("message", {}).get("tool_calls"):
            return response

        content = response.get("message", {}).get("content", "")
        if not content:
            return response

        # Try primary parser
        tool_calls = self.primary_syntax.parse_tool_calls(content)

        # Try fallbacks if primary failed
        if not tool_calls:
            for fallback in self.fallback_syntaxes:
                tool_calls = fallback.parse_tool_calls(content)
                if tool_calls:
                    break

        # Inject parsed tool_calls into response
        if tool_calls:
            if "message" not in response:
                response["message"] = {}
            response["message"]["tool_calls"] = tool_calls

        return response


class JsonToolCallingSyntax:
    """Standard Ollama JSON tool calling format."""

    def get_example(self) -> str:
        """Return JSON format example for context."""
        return """## Tool Calling Format

When calling tools, respond with ONLY a JSON object (no explanatory text):

```json
{
  "name": "tool_name",
  "arguments": {
    "arg1": "value1",
    "arg2": "value2"
  }
}
```

**CRITICAL RULES:**
1. NO text before the JSON
2. NO text after the JSON
3. ONLY the JSON object

Examples:

❌ WRONG: "Let me call this tool: {\\"name\\": \\"write_file\\", ...}"
❌ WRONG: "{\\"name\\": \\"write_file\\", ...} This will create the file"
❌ WRONG: <function=write_file><parameter=path>test.py</parameter></function>

✅ CORRECT: {"name": "write_file", "arguments": {"path": "test.py", "content": "print('hello')"}}
✅ CORRECT: {"name": "mark_complete", "arguments": {"summary": "Task finished successfully"}}
"""

    def parse_tool_calls(self, content: str) -> list[dict] | None:
        """
        Parse JSON tool calls from content.

        JSON is normally parsed by Ollama natively, so this is primarily a
        fallback for edge cases where JSON is embedded in text.

        Args:
            content: LLM response content

        Returns:
            List of tool call dicts, or None if no valid JSON found
        """
        import json

        # Strategy 1: Try parsing the full content as JSON (common case)
        try:
            content_stripped = content.strip()
            parsed = json.loads(content_stripped)
            if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                # Validate arguments is a dict, not a string
                if not isinstance(parsed.get("arguments"), dict):
                    pass  # Skip to next strategy
                else:
                    return [{
                        "function": {
                            "name": parsed["name"],
                            "arguments": parsed["arguments"]
                        }
                    }]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # Strategy 2: Find JSON by brace counting (handles trailing garbage)
        # LLM might output: {"name": "write_file", "arguments": {...}}\n</parameter>}
        # We want to extract just the valid JSON part
        content = content.strip()
        if content.startswith('{'):
            brace_count = 0
            json_end = 0
            for i, char in enumerate(content):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break

            if json_end > 0:
                json_str = content[:json_end]
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                        # Validate arguments is a dict, not a string
                        if not isinstance(parsed.get("arguments"), dict):
                            pass  # Skip to next strategy
                        else:
                            return [{
                                "function": {
                                    "name": parsed["name"],
                                    "arguments": parsed["arguments"]
                                }
                            }]
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass

        # Strategy 3: Regex fallback (for JSON embedded in text)
        json_pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}[^{}]*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)
        if not matches:
            return None

        tool_calls = []
        for match in matches:
            try:
                parsed = json.loads(match)
                # Validate arguments is a dict, not a string
                if not isinstance(parsed.get("arguments"), dict):
                    continue  # Skip this match
                tool_calls.append({
                    "function": {
                        "name": parsed["name"],
                        "arguments": parsed["arguments"]
                    }
                })
            except (json.JSONDecodeError, KeyError):
                continue

        return tool_calls if tool_calls else None


class XmlToolCallingSyntax:
    """XML tool calling format (Anthropic/Claude style)."""

    def get_example(self) -> str:
        """Return XML format example for context."""
        return """## Tool Calling Format

When calling tools, use this XML format:

<function_calls>
<invoke name="tool_name">
<parameter name="arg1">value1</parameter>
<parameter name="arg2">value2</parameter>
</invoke>
</function_calls>

Examples:

<function_calls>
<invoke name="write_file">
<parameter name="path">test.py</parameter>
<parameter name="content">print('hello')</parameter>
</invoke>
</function_calls>

<function_calls>
<invoke name="mark_complete">
<parameter name="summary">Task finished successfully</parameter>
</invoke>
</function_calls>
"""

    def parse_tool_calls(self, content: str) -> list[dict] | None:
        """
        Extract XML tool calls and convert to standard format.

        Parses two XML formats:
        1. Anthropic style: <invoke name="tool_name"><parameter name="arg">value</parameter></invoke>
        2. qwen3-coder style: <function=tool_name><parameter=arg>value</parameter></function>

        Args:
            content: LLM response content

        Returns:
            List of tool call dicts in standard format, or None if no XML found
        """
        tool_calls = []

        # Try Anthropic format first: <invoke name="tool_name">..params..</invoke>
        invoke_pattern = r'<invoke name="([^"]+)">(.*?)</invoke>'
        param_named_pattern = r'<parameter name="([^"]+)">([^<]*)</parameter>'

        for tool_match in re.finditer(invoke_pattern, content, re.DOTALL):
            tool_name = tool_match.group(1)
            params_block = tool_match.group(2)

            arguments = {}
            for param_match in re.finditer(param_named_pattern, params_block):
                arg_name = param_match.group(1)
                arg_value = param_match.group(2).strip()
                arguments[arg_name] = arg_value

            tool_calls.append({
                "function": {
                    "name": tool_name,
                    "arguments": arguments
                }
            })

        # Try qwen3-coder format: <function=tool_name>..params..</function>
        function_pattern = r'<function=([^>]+)>(.*?)</function>'
        param_equals_pattern = r'<parameter=([^>]+)>(.*?)</parameter>'

        for tool_match in re.finditer(function_pattern, content, re.DOTALL):
            tool_name = tool_match.group(1)
            params_block = tool_match.group(2)

            arguments = {}
            for param_match in re.finditer(param_equals_pattern, params_block, re.DOTALL):
                arg_name = param_match.group(1)
                arg_value = param_match.group(2).strip()
                arguments[arg_name] = arg_value

            tool_calls.append({
                "function": {
                    "name": tool_name,
                    "arguments": arguments
                }
            })

        return tool_calls if tool_calls else None
