"""
Message serialization utilities.

Provides unified serialization logic for messages and tool calls
to ensure consistent JSON-compatible formatting across the codebase.
"""
from __future__ import annotations
from typing import Any


def serialize_tool_call(tool_call: Any) -> dict[str, Any]:
    """
    Serialize a single tool call to JSON-compatible format.

    Args:
        tool_call: ToolCall object (may be Pydantic model, dict, or object)

    Returns:
        JSON-serializable dict representation of the tool call
    """
    if hasattr(tool_call, "model_dump"):
        # Pydantic model
        return tool_call.model_dump()
    elif hasattr(tool_call, "to_dict"):
        # Has to_dict method
        return tool_call.to_dict()
    elif isinstance(tool_call, dict):
        # Already a dict
        return tool_call
    else:
        # Manual extraction fallback
        try:
            return {
                "id": getattr(tool_call, "id", None),
                "type": getattr(tool_call, "type", "function"),
                "function": {
                    "name": getattr(tool_call.function, "name", "") if hasattr(tool_call, "function") else "",
                    "arguments": getattr(tool_call.function, "arguments", {}) if hasattr(tool_call, "function") else {}
                }
            }
        except Exception:
            # Last resort: convert to string representation
            return {"_error": "Could not serialize ToolCall", "_repr": str(tool_call)}


def serialize_message(message: dict[str, Any]) -> dict[str, Any]:
    """
    Serialize a message to JSON-compatible format.

    Handles ToolCall objects and other non-JSON-serializable types.

    Args:
        message: Message dict (may contain ToolCall objects in tool_calls field)

    Returns:
        JSON-serializable message dict
    """
    serialized = {}
    for key, value in message.items():
        if key == "tool_calls" and value is not None:
            # Convert ToolCall objects to dicts
            serialized[key] = [serialize_tool_call(tc) for tc in value]
        else:
            serialized[key] = value
    return serialized
