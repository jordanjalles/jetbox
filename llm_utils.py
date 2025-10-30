"""
LLM utilities for Jetbox agents.

Provides wrapper functions for calling Ollama models with proper timeout handling.
"""
from __future__ import annotations

import os
from threading import Thread
from typing import Any

from ollama import Client

# Initialize Ollama client with proper host configuration
OLLAMA_CLIENT = Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))


def chat_with_inactivity_timeout(
    model: str,
    messages: list,
    options: dict,
    inactivity_timeout: int = 30,
    tools: list | None = None,
    max_total_time: int | None = None,
) -> dict[str, Any]:
    """
    Call ollama chat with INACTIVITY timeout and optional TOTAL time limit.

    This allows complex tasks to take as long as needed, but detects
    when Ollama has actually stopped responding (no chunks for N seconds)
    and optionally enforces a maximum total generation time.

    Args:
        model: Model name
        messages: List of messages
        options: Options dict (temperature, etc)
        inactivity_timeout: Max seconds without activity (default 30s)
        tools: Optional list of tool specifications for function calling
        max_total_time: Optional maximum total time in seconds (default None = no limit)

    Returns:
        Response dict from ollama

    Raises:
        TimeoutError: If no response activity for inactivity_timeout seconds
                     or if max_total_time exceeded
    """
    from queue import Queue, Empty

    result_queue: Queue = Queue()

    def _stream_chat():
        try:
            # Ensure options includes large context window
            # Default Ollama is only 2048 tokens, but models support much more
            options_with_context = options.copy()
            if "num_ctx" not in options_with_context:
                # Set to 128K (131072) to match gpt-oss:20b capacity
                options_with_context["num_ctx"] = 131072

            # Build chat arguments
            chat_args = {
                "model": model,
                "messages": messages,
                "options": options_with_context,
                "stream": True
            }
            if tools is not None:
                chat_args["tools"] = tools

            full_response = {"message": {"role": "assistant", "content": ""}}

            # Use streaming to detect activity
            for chunk in OLLAMA_CLIENT.chat(**chat_args):
                # Signal activity
                result_queue.put(("chunk", chunk))

                # Accumulate content
                content = chunk.get("message", {}).get("content", "")
                if content:
                    full_response["message"]["content"] += content

                # Get tool calls from final chunk
                tool_calls = chunk.get("message", {}).get("tool_calls")
                if tool_calls:
                    full_response["message"]["tool_calls"] = tool_calls

                # Preserve token counts from final chunk
                if chunk.get("done"):
                    chunk_dict = dict(chunk) if hasattr(chunk, "__dict__") or hasattr(chunk, "keys") else chunk
                    for key in ["prompt_eval_count", "eval_count", "total_duration", "prompt_eval_duration", "eval_duration"]:
                        if key in chunk_dict:
                            full_response[key] = chunk_dict[key]

            # Stream complete
            result_queue.put(("done", full_response))
        except Exception as e:
            result_queue.put(("error", e))

    thread = Thread(target=_stream_chat, daemon=True)
    thread.start()

    # Track total time
    import time
    start_time = time.time()

    # Monitor for inactivity and total time
    while True:
        try:
            msg_type, data = result_queue.get(timeout=inactivity_timeout)

            if msg_type == "chunk":
                # Activity detected, check total time
                elapsed = time.time() - start_time
                if max_total_time and elapsed > max_total_time:
                    # Total time exceeded - dump context and raise
                    _dump_timeout_context(model, messages, tools, elapsed, "max_total_time")
                    raise TimeoutError(
                        f"LLM call exceeded max_total_time of {max_total_time}s "
                        f"(elapsed: {elapsed:.1f}s). Context dumped to .agent_context/timeout_dumps/"
                    )
                continue
            elif msg_type == "done":
                # Success - return full response
                return data
            elif msg_type == "error":
                # Error during streaming
                raise data

        except Empty:
            # No activity for inactivity_timeout seconds = Ollama is hung
            elapsed = time.time() - start_time
            _dump_timeout_context(model, messages, tools, elapsed, "inactivity")
            raise TimeoutError(
                f"No response from Ollama for {inactivity_timeout}s - likely hung or dead. "
                f"Context dumped to .agent_context/timeout_dumps/"
            )


def _dump_timeout_context(
    model: str,
    messages: list,
    tools: list | None,
    elapsed_time: float,
    timeout_type: str,
) -> None:
    """
    Dump context to file when timeout occurs for diagnosis.

    Args:
        model: Model name
        messages: Context messages
        tools: Tool definitions
        elapsed_time: Time elapsed before timeout
        timeout_type: Type of timeout ("inactivity" or "max_total_time")
    """
    import json
    from pathlib import Path
    from datetime import datetime

    # Create dump directory
    dump_dir = Path(".agent_context/timeout_dumps")
    dump_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_file = dump_dir / f"timeout_{timeout_type}_{timestamp}.json"

    # Calculate context stats
    total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
    estimated_tokens = total_chars // 4

    # Build dump data
    dump_data = {
        "timestamp": timestamp,
        "timeout_type": timeout_type,
        "elapsed_time_seconds": round(elapsed_time, 2),
        "model": model,
        "context_stats": {
            "message_count": len(messages),
            "total_chars": total_chars,
            "estimated_tokens": estimated_tokens,
        },
        "messages": messages,
        "tools": tools,
    }

    # Write dump file
    try:
        with open(dump_file, "w") as f:
            json.dump(dump_data, f, indent=2, default=str)
        print(f"\n[timeout_dump] Context saved to {dump_file}")
        print(f"[timeout_dump] Stats: {len(messages)} messages, ~{estimated_tokens:,} tokens, {elapsed_time:.1f}s elapsed")
    except Exception as e:
        print(f"\n[timeout_dump] Failed to save context: {e}")


def check_ollama_health(timeout: int = 5) -> bool:
    """
    Check if Ollama is responsive using stdlib only.

    Args:
        timeout: Timeout in seconds (default 5)

    Returns:
        True if Ollama is responsive, False otherwise
    """
    try:
        import urllib.request
        # Use OLLAMA_HOST environment variable, fallback to localhost
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        # Ensure URL has proper format
        if not ollama_host.startswith("http"):
            ollama_host = f"http://{ollama_host}"
        health_url = f"{ollama_host}/api/tags"
        req = urllib.request.Request(health_url)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def estimate_tokens(text: str) -> int:
    """
    Rough estimate of token count (4 chars per token heuristic).

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // 4


def clear_ollama_context(model: str, system_prompt: str):
    """
    Clear Ollama's internal state by making a minimal call with system prompt.

    This prevents state corruption where a previous task (especially one that hung)
    can contaminate subsequent tasks. By sending a fresh system prompt, we reset
    Ollama's KV cache and internal reasoning state.

    This should be called:
    - After agent completes a task (success or failure)
    - Before handing off between agents
    - After timeouts or errors that may have left model in bad state

    Args:
        model: Ollama model name to clear
        system_prompt: System prompt to load (resets context)
    """
    try:
        # Make a minimal call to reset model state
        # Use non-streaming to ensure it completes
        OLLAMA_CLIENT.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Acknowledged. Ready for next task."}
            ],
            options={"temperature": 0.2},
            stream=False
        )
    except Exception:
        # If clearing fails, don't crash - just continue
        # The next task will still work, just might be slower
        pass


def extract_tool_call_from_parse_error(error_msg: str) -> dict | None:
    """
    Extract tool call JSON from Ollama parsing error message.

    When the LLM generates text before JSON in a tool call, Ollama's parser
    fails and raises ResponseError with format:
        "error parsing tool call: raw='<mixed text and JSON>', err=<error>"

    This function attempts to extract valid JSON from the raw string.

    Args:
        error_msg: Error message string from ResponseError

    Returns:
        Dict with tool call structure, or None if extraction fails

    Example:
        >>> error = "error parsing tool call: raw='Text here.{\"name\":\"test\",\"args\":{\"x\":1}}', err=..."
        >>> result = extract_tool_call_from_parse_error(error)
        >>> result
        {'name': 'test', 'arguments': {'x': 1}}
    """
    import re
    import json

    # Extract the raw string from error message
    # Format: "error parsing tool call: raw='...', err=..."
    match = re.search(r"raw='(.*?)'(?:,\s*err=|$)", error_msg, re.DOTALL)
    if not match:
        return None

    raw = match.group(1)

    # Strategy 1: Try to find JSON object {...}
    # Use non-greedy matching to get the first complete JSON object
    json_match = re.search(r'\{(?:[^{}]|\{[^{}]*\})*\}', raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            # Check if this looks like a tool call
            # Should have either {'name': ..., 'arguments': ...} format
            # OR just be the arguments dict directly
            if isinstance(parsed, dict):
                # If it has 'name' and 'arguments', it's already a function call
                if 'name' in parsed and 'arguments' in parsed:
                    return parsed
                # Otherwise, treat it as arguments for an unknown function
                # Return in Ollama's expected format
                return parsed  # Let caller decide how to use it
        except json.JSONDecodeError:
            pass

    # Strategy 2: Try array [...] (less common)
    json_match = re.search(r'\[(?:[^\[\]]|\[[^\[\]]*\])*\]', raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed[0]  # Return first element
        except json.JSONDecodeError:
            pass

    return None
