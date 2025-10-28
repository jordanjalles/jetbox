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
) -> dict[str, Any]:
    """
    Call ollama chat with INACTIVITY timeout (not total time timeout).

    This allows complex tasks to take as long as needed, but detects
    when Ollama has actually stopped responding (no chunks for N seconds).

    Args:
        model: Model name
        messages: List of messages
        options: Options dict (temperature, etc)
        inactivity_timeout: Max seconds without activity (default 30s)
        tools: Optional list of tool specifications for function calling

    Returns:
        Response dict from ollama

    Raises:
        TimeoutError: If no response activity for inactivity_timeout seconds
    """
    from queue import Queue, Empty

    result_queue: Queue = Queue()

    def _stream_chat():
        try:
            # Build chat arguments
            chat_args = {
                "model": model,
                "messages": messages,
                "options": options,
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

            # Stream complete
            result_queue.put(("done", full_response))
        except Exception as e:
            result_queue.put(("error", e))

    thread = Thread(target=_stream_chat, daemon=True)
    thread.start()

    # Monitor for inactivity
    while True:
        try:
            msg_type, data = result_queue.get(timeout=inactivity_timeout)

            if msg_type == "chunk":
                # Activity detected, keep waiting
                continue
            elif msg_type == "done":
                # Success - return full response
                return data
            elif msg_type == "error":
                # Error during streaming
                raise data

        except Empty:
            # No activity for inactivity_timeout seconds = Ollama is hung
            raise TimeoutError(
                f"No response from Ollama for {inactivity_timeout}s - likely hung or dead"
            )


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
