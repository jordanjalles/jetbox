#!/usr/bin/env python3
"""Analyze and optimize context size for faster LLM processing."""
import sys
from pathlib import Path

def analyze_context_size():
    """Analyze different context strategies and their token counts."""

    # Simulated contexts
    contexts = {
        "full_history": """System: You are a coding agent...

User: Create mathx package with add and multiply