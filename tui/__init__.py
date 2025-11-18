"""
Jetbox TUI (Terminal User Interface) System

This module provides a centralized, pluggable display system for agent output.

Architecture:
- DisplayInterface: Abstract base class
- PlainDisplay: Traditional print() output (default fallback)
- TextualDisplay: Interactive TUI dashboard (main implementation)
- DisplayFactory: Auto-detects best display mode

To disable TUI: Set JETBOX_TUI=plain or pass --no-tui flag
To force TUI: Set JETBOX_TUI=textual or pass --tui flag
"""

from .display_interface import DisplayInterface, AgentEvent, EventType
from .plain_display import PlainDisplay
from .textual_display import TextualDisplay
from .display_factory import DisplayFactory

__all__ = [
    "DisplayInterface",
    "AgentEvent",
    "EventType",
    "PlainDisplay",
    "TextualDisplay",
    "DisplayFactory",
]
