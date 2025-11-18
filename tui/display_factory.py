"""
Display Factory - Auto-detect and create the appropriate display.

This centralizes the logic for choosing which display to use.
"""

import os
import sys
from .display_interface import DisplayInterface
from .plain_display import PlainDisplay
from .textual_display import TextualDisplay


class DisplayFactory:
    """
    Factory for creating display instances.

    This handles auto-detection and fallback logic.
    """

    @staticmethod
    def create(
        force_mode: str | None = None,
        verbose: bool = True,
    ) -> DisplayInterface:
        """
        Create the appropriate display based on environment.

        Args:
            force_mode: If set, force specific mode:
                - "plain": Use PlainDisplay
                - "textual": Use TextualDisplay
                - None: Auto-detect
            verbose: Whether to show verbose output (for PlainDisplay)

        Returns:
            DisplayInterface instance
        """

        # 1. Check for forced mode (CLI flag or env var)
        mode = force_mode or os.environ.get("JETBOX_TUI", "auto")

        if mode == "plain":
            return PlainDisplay(verbose=verbose)
        elif mode == "textual":
            return DisplayFactory._create_textual()
        elif mode == "auto":
            return DisplayFactory._auto_detect(verbose)
        else:
            # Unknown mode - fallback to auto
            return DisplayFactory._auto_detect(verbose)

    @staticmethod
    def _auto_detect(verbose: bool) -> DisplayInterface:
        """
        Auto-detect the best display mode.

        Detection logic:
        1. Not a TTY? → Plain
        2. Terminal too small? → Plain
        3. TERM=dumb? → Plain
        4. Textual not installed? → Plain
        5. Otherwise → Textual
        """

        # Check if TTY
        if not sys.stdout.isatty():
            return PlainDisplay(verbose=verbose)

        # Check terminal size
        try:
            size = os.get_terminal_size()
            if size.columns < 80 or size.lines < 20:
                # Too small for TUI
                return PlainDisplay(verbose=verbose)
        except OSError:
            # Can't get size - probably not a real terminal
            return PlainDisplay(verbose=verbose)

        # Check TERM variable
        term = os.environ.get("TERM", "dumb")
        if term in ["dumb", "unknown"]:
            return PlainDisplay(verbose=verbose)

        # Check if Textual is available
        try:
            import textual
            return DisplayFactory._create_textual()
        except ImportError:
            # Textual not installed - fallback to plain
            return PlainDisplay(verbose=verbose)

    @staticmethod
    def _create_textual() -> TextualDisplay:
        """Create TextualDisplay with error handling."""
        try:
            return TextualDisplay()
        except Exception as e:
            # TUI failed to initialize - fallback to plain
            print(f"Warning: TUI initialization failed: {e}")
            print("Falling back to plain display")
            return PlainDisplay(verbose=True)
