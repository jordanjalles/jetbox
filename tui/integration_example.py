"""
Example: How to integrate TUI into base_agent.py

This shows the MINIMAL changes needed to base_agent.py to use the TUI system.

BEFORE (old code with print):
    print(f"[{self.name}] Round {round}/{max_rounds}")
    print(f"Tool: {tool_name}({args})")
    print(f"✓ Success")

AFTER (new code with TUI):
    self.display.update_status(round=round, max_rounds=max_rounds, ...)
    self.display.log_event(AgentEvent(type=EventType.TOOL_CALL, ...))
    self.display.log_event(AgentEvent(type=EventType.SUCCESS, ...))
"""

from tui import DisplayFactory, AgentEvent, EventType
import time


class BaseAgentWithTUI:
    """
    Example: BaseAgent modified to use TUI system.

    This shows the integration pattern.
    """

    def __init__(self, goal: str, display_mode: str = "auto"):
        """
        Initialize agent with TUI.

        Args:
            goal: Agent goal
            display_mode: "auto", "plain", or "textual"
        """
        self.goal = goal
        self.name = "task_executor"
        self.model = "qwen3:14b"
        self.current_round = 0
        self.max_rounds = 50
        self.start_time = time.time()

        # CREATE DISPLAY (single line change)
        self.display = DisplayFactory.create(force_mode=display_mode)

    def run(self):
        """
        Main agent loop with TUI integration.

        Changes from old code:
        1. display.start() at beginning
        2. Replace all print() with display.log_event()
        3. display.update_status() instead of status print()
        4. display.stop() at end
        """

        # START DISPLAY
        self.display.start()

        try:
            # Main loop
            while self.current_round < self.max_rounds:
                self.current_round += 1

                # UPDATE STATUS (replaces print(f"Round {round}"))
                self.display.update_status(
                    goal=self.goal,
                    agent_name=self.name,
                    model=self.model,
                    current_round=self.current_round,
                    max_rounds=self.max_rounds,
                    elapsed_time=time.time() - self.start_time,
                    status="Running",
                    tokens_used=self.current_round * 500,  # Example
                    tokens_max=128000,
                )

                # SIMULATE TOOL CALL
                tool_name = "write_file"
                file_path = f"file_{self.current_round}.py"

                # LOG TOOL CALL (replaces print(f"Tool: {tool_name}"))
                self.display.log_event(AgentEvent(
                    type=EventType.TOOL_CALL,
                    message=f"{tool_name}(path={file_path})",
                ))

                time.sleep(0.5)  # Simulate work

                # LOG SUCCESS (replaces print("✓ Created file"))
                self.display.log_event(AgentEvent(
                    type=EventType.SUCCESS,
                    message=f"Created {file_path}",
                    details={"size": "245 bytes"},
                ))

                # CHECK PAUSE (TUI-only feature)
                if self.display.can_pause():
                    self.display.wait_if_paused()

                # Simulate completion
                if self.current_round >= 10:
                    break

            # SHOW COMPLETION (replaces final print block)
            self.display.show_completion(
                success=True,
                summary=f"Completed {self.goal}",
                duration=time.time() - self.start_time,
                files_created=[f"file_{i}.py" for i in range(1, 11)],
            )

        finally:
            # STOP DISPLAY (cleanup)
            self.display.stop()


# Example usage
if __name__ == "__main__":
    import sys

    # Parse command line
    mode = "auto"
    if "--no-tui" in sys.argv:
        mode = "plain"
    elif "--tui" in sys.argv:
        mode = "textual"

    # Run agent
    agent = BaseAgentWithTUI(
        goal="Create calculator with tests",
        display_mode=mode,
    )
    agent.run()
