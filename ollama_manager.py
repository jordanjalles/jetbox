"""
Ollama Manager - Prevent stuck contexts and manage Ollama state.

This module provides utilities to:
- Detect when Ollama is busy processing requests
- Wait for Ollama to become idle
- Clear all Ollama contexts (prevent stuck state)
- Get list of loaded models

Usage:
    from ollama_manager import OllamaManager

    # Before starting tests
    if OllamaManager.is_ollama_busy():
        OllamaManager.wait_for_ollama(timeout=60)

    # Clear all contexts
    OllamaManager.clear_all_contexts()
"""
import subprocess
import time
from llm_utils import clear_ollama_context


class OllamaManager:
    """Manage Ollama state and prevent stuck contexts."""

    @staticmethod
    def clear_all_contexts(model: str = None):
        """
        Clear all Ollama contexts for the specified model (or all models).

        Should be called:
        - Before starting new test suite
        - After abnormal termination
        - When Claude restarts

        Args:
            model: Specific model to clear (if None, clears all loaded models)
        """
        if model:
            models = [model]
        else:
            # Get all loaded models
            models = OllamaManager.get_loaded_models()

        if not models:
            print("[ollama_manager] No models loaded, nothing to clear")
            return

        print(f"[ollama_manager] Clearing contexts for {len(models)} model(s)...")
        for m in models:
            try:
                clear_ollama_context(m, "")
                print(f"[ollama_manager] ✓ Cleared context for {m}")
            except Exception as e:
                print(f"[ollama_manager] ⚠️  Failed to clear {m}: {e}")

    @staticmethod
    def get_loaded_models():
        """
        Get list of currently loaded Ollama models.

        Returns:
            list: List of model names currently loaded in Ollama
        """
        try:
            result = subprocess.run(
                ["ollama", "ps"],
                capture_output=True,
                text=True,
                timeout=5
            )

            # Parse output (format: NAME  ID  SIZE  PROCESSOR  UNTIL)
            lines = result.stdout.strip().split('\n')
            if len(lines) <= 1:  # Just header or empty
                return []

            models = []
            for line in lines[1:]:  # Skip header
                parts = line.split()
                if parts:
                    models.append(parts[0])  # First column is model name

            return models
        except subprocess.TimeoutExpired:
            print("[ollama_manager] ⚠️  Timeout getting loaded models")
            return []
        except FileNotFoundError:
            print("[ollama_manager] ⚠️  Ollama command not found")
            return []
        except Exception as e:
            print(f"[ollama_manager] ⚠️  Failed to get loaded models: {e}")
            return []

    @staticmethod
    def is_ollama_busy():
        """
        Check if Ollama is currently processing requests.

        Returns:
            bool: True if Ollama appears busy (should wait before starting new tests)
        """
        try:
            result = subprocess.run(
                ["ollama", "ps"],
                capture_output=True,
                text=True,
                timeout=5
            )

            # Parse output to see how many models are loaded
            lines = result.stdout.strip().split('\n')

            # More than 2 lines (header + 1 model) = probably busy
            # This is a heuristic - multiple loaded models may indicate ongoing work
            num_models = len(lines) - 1  # Subtract header
            return num_models > 1
        except subprocess.TimeoutExpired:
            print("[ollama_manager] ⚠️  Timeout checking Ollama status")
            return True  # Assume busy if can't check
        except FileNotFoundError:
            print("[ollama_manager] ⚠️  Ollama command not found")
            return False  # Can't be busy if not installed
        except Exception as e:
            print(f"[ollama_manager] ⚠️  Failed to check Ollama status: {e}")
            return False  # Assume not busy if can't determine

    @staticmethod
    def wait_for_ollama(timeout: int = 60):
        """
        Wait for Ollama to become idle.

        Args:
            timeout: Max seconds to wait

        Returns:
            bool: True if Ollama became idle, False if timeout
        """
        print("[ollama_manager] Waiting for Ollama to become idle...")
        start = time.time()

        while time.time() - start < timeout:
            if not OllamaManager.is_ollama_busy():
                elapsed = int(time.time() - start)
                print(f"[ollama_manager] ✓ Ollama is idle (waited {elapsed}s)")
                return True

            elapsed = int(time.time() - start)
            print(f"[ollama_manager] Ollama busy, waiting... ({elapsed}s elapsed)")
            time.sleep(5)

        print(f"[ollama_manager] ⚠️  Timeout waiting for Ollama (>{timeout}s)")
        return False

    @staticmethod
    def get_ollama_status():
        """
        Get detailed status of Ollama.

        Returns:
            dict: Status information including loaded models, busy state, etc.
        """
        models = OllamaManager.get_loaded_models()
        busy = OllamaManager.is_ollama_busy()

        return {
            "loaded_models": models,
            "num_models": len(models),
            "is_busy": busy,
            "status": "busy" if busy else "idle"
        }

    @staticmethod
    def print_status():
        """Print human-readable Ollama status."""
        status = OllamaManager.get_ollama_status()

        print("="*70)
        print("OLLAMA STATUS")
        print("="*70)
        print(f"Status: {status['status'].upper()}")
        print(f"Loaded models: {status['num_models']}")
        if status['loaded_models']:
            for model in status['loaded_models']:
                print(f"  - {model}")
        else:
            print("  (none)")
        print("="*70)


if __name__ == "__main__":
    # When run directly, print status
    OllamaManager.print_status()
