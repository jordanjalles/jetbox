import pytest
import sys
from pathlib import Path

# Add workspace root to Python path so tests can import utils and other modules
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))

def pytest_collection_modifyitems(config, items):
    # Note: Skip marker disabled to allow validator tests to run
    # Re-enable if needed for specific test environments
    pass
