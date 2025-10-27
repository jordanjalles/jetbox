import importlib.util, os, sys

# Path to the actual dsl package inside the subfolder
subfolder = os.path.join(os.path.dirname(__file__), '.agent_workspace/design-a-simple-domain-specific-language-dsl-for-m/dsl')
# Load the __init__.py of the actual dsl package
spec = importlib.util.spec_from_file_location('dsl', os.path.join(subfolder, '__init__.py'))
if spec is None or spec.loader is None:
    raise ImportError('Could not load the actual dsl package')
dsl_mod = importlib.util.module_from_spec(spec)
# Mark as package and set package name
dsl_mod.__path__ = [subfolder]
dsl_mod.__package__ = 'dsl'
# Register in sys.modules before executing
sys.modules['dsl'] = dsl_mod
spec.loader.exec_module(dsl_mod)

# Expose evaluate function
evaluate = dsl_mod.evaluate

__all__ = ['evaluate']
