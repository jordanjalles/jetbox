import importlib.util
spec = importlib.util.spec_from_file_location('test', '/workspace/.agent_workspace/design-a-simple-domain-specific-language-dsl-for-m/tests/test_dsl_complex.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('Imported successfully')
