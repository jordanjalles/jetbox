"""Prompt loader for agent - loads prompts from prompts.yaml"""
from pathlib import Path
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

class PromptLoader:
    def __init__(self, prompts_file="prompts.yaml"):
        self.prompts = {}
        if Path(prompts_file).exists() and YAML_AVAILABLE:
            try:
                with open(prompts_file) as f:
                    self.prompts = yaml.safe_load(f) or {}
            except:
                pass

    def get(self, name, **kwargs):
        """Get prompt by name and format with kwargs"""
        template = self.prompts.get(name, "")
        if kwargs:
            return template.format(**kwargs)
        return template

# Global instance
prompts = PromptLoader()
