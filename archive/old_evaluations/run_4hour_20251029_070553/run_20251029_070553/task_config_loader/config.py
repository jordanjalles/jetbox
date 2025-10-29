"""Configuration loader with environment variable interpolation.

This module provides a :class:`Config` class that can load configuration
files in either YAML or JSON format.  The loader supports environment
variable interpolation using the ``${VAR}`` syntax.  If an environment
variable is not defined, the placeholder is left unchanged.

Example usage::

    from config import Config
    cfg = Config("settings.yaml")
    print(cfg.get("database.host"))

The configuration is stored internally as a nested dictionary.  Keys can
be accessed using dot notation via :meth:`get`.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


class Config:
    """Load and access configuration data.

    Parameters
    ----------
    path:
        Path to a YAML or JSON file.  The file extension determines the
        parser used.  ``.yaml``/``.yml`` uses :mod:`yaml` if available,
        otherwise :mod:`json` is used.
    env_prefix:
        Optional prefix to add to all environment variable names when
        performing interpolation.  Useful when you want to namespace
        variables.
    """

    def __init__(self, path: str | Path, env_prefix: str | None = None):
        self.path = Path(path)
        self.env_prefix = env_prefix or ""
        self._data: Dict[str, Any] = {}
        self.load()

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Read the configuration file and store the data.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        ValueError
            If the file format is unsupported or parsing fails.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.path}")

        text = self.path.read_text(encoding="utf-8")
        # Perform environment interpolation before parsing
        text = self._interpolate_env(text)

        if self.path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise ValueError("PyYAML is required to parse YAML files")
            try:
                self._data = yaml.safe_load(text) or {}
            except Exception as exc:  # pragma: no cover - parsing error
                raise ValueError(f"Failed to parse YAML: {exc}") from exc
        elif self.path.suffix.lower() == ".json":
            try:
                self._data = json.loads(text)
            except Exception as exc:  # pragma: no cover - parsing error
                raise ValueError(f"Failed to parse JSON: {exc}") from exc
        else:
            raise ValueError(
                f"Unsupported configuration file type: {self.path.suffix}"
            )

    def _interpolate_env(self, text: str) -> str:
        """Replace ``${VAR}`` placeholders with environment values.

        If a variable is not defined, the placeholder is left unchanged.
        """

        def repl(match: re.Match[str]) -> str:
            var_name = match.group(1)
            full_name = f"{self.env_prefix}{var_name}" if self.env_prefix else var_name
            return os.getenv(full_name, match.group(0))

        return _ENV_VAR_PATTERN.sub(repl, text)

    # ------------------------------------------------------------------
    # Data access helpers
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any | None = None) -> Any:
        """Retrieve a value using dot‑notation.

        Parameters
        ----------
        key:
            Dot‑separated key path, e.g. ``"database.host"``.
        default:
            Value to return if the key is missing.
        """
        parts: Iterable[str] = key.split(".")
        current: Any = self._data
        for part in parts:
            if isinstance(current, Mapping) and part in current:
                current = current[part]
            else:
                return default
        return current

    def __getitem__(self, key: str) -> Any:  # pragma: no cover - thin wrapper
        return self.get(key)

    def __contains__(self, key: str) -> bool:  # pragma: no cover - thin wrapper
        return self.get(key, None) is not None

    def as_dict(self) -> Dict[str, Any]:
        """Return the underlying configuration dictionary."""
        return self._data

# ----------------------------------------------------------------------
# Convenience function for loading a config from a path
# ----------------------------------------------------------------------

def load_config(path: str | Path, env_prefix: str | None = None) -> Config:
    """Return a :class:`Config` instance for *path*.

    This helper is useful when you want a quick one‑liner without
    explicitly creating the :class:`Config` object.
    """
    return Config(path, env_prefix=env_prefix)

# ----------------------------------------------------------------------
# Example usage (uncomment to test manually)
# ----------------------------------------------------------------------
# if __name__ == "__main__":
#     cfg = Config("config.yaml")
#     print(cfg.as_dict())
#     print(cfg.get("some.key"))
"""
