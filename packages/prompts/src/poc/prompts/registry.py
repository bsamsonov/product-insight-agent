from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class PromptTemplate:
    def __init__(
        self, name: str, version: str, description: str, system: str, user_template: str
    ) -> None:
        self.name = name
        self.version = version
        self.description = description
        self.system = system.strip()
        self._user_template = user_template.strip()

    def render_user(self, **kwargs: Any) -> str:
        return self._user_template.format(**kwargs)


class PromptRegistry:
    """Loads prompt templates from YAML files."""

    def __init__(self, data_dir: Path | None = None) -> None:
        if data_dir is None:
            # Locate the data/ directory relative to this package
            pkg_root = Path(__file__).parent.parent.parent.parent
            data_dir = pkg_root / "data"
        self._data_dir = data_dir
        self._cache: dict[str, dict[str, PromptTemplate]] = {}

    def get(self, prompt_file: str, prompt_name: str) -> PromptTemplate:
        """Get a prompt template by file and name.

        Args:
            prompt_file: YAML file name without extension (e.g. 'rag_baseline')
            prompt_name: Key inside the file's 'prompts' dict
        """
        if prompt_file not in self._cache:
            self._cache[prompt_file] = self._load(prompt_file)
        templates = self._cache[prompt_file]
        if prompt_name not in templates:
            raise KeyError(f"Prompt '{prompt_name}' not found in '{prompt_file}.yaml'")
        return templates[prompt_name]

    def _load(self, prompt_file: str) -> dict[str, PromptTemplate]:
        path = self._data_dir / f"{prompt_file}.yaml"
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        result: dict[str, PromptTemplate] = {}
        for name, cfg in data.get("prompts", {}).items():
            result[name] = PromptTemplate(
                name=name,
                version=cfg.get("version", "1.0"),
                description=cfg.get("description", ""),
                system=cfg.get("system", ""),
                user_template=cfg.get("user_template", ""),
            )
        return result


# Module-level default registry
_default_registry: PromptRegistry | None = None


def get_registry() -> PromptRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = PromptRegistry()
    return _default_registry
