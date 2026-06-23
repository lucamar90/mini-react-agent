"""Helpers for the bundled offline demo (task + recorded reasoning trace)."""

from __future__ import annotations

import json
from importlib.resources import files


def load_demo() -> tuple[str, list[str]]:
    """Return ``(task, recorded_outputs)`` for the packaged demo script."""
    raw = files("mini_agent").joinpath("data/demo_script.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    return data["task"], data["steps"]


def load_script(path: str) -> tuple[str, list[str]]:
    """Load a custom replay script ``{ "task": ..., "steps": [...] }`` from disk."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["task"], data["steps"]
