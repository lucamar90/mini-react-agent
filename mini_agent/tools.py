"""Tools the agent can call.

A tool has a ``name``, a one-line ``description`` (shown to the model so it
knows when to use it), and a ``run(tool_input) -> ToolResult`` method. Add your
own by subclassing :class:`Tool` and dropping it into ``default_tools()``.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass


@dataclass
class ToolResult:
    """What a tool hands back to the agent as an Observation."""

    ok: bool
    output: str


class Tool:
    """Base class for tools. Subclasses set ``name``/``description`` and ``run``."""

    name: str = "tool"
    description: str = ""

    def run(self, tool_input: str) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# calculator
# --------------------------------------------------------------------------- #

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a *safe* arithmetic AST (numbers and operators only)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("only numbers and + - * / // % ** are allowed")


def _format_number(value: float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(round(value, 6))


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate an arithmetic expression, e.g. '23 * (47 + 1)'."

    def run(self, tool_input: str) -> ToolResult:
        try:
            tree = ast.parse(tool_input, mode="eval")
            value = _eval_node(tree.body)
        except (SyntaxError, ValueError, ZeroDivisionError) as exc:
            return ToolResult(ok=False, output=f"Error: {exc}")
        return ToolResult(ok=True, output=_format_number(value))


# --------------------------------------------------------------------------- #
# search (tiny offline knowledge base — swap for a real API in production)
# --------------------------------------------------------------------------- #

_KNOWLEDGE_BASE = {
    "capital of france": "Paris is the capital of France.",
    "capital of japan": "Tokyo is the capital of Japan.",
    "capital of italy": "Rome is the capital of Italy.",
    "capital of australia": "Canberra is the capital of Australia.",
    "speed of light": "The speed of light is about 299,792 kilometres per second.",
    "python": "Python is a programming language first released in 1991 by Guido van Rossum.",
    "react agent": "ReAct is a prompting pattern interleaving Reasoning and Acting (tool use).",
}


class SearchTool(Tool):
    name = "search"
    description = "Look up a fact in a small built-in knowledge base by keyword."

    def run(self, tool_input: str) -> ToolResult:
        query = tool_input.strip().lower()
        for key, fact in _KNOWLEDGE_BASE.items():
            if key in query or query in key:
                return ToolResult(ok=True, output=fact)
        return ToolResult(ok=True, output="No results found.")


# --------------------------------------------------------------------------- #
# word_count
# --------------------------------------------------------------------------- #


class WordCountTool(Tool):
    name = "word_count"
    description = "Count the number of words in the given text."

    def run(self, tool_input: str) -> ToolResult:
        count = len(tool_input.split())
        return ToolResult(ok=True, output=str(count))


def default_tools() -> dict[str, Tool]:
    """The tool set the agent ships with, keyed by name."""
    tools = [CalculatorTool(), SearchTool(), WordCountTool()]
    return {t.name: t for t in tools}
