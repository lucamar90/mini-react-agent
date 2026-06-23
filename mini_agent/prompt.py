"""The ReAct prompt template and the parser for the model's output.

The agent and the model speak a small line-based protocol::

    Thought: <reasoning>
    Action: <tool name>
    Action Input: <input on a single line>
    Observation: <filled in by the runtime>
    ... (repeat) ...
    Thought: I now know the final answer
    Final Answer: <answer>
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .tools import Tool

# Models should stop generating here so the runtime can inject the real result.
STOP_SEQUENCE = "Observation:"

SYSTEM_PROMPT = (
    "You are a careful reasoning agent. Solve the user's task by thinking step "
    "by step and using the available tools. Always follow the exact format and "
    "stop after 'Action Input:' so the system can return the Observation."
)


@dataclass
class Action:
    tool: str
    tool_input: str


@dataclass
class FinalAnswer:
    text: str


class OutputParseError(ValueError):
    """Raised when the model output contains neither an Action nor a Final Answer."""


_THOUGHT_RE = re.compile(r"^\s*Thought\s*:\s*(.+?)\s*$", re.MULTILINE)
_ACTION_RE = re.compile(r"^\s*Action\s*:\s*(.+?)\s*$", re.MULTILINE)
_INPUT_RE = re.compile(r"^\s*Action Input\s*:\s*(.*)$", re.MULTILINE)
_FINAL_RE = re.compile(r"Final Answer\s*:\s*(.*)", re.DOTALL)


def build_prompt(task: str, tools: Iterable[Tool], scratchpad: str) -> str:
    """Render the full user prompt for one step of the loop."""
    tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    tool_names = ", ".join(t.name for t in tools)
    return (
        f"Answer the following question as best you can.\n\n"
        f"You have access to these tools:\n{tool_lines}\n\n"
        f"Use exactly this format:\n"
        f"Thought: your reasoning about what to do next\n"
        f"Action: the tool to use, one of [{tool_names}]\n"
        f"Action Input: the input to the tool\n"
        f"Observation: the result of the tool (the system fills this in)\n"
        f"... (this Thought/Action/Action Input/Observation can repeat) ...\n"
        f"Thought: I now know the final answer\n"
        f"Final Answer: the final answer to the original question\n\n"
        f"Question: {task}\n"
        f"{scratchpad}"
    )


def extract_thought(text: str) -> str:
    match = _THOUGHT_RE.search(text)
    return match.group(1).strip() if match else ""


def _clean_tool_name(raw: str) -> str:
    return raw.strip().strip("`").strip('"').strip("'").strip("[]").strip()


def parse(text: str) -> Action | FinalAnswer:
    """Turn one model output into an :class:`Action` or :class:`FinalAnswer`."""
    final = _FINAL_RE.search(text)
    if final:
        return FinalAnswer(final.group(1).strip())

    action = _ACTION_RE.search(text)
    if not action:
        raise OutputParseError(
            f"Expected an 'Action:' or 'Final Answer:' line, got:\n{text.strip()}"
        )

    tool = _clean_tool_name(action.group(1))
    inp_match = _INPUT_RE.search(text, action.end())
    tool_input = inp_match.group(1).strip() if inp_match else ""
    return Action(tool=tool, tool_input=tool_input)
