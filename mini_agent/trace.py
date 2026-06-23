"""Data structures and rendering for an agent run's reasoning trace."""

from __future__ import annotations

from dataclasses import dataclass, field

# Minimal ANSI palette (no external dependency).
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


@dataclass
class Step:
    """One Thought → Action → Observation cycle (Action/Observation empty if final)."""

    thought: str
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None


@dataclass
class AgentResult:
    task: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str | None = None
    # One of: "final_answer", "max_steps", "error".
    stopped_reason: str = "final_answer"

    @property
    def succeeded(self) -> bool:
        return self.stopped_reason == "final_answer" and self.final_answer is not None


def _c(text: str, color: str, use_color: bool) -> str:
    return f"{color}{text}{RESET}" if use_color else text


def render_trace(result: AgentResult, use_color: bool = True) -> str:
    """Pretty-print the full reasoning trace for the terminal."""
    lines = [
        "",
        _c("  Task: ", BOLD, use_color) + result.task,
        "  " + "─" * 56,
    ]
    for i, step in enumerate(result.steps, 1):
        if step.thought:
            lines.append(_c(f"  {i}. Thought: ", CYAN, use_color) + step.thought)
        if step.action:
            lines.append(
                _c("     Action: ", YELLOW, use_color)
                + f"{step.action}({step.action_input!r})"
            )
            lines.append(_c("     Observation: ", DIM, use_color) + (step.observation or ""))
    lines.append("  " + "─" * 56)
    if result.final_answer is not None:
        lines.append(_c("  Final Answer: ", GREEN, use_color) + result.final_answer)
    else:
        lines.append(
            _c(f"  Stopped: {result.stopped_reason} (no final answer)", YELLOW, use_color)
        )
    lines.append("")
    return "\n".join(lines)
