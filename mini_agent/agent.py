"""The ReAct agent loop: think → act → observe, until a final answer."""

from __future__ import annotations

from .llms import LLM
from .prompt import Action, FinalAnswer, build_prompt, extract_thought, parse
from .tools import Tool
from .trace import AgentResult, Step


class Agent:
    """Drives a language model through a tool-using reasoning loop."""

    def __init__(
        self,
        llm: LLM,
        tools: dict[str, Tool],
        max_steps: int = 6,
        on_step=None,
    ):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        # Optional callback(step_index, Step) for live progress.
        self.on_step = on_step

    def run(self, task: str) -> AgentResult:
        scratchpad = ""
        result = AgentResult(task=task)

        for _ in range(self.max_steps):
            output = self.llm.complete(build_prompt(task, self.tools.values(), scratchpad))
            thought = extract_thought(output)

            try:
                parsed = parse(output)
            except ValueError as exc:
                # Feed the error back so a real model can self-correct.
                observation = f"Error: {exc}"
                step = Step(thought=thought, action=None, action_input=None, observation=observation)
                result.steps.append(step)
                self._emit(len(result.steps), step)
                scratchpad += f"\nObservation: {observation}\n"
                continue

            if isinstance(parsed, FinalAnswer):
                step = Step(thought=thought)
                result.steps.append(step)
                self._emit(len(result.steps), step)
                result.final_answer = parsed.text
                result.stopped_reason = "final_answer"
                return result

            observation = self._run_tool(parsed)
            step = Step(
                thought=thought,
                action=parsed.tool,
                action_input=parsed.tool_input,
                observation=observation,
            )
            result.steps.append(step)
            self._emit(len(result.steps), step)
            scratchpad += (
                f"\nThought: {thought}\n"
                f"Action: {parsed.tool}\n"
                f"Action Input: {parsed.tool_input}\n"
                f"Observation: {observation}\n"
            )

        result.stopped_reason = "max_steps"
        return result

    def _run_tool(self, action: Action) -> str:
        tool = self.tools.get(action.tool)
        if tool is None:
            return (
                f"Error: unknown tool '{action.tool}'. "
                f"Available tools: {', '.join(self.tools)}."
            )
        return tool.run(action.tool_input).output

    def _emit(self, index: int, step: Step) -> None:
        if self.on_step is not None:
            self.on_step(index, step)
