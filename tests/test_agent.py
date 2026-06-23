"""Integration tests for the agent loop, driven by the deterministic ReplayLLM."""

from mini_agent.agent import Agent
from mini_agent.demo import load_demo
from mini_agent.llms import ReplayLLM
from mini_agent.tools import default_tools


def _agent(outputs, max_steps=6):
    return Agent(ReplayLLM(outputs), default_tools(), max_steps=max_steps)


def test_demo_run_uses_every_tool_and_finishes():
    task, outputs = load_demo()
    result = _agent(outputs).run(task)

    assert result.succeeded
    assert result.stopped_reason == "final_answer"

    actions = [s.action for s in result.steps if s.action]
    assert actions == ["search", "calculator", "word_count"]

    observations = [s.observation for s in result.steps if s.observation]
    assert any("Tokyo" in o for o in observations)
    assert "12" in observations
    assert "3" in observations
    assert "Tokyo" in result.final_answer


def test_stops_at_max_steps_without_final_answer():
    # Two tool actions, never a Final Answer, with max_steps=2.
    outputs = [
        "Action: calculator\nAction Input: 1 + 1",
        "Action: calculator\nAction Input: 2 + 2",
    ]
    result = _agent(outputs, max_steps=2).run("loop forever")
    assert result.stopped_reason == "max_steps"
    assert result.final_answer is None
    assert len(result.steps) == 2


def test_unknown_tool_is_reported_as_observation():
    outputs = [
        "Action: teleporter\nAction Input: anywhere",
        "Final Answer: gave up on the teleporter",
    ]
    result = _agent(outputs).run("use a missing tool")
    assert "unknown tool" in result.steps[0].observation
    assert result.succeeded
