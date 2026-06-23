"""Tests for trace rendering."""

from mini_agent.trace import AgentResult, Step, render_trace


def test_render_includes_normal_action_and_observation():
    result = AgentResult(
        task="t",
        steps=[Step(thought="thinking", action="calculator", action_input="2+2", observation="4")],
        final_answer="4",
    )
    out = render_trace(result, use_color=False)
    assert "calculator" in out and "Observation: 4" in out and "Final Answer: 4" in out


def test_render_surfaces_error_feedback_for_no_action_step():
    # A malformed model output produces a step with no action but an error
    # observation; render_trace must show it (as a Note) so recovery is visible.
    result = AgentResult(
        task="t",
        steps=[Step(thought="oops", action=None, observation="Error: bad format")],
        final_answer=None,
        stopped_reason="max_steps",
    )
    out = render_trace(result, use_color=False)
    assert "Error: bad format" in out
