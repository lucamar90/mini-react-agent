"""Unit tests for the ReAct output parser."""

import pytest

from mini_agent.prompt import (
    Action,
    FinalAnswer,
    OutputParseError,
    extract_thought,
    parse,
)


def test_parse_action():
    text = "Thought: I should add.\nAction: calculator\nAction Input: 1 + 1"
    result = parse(text)
    assert isinstance(result, Action)
    assert result.tool == "calculator"
    assert result.tool_input == "1 + 1"


def test_parse_strips_decoration_around_tool_name():
    text = "Action: [calculator]\nAction Input: 2 + 2"
    assert parse(text).tool == "calculator"


def test_parse_final_answer():
    result = parse("Thought: done.\nFinal Answer: 42")
    assert isinstance(result, FinalAnswer)
    assert result.text == "42"


def test_final_answer_takes_priority_over_action():
    # If both appear, the run is finished.
    result = parse("Action: search\nFinal Answer: all done")
    assert isinstance(result, FinalAnswer)


def test_parse_raises_on_garbage():
    with pytest.raises(OutputParseError):
        parse("just some text with no action or final answer")


def test_extract_thought():
    assert extract_thought("Thought: hello\nAction: x") == "hello"
    assert extract_thought("Action: x") == ""
