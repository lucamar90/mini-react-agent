"""Unit tests for the built-in tools."""

from mini_agent.tools import default_tools


def test_calculator_basic():
    calc = default_tools()["calculator"]
    assert calc.run("144 / 12").output == "12"
    assert calc.run("23 * (47 + 1)").output == "1104"
    assert calc.run("2 ** 10").output == "1024"


def test_calculator_accepts_unicode_symbols():
    calc = default_tools()["calculator"]
    assert calc.run("18 × 24").output == "432"
    assert calc.run("144 ÷ 12").output == "12"


def test_calculator_rejects_unsafe_input():
    calc = default_tools()["calculator"]
    result = calc.run("__import__('os').system('echo hi')")
    assert result.ok is False and result.output.startswith("Error")


def test_calculator_handles_division_by_zero():
    result = default_tools()["calculator"].run("1 / 0")
    assert result.ok is False and "Error" in result.output


def test_search_hit_and_miss():
    search = default_tools()["search"]
    assert "Tokyo" in search.run("capital of Japan").output
    assert search.run("capital of Mars").output == "No results found."


def test_search_italian_and_accent_insensitive():
    search = default_tools()["search"]
    assert "Roma" in search.run("qual è la capitale d'Italia?").output
    assert "299" in search.run("velocita della luce").output  # missing accent still matches


def test_word_count():
    wc = default_tools()["word_count"]
    assert wc.run("the quick brown fox").output == "4"
    assert wc.run("   ").output == "0"
