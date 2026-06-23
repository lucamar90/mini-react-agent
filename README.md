# mini-react-agent

> A tiny, readable **ReAct agent** — think → act → observe — with pluggable tools. Runs fully offline, no API key required.

[![CI](https://github.com/lucamar90/mini-react-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/lucamar90/mini-react-agent/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**A mini-project by [Innova Web Design Studio](https://www.innovadesignstudio.it)** · _Creating Digital Experiences_

An LLM agent is just a loop: the model **thinks**, picks a **tool**, sees the **observation**, and repeats until it has an answer. This is that loop, written small and clearly enough to read in one sitting — the [ReAct](https://arxiv.org/abs/2210.03629) pattern in ~200 lines, with **zero runtime dependencies** in the core.

It ships with a deterministic **replay backend** so you can watch a full agent run offline, no API key — then swap in OpenAI or Anthropic to have it reason for real.

---

## Quickstart (offline, no API key)

```bash
git clone https://github.com/lucamar90/mini-react-agent.git
cd mini-react-agent
pip install -e .          # no runtime dependencies

mini-agent demo
```

Output — the agent's full reasoning trace:

```text
  Task: Trova la capitale d'Italia, calcola 144 / 12 e conta le parole in 'Creating Digital Experiences'.
  ────────────────────────────────────────────────────────
  1. Thought: Prima cerco la capitale d'Italia.
     Action: search("capitale d'Italia")
     Observation: Roma è la capitale d'Italia.
  2. Thought: Ora calcolo 144 diviso 12.
     Action: calculator('144 / 12')
     Observation: 12
  3. Thought: Infine conto le parole nella frase.
     Action: word_count('Creating Digital Experiences')
     Observation: 3
  4. Thought: Ho tutti e tre i risultati.
  ────────────────────────────────────────────────────────
  Final Answer: La capitale d'Italia è Roma, 144 / 12 = 12, e 'Creating Digital Experiences' ha 3 parole.
```

The tool calls and observations are **executed for real** — only the model's "thinking" is replayed from a recorded script, so the run is deterministic and key-free.

> No install needed either: `python -m mini_agent demo`.

---

## What it does

- **The ReAct loop, in full** — prompt construction, output parsing, tool dispatch, observation feedback, and stopping conditions (final answer **or** a `max_steps` budget).
- **Pluggable tools** — `calculator` (safe AST-based eval), `search` (offline knowledge base), `word_count`. Add one by subclassing `Tool`.
- **Swappable backends** — `replay` (offline, deterministic), `openai`, `anthropic`, behind a one-method `complete()` interface.
- **Correct ReAct mechanics** — real backends use a `stop` sequence on `Observation:` so the model never hallucinates its own tool results.
- **Robust by design** — unknown tools and malformed output are fed back to the model as observations so it can self-correct, instead of crashing.
- **Readable trace** — every Thought / Action / Observation is printed, so you can see exactly *why* the agent did what it did.
- **Live web UI** — `mini-agent serve` streams the agent's reasoning into the browser, step by step.

---

## Run it for real

```bash
pip install -e ".[openai]"        # or ".[anthropic]"
export OPENAI_API_KEY=sk-...

mini-agent run "If a train travels 240 km in 3 hours, what is its speed in km/h?" --llm openai
mini-agent run "Cosa fa Innova, e quanto fa 18 × 24?" --llm anthropic
```

List the tools the agent can use:

```bash
mini-agent tools
```

---

## Live web UI

Watch the agent reason in your browser, with each step streaming in live:

```bash
pip install -e ".[web]"
mini-agent serve            # → http://127.0.0.1:5001
```

Pick the offline demo or a live model, type a task, and watch **Thought → Action → Observation** appear one step at a time (over Server-Sent Events). The interface carries the Innova Web Design Studio brand.

---

## How it works

```
mini_agent/
├── agent.py     # the ReAct loop: think -> act -> observe -> repeat
├── tools.py     # Tool base class + calculator / search / word_count
├── llms.py      # replay / openai / anthropic backends (one complete() each)
├── prompt.py    # ReAct prompt template + output parser
├── trace.py     # Step / AgentResult + the pretty-printed trace
├── web.py       # optional Flask + SSE web UI (`mini-agent serve`)
└── cli.py       # `mini-agent demo | run | tools | serve`
```

One reasoning step looks like this:

```python
from mini_agent import Agent
from mini_agent.llms import get_llm
from mini_agent.tools import default_tools

agent = Agent(get_llm("openai"), default_tools(), max_steps=6)
result = agent.run("What is 12 * 12, and what's the capital of France?")

print(result.final_answer)
print(result.stopped_reason)   # "final_answer" | "max_steps"
```

### Adding a tool

A tool is a class with a `name`, a `description` (the model reads it to decide when to use it), and a `run()` method:

```python
from mini_agent.tools import Tool, ToolResult

class ReverseTool(Tool):
    name = "reverse"
    description = "Reverse the given text."

    def run(self, tool_input: str) -> ToolResult:
        return ToolResult(ok=True, output=tool_input[::-1])
```

Drop it into `default_tools()` and the agent can call it.

---

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

The test suite covers the tools, the output parser, and the full agent loop (happy path, the `max_steps` cutoff, and unknown-tool handling) — all driven by the deterministic replay backend, so it runs in CI without any API key.

---

## Roadmap

- [x] Live web UI with streaming steps (`mini-agent serve`)
- [ ] More tools (HTTP fetch, Python REPL sandbox, real web search)
- [ ] Parallel / multi-tool actions per step
- [ ] Token & step budgeting with cost reporting

---

## License

MIT © Luca Marullo — see [LICENSE](LICENSE).
