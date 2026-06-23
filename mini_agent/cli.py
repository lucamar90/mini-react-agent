"""Command-line interface.

Examples::

    mini-agent demo                       # offline, deterministic, no API key
    mini-agent run "What is 144 / 12?" --llm openai
    mini-agent run "..." --llm replay --script my_script.json
    mini-agent tools
"""

from __future__ import annotations

import argparse
import sys

from .agent import Agent
from .demo import load_demo, load_script
from .llms import get_llm
from .tools import default_tools
from .trace import render_trace


def _use_color(args: argparse.Namespace) -> bool:
    return sys.stdout.isatty() and not args.no_color


def _run_agent(task: str, llm, max_steps: int, use_color: bool) -> int:
    agent = Agent(llm, default_tools(), max_steps=max_steps)
    result = agent.run(task)
    print(render_trace(result, use_color=use_color))
    return 0 if result.succeeded else 1


def _cmd_demo(args: argparse.Namespace) -> int:
    task, outputs = load_demo()
    llm = get_llm("replay", outputs=outputs)
    return _run_agent(task, llm, max_steps=args.max_steps, use_color=_use_color(args))


def _cmd_run(args: argparse.Namespace) -> int:
    task = args.task
    if args.llm == "replay":
        if not args.script:
            print("error: --llm replay requires --script PATH (or use `mini-agent demo`).")
            return 2
        script_task, outputs = load_script(args.script)
        task = task or script_task
        llm = get_llm("replay", outputs=outputs)
    else:
        llm = get_llm(args.llm, model=args.model)
    return _run_agent(task, llm, max_steps=args.max_steps, use_color=_use_color(args))


def _cmd_tools(args: argparse.Namespace) -> int:
    print("Available tools:")
    for tool in default_tools().values():
        print(f"  - {tool.name}: {tool.description}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        from .web import run_server
    except ImportError:
        print("The web UI needs Flask. Install it with: pip install 'mini-react-agent[web]'")
        return 2
    print(f"mini-agent web UI → http://{args.host}:{args.port}  (Ctrl+C to stop)")
    run_server(host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-agent",
        description="A tiny ReAct agent: think -> act -> observe, with pluggable tools.",
    )
    # Shared flag so `--no-color` works on any subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-color", action="store_true", help="Disable ANSI colours.")

    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser(
        "demo", parents=[common], help="Run the bundled offline demo (no API key needed)."
    )
    demo.add_argument("--max-steps", type=int, default=6)
    demo.set_defaults(func=_cmd_demo)

    run = sub.add_parser("run", parents=[common], help="Run the agent on your own task.")
    run.add_argument("task", nargs="?", default="", help="The task/question for the agent.")
    run.add_argument(
        "--llm", default="openai", help="Backend: replay | openai | anthropic (default: openai)."
    )
    run.add_argument("--model", default=None, help="Override the backend's default model.")
    run.add_argument("--script", default=None, help="Replay script JSON (for --llm replay).")
    run.add_argument("--max-steps", type=int, default=6, help="Max reasoning steps (default: 6).")
    run.set_defaults(func=_cmd_run)

    tools = sub.add_parser("tools", help="List the available tools.")
    tools.set_defaults(func=_cmd_tools)

    serve = sub.add_parser("serve", help="Launch the local web UI (needs the 'web' extra).")
    serve.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1).")
    serve.add_argument("--port", type=int, default=5001, help="Port (default: 5001).")
    serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
