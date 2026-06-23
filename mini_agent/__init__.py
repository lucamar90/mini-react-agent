"""mini-react-agent: a tiny, readable ReAct agent with pluggable tools.

    >>> from mini_agent import Agent
    >>> from mini_agent.llms import ReplayLLM
    >>> from mini_agent.tools import default_tools
    >>> agent = Agent(ReplayLLM([...]), default_tools())
    >>> result = agent.run("What is 144 / 12?")
"""

from .agent import Agent
from .trace import AgentResult, Step

__version__ = "0.1.0"
__all__ = ["Agent", "AgentResult", "Step"]
