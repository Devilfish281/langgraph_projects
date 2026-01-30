# -*- coding: utf-8 -*-
# langgraph_projects/src/langgraph_projects/level-1/agent6.py
"""
## Goals

Now, we can extend this into a generic agent architecture.

In the above router, we invoked the model and, if it chose to call a tool, we returned a `ToolMessage` to the user.

But, what if we simply pass that `ToolMessage` *back to the model*?

We can let it either (1) call another tool or (2) respond directly.

This is the intuition behind [ReAct](https://react-lm.github.io/), a general agent architecture.

* `act` - let the model call specific tools
* `observe` - pass the tool output back to the model
* `reason` - let the model reason about the tool output to decide what to do next (e.g., call another tool or just respond directly)

This [general purpose architecture](https://blog.langchain.com/planning-for-agents/) can be applied to many types of tools.
"""

"""


###  ReAct: A General Agent Architecture

This is the core intuition behind ReAct, a foundational architecture for intelligent agents. It operates on a simple yet powerful loop:

* `act` — Allow the model to invoke specific tools.
* `observe` — Capture and return the tool's output back to the model.
* `reason` — Let the model analyze the result and decide what to do next
  (e.g., invoke another tool or generate a direct response).
---

This general-purpose architecture supports a wide range of tool-use scenarios, enabling dynamic reasoning and tool orchestration.
"""

# %pip install --quiet -U langchain_openai langchain_core langgraph langgraph-prebuilt

###########################################################
## Supporting Code
###########################################################
import logging
import os
import threading
from pathlib import Path
from typing import Annotated, Literal

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from PIL import Image
from typing_extensions import TypedDict

from langgraph_projects.my_utils.load_env import load_dotenv_only, validate_environment
from langgraph_projects.my_utils.logger_setup import setup_logger

logger = logging.getLogger(__name__)  # Reuse the global logger
_MODEL = None
_LANGSMITH_ENABLED: bool = False
_LANGSMITH_INITIALIZED = False
_RUNTIME_INITIALIZED = False


_INIT_LOCK = threading.Lock()
_MODEL_LOCK = threading.Lock()

LANGCHAIN_PROJECT_NAME = "Agent Project"


#####################################################################
### START
#####################################################################
def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def init_runtime() -> None:
    """Initialize logging and environment variables at runtime.

    This function exists to avoid import-time side effects, which can break or slow down
    documentation builds (e.g., when Sphinx autodoc imports the module).

    Side Effects
    ------------
    - Configures logging via :func:`my_utils.logger_setup.setup_logger`.
    - Loads and validates environment variables via :func:`my_utils.load_env.load_environment`.

    :returns: ``None``.
    """
    global _RUNTIME_INITIALIZED
    global logger
    if _RUNTIME_INITIALIZED:
        return
    with _INIT_LOCK:
        if _RUNTIME_INITIALIZED:
            return

        load_dotenv_only()  # Changed Code
        logger = setup_logger()  # Changed Code
        validate_environment(log=logger)  # Added Code:
        logger.debug("my_langchain_logger Started!")
        logger.debug(
            f"Effective LOG_LEVEL from env: {os.getenv('LOG_LEVEL')}"
        )  # Added Code:
        logger.debug(
            f"Logger effective level: {logger.getEffectiveLevel()}"
        )  # Added Code:

        _RUNTIME_INITIALIZED = True


# You can see pricing for various models [here](https://openai.com/api/pricing/).
# We will default to `gpt-4o` because it offers a good balance of quality, price, and speed.

# There are [a few standard parameters](https://docs.langchain.com/oss/python/langchain/models#parameters) that we can set with chat models.
# Two of the most common are:
# * `model`: the name of the model
# * `temperature`: the sampling temperature


# `Temperature` controls the randomness or creativity of the model's output where low temperature (close to 0)
# is more deterministic and focused outputs. This is good for tasks requiring accuracy or factual responses.
# High temperature (close to 1) is good for creative tasks or generating varied responses.
def get_model() -> ChatOpenAI:
    """
    Create the LLM client lazily and return a cached instance.

    The model is created only once per process (thread-safe) to prevent repeated initialization
    and to avoid import-time side effects.

    Uses:
        - ``OPENAI_MODEL`` (defaults to ``gpt-4o``)

    :returns: A configured :class:`langchain_openai.ChatOpenAI` instance.
    :rtype: langchain_openai.ChatOpenAI
    """
    global _MODEL
    if not _RUNTIME_INITIALIZED:
        init_runtime()
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                _MODEL = ChatOpenAI(
                    temperature=0,
                    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                    streaming=True,
                )

    return _MODEL


from langsmith import utils


def init_langsmith() -> None:
    """
    Initialize LangSmith/LangChain tracing configuration (optional).

    This is gated by the ``LANGSMITH_ENABLED`` environment variable.

    If enabled, the following environment variables are set:
    - ``LANGCHAIN_TRACING_V2`` = ``"true"``
    - ``LANGCHAIN_PROJECT`` = ``"Router Agent Project"``

    This function is safe to call multiple times; initialization runs once.

    :returns: ``None``.
    """
    global _LANGSMITH_INITIALIZED
    global _LANGSMITH_ENABLED

    if _LANGSMITH_INITIALIZED:
        return
    _LANGSMITH_INITIALIZED = True

    _LANGSMITH_ENABLED = env_bool("LANGSMITH_ENABLED", default=False)

    if _LANGSMITH_ENABLED:
        #  LangSmith
        # Enable LangChain's advanced tracing features by setting the environment variable.
        # This activation allows for detailed tracking of operations within LangChain,
        # facilitating debugging and performance monitoring.
        # Setting this variable to "true" enables tracing; any other value will disable it.
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

        # Specify the project name for organizing traces in LangChain.
        # By defining this environment variable, all traces generated during the application's
        # execution will be associated with the "Router Agent Project".
        # This categorization aids in the systematic analysis and retrieval of trace data.
        # If the specified project does not exist, it will be created automatically upon the first trace.

        os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT_NAME

    # Check if tracing is enabled in the current environment
    # This function evaluates the current tracing status, which is determined by
    # environment variables or prior configuration within the application.
    # It returns a boolean value:
    # - True: Tracing is active.
    # - False: Tracing is inactive.
    # This check is essential for conditional tracing, allowing developers to
    # enable or disable tracing based on specific conditions or configurations.
    is_tracing_active = utils.tracing_is_enabled()
    # Print the status of tracing
    logger.debug(
        f"LangSmith Tracing is {'enabled' if is_tracing_active else 'disabled'}."
    )


def display_graph_if_enabled(
    graph,
    *,
    save_env: str = "VIEW_GRAPH",
    open_env: str = "VIEW_GRAPH_OPEN",
    default_save: bool = False,
    default_open: bool = False,
    xray: int = 1,
    images_dir_name: str = "images",
    output_name: str = "agent6_graph_image.png",
) -> None:
    if not env_bool(save_env, default=default_save):
        return

    current_dir = Path(__file__).parent
    images_directory = current_dir / images_dir_name
    images_directory.mkdir(parents=True, exist_ok=True)

    output_file_path = images_directory / output_name
    logger.info("Rendering graph PNG to: %s", output_file_path)

    png_bytes = graph.get_graph(xray=xray).draw_mermaid_png()
    output_file_path.write_bytes(png_bytes)

    # only open viewer if explicitly enabled
    if env_bool(open_env, default=default_open):
        try:
            Image.open(output_file_path).show()
        except Exception:
            logger.exception("Saved graph PNG but failed to open viewer.")


def make_state_type():
    class CustomMessagesState(MessagesState):
        # Add any keys needed beyond messages, which is pre-built
        pass

    return CustomMessagesState


# CustomMessagesState = make_state_type()
# builder = StateGraph(CustomMessagesState)
#####################################################################
### END
#####################################################################


# This will be a tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    value = a * b
    logger.info(f"Tool `multiply` called with a={a}, b={b}, result={value}")
    return value


# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    value = a + b
    logger.info(f"Tool `add` called with a={a}, b={b}, result={value}")
    return value


def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    value = a / b
    logger.info(f"Tool `divide` called with a={a}, b={b}, result={value}")
    return value


def build_app():
    init_runtime()
    init_langsmith()

    tools = [add, multiply, divide]

    # For this ipynb we set parallel tool calling to false as math generally is done sequentially, and this time we have 3 tools that can do math
    # the OpenAI model specifically defaults to parallel tool calling for efficiency, see https://python.langchain.com/docs/how_to/tool_calling_parallel/
    # play around with it and see how the model behaves with math equations!
    llm_with_tools = get_model().bind_tools(tools, parallel_tool_calls=False)

    """Let's create our LLM and prompt it with the overall desired agent behavior."""

    # System message
    sys_msg = SystemMessage(
        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
    )

    # Node
    def assistant_node(state: MessagesState):
        logger.debug("Invoking LLM with tool calling capabilities...")
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    """
    As before, we use `MessagesState` and define a `Tools` node with our list of tools.

    The `Assistant` node is just our model with bound tools.

    We create a graph with `Assistant` and `Tools` nodes.

    We add `tools_condition` edge, which routes to `End` or to `Tools` based on  whether the `Assistant` calls a tool.

    Now, we add one new step:

    We connect the `Tools` node *back* to the `Assistant`, forming a loop.

    * After the `assistant` node executes, `tools_condition` checks if the model's output is a tool call.
    * If it is a tool call, the flow is directed to the `tools` node.
    * The `tools` node connects back to `assistant`.
    * This loop continues as long as the model decides to call tools.
    * If the model response is not a tool call, the flow is directed to END, terminating the process.
    """

    # Graph
    builder = StateGraph(MessagesState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    # (NEW) Loop back from tools to assistant
    builder.add_edge("tools", "assistant")
    graph = builder.compile()

    return graph


# from langchain_core.messages import HumanMessage

graph = build_app()

# Only run demos when you execute the file directly (NOT when Studio imports it).

if __name__ == "__main__":
    # Demo / manual run (kept under __main__ so imports remain side-effect-light).
    #######################################################
    ## Display the graph for Analysts
    #######################################################
    # display_graph_if_enabled(graph)
    display_graph_if_enabled(
        graph,
        save_env="VIEW_GRAPH",
        open_env="VIEW_GRAPH_OPEN",
        default_save=True,
        default_open=True,
        xray=1,
        images_dir_name="images",
        output_name="agent6_graph_image.png",
    )

    #######################################################
    messages = [
        HumanMessage(
            content="Add 3 and 4. Multiply the output by 2. Divide the output by 5"
        )
    ]

    logger.info("Invoking model...")
    messages = graph.invoke({"messages": messages})
    logger.info("Model returned.")

    for m in messages["messages"]:
        m.pretty_print()

    """
    LangSmith

    We can look at traces in LangSmith.
    https://smith.langchain.com/
    """

    print("Program Done.")
