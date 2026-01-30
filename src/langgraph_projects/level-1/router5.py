# -*- coding: utf-8 -*-
# langgraph_projects/src/langgraph_projects/level-1/router5.py
"""

## Goals

We can think of this as a router, where the chat model routes between a direct response or a tool call based upon the user input.

This is a simple example of an agent, where the LLM is directing the control flow either by calling a tool or just responding directly.

![Screenshot 2024-08-21 at 9.24.09 AM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbac6543c3d4df239a4ed1_router1.png)

Let's extend our graph to work with either output!

For this, we can use two ideas:

(1) Add a node that will call our tool.

(2) Add a conditional edge that will look at the chat model output, and route to our tool calling node or simply end if no tool call is performed.
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
        os.environ["LANGCHAIN_PROJECT"] = "Router Agent Project"

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
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    value = a * b
    logger.debug(f"Tool `multiply` called with a={a}, b={b}, result={value}")
    return value


def build_app():
    init_runtime()
    init_langsmith()

    llm_with_tools = get_model().bind_tools([multiply])

    """ 
    We use the  [built-in `ToolNode`]
    (https://langchain-ai.github.io/langgraph/reference/agents/#langgraph.prebuilt.tool_node.ToolNode) 
    and simply pass a list of our tools to initialize it.

    We use the [built-in `tools_condition`]
    (https://langchain-ai.github.io/langgraph/reference/agents/#langgraph.prebuilt.tool_node.tools_condition) 
    as our conditional edge.
    """

    # Node
    def tool_calling_llm(state: MessagesState):
        logger.debug("Invoking LLM with tool calling capabilities...")
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # Build graph
    builder = StateGraph(MessagesState)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode([multiply]))
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("tools", END)
    graph = builder.compile()

    return graph


graph = build_app()

from langchain_core.messages import HumanMessage

# Only run demos when you execute the file directly (NOT when Studio imports it).
if __name__ == "__main__":
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
        output_name="router_graph_image.png",
    )

    #######################################################
    messages = [HumanMessage(content="Hello World! What is the capital of France?")]

    logger.debug("Invoking model...")
    messages = graph.invoke({"messages": messages})
    logger.debug("Model returned.")

    for m in messages["messages"]:
        m.pretty_print()
    #######################################################
    messages = [HumanMessage(content="Hello, what is 2 multiplied by 2?")]

    logger.debug("Invoking model...")
    messages = graph.invoke({"messages": messages})
    logger.debug("Model returned.")

    for m in messages["messages"]:
        m.pretty_print()
    print("Program Done.")
