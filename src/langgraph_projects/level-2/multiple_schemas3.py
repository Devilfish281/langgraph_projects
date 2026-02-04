# src/langgraph_projects/level-2/multiple_schemas3.py

"""
multiple-schemas.ipynb
## Goals
But, there are cases where I may want a bit more control over this:
* Internal nodes may pass information that is *not required* in the graph's input / output.
* I may also want to use different input / output schemas for the graph. The output might, for example, only contain a single relevant output key.
I'll show a few ways to customize graphs with multiple schemas.
"""

# %pip install --quiet -U langgraph
###########################################################
## Supporting Code
###########################################################
import json
import logging
import os
import threading
from pathlib import Path
from typing import Annotated, Literal, NotRequired, TypedDict

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
# We will default to `gpt-5.1` because it offers a good balance of quality, price, and speed.

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
        - ``OPENAI_MODEL`` (defaults to ``gpt-5.1``)

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
                    model=os.getenv("OPENAI_MODEL", "gpt-5.1"),
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


######################################################################
### Custom State Example
# EXAMPLE USING :
# CustomStateMessages = make_state_for_graph()
# builder = StateGraph(CustomStateMessages)
######################################################################
def make_state_for_graph():
    class CustomStateMessages(MessagesState):
        # Add any keys needed beyond messages, which is pre-built
        pass

    return CustomStateMessages


#####################################################################
### END
#####################################################################


##############################################
# Private State
##############################################
def private_state_example():
    """
    ## Private State
    First, let's cover the case of passing [private state](https://docs.langchain.com/oss/python/langgraph/use-graph-api#pass-private-state-between-nodes) between nodes.
    This is useful for anything needed as part of the intermediate working logic of the graph, but not relevant for the overall graph input or output.
    I'll define an `OverallState` and a `PrivateState`.
    `node_2` uses `PrivateState` as input, but writes out to `OverallState`.
    """

    # from typing_extensions import TypedDict
    # from langgraph.graph import StateGraph, START, END

    class OverallState(TypedDict):
        foo: int

    class PrivateState(TypedDict):
        baz: int

    def node_1(state: OverallState) -> PrivateState:
        print("---Node 1---")
        return {"baz": state["foo"] + 1}

    def node_2(state: PrivateState) -> OverallState:
        print("---Node 2---")
        return {"foo": state["baz"] + 1}

    # Build graph
    builder = StateGraph(OverallState)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2)

    # Logic
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")
    builder.add_edge("node_2", END)

    # Add
    graph = builder.compile()

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
        output_name="private_state.png",
    )

    final_state = graph.invoke({"foo": 1})
    logger.info("Final state (json):\n%s", json.dumps(final_state, indent=2))

    """`baz` is only included in `PrivateState`.

    `node_2` uses `PrivateState` as input, but writes out to `OverallState`.

    So, we can see that `baz` is excluded from the graph output because it is not in `OverallState`.
    """
    return graph


##############################################
# Input / Output Schema
##############################################
def input_output_schema_example():
    """
    ##
    # Input / Output Schema
    By default, `StateGraph` takes in a single schema and all nodes are expected to communicate with that schema.
    However, it is also possible to [define explicit input and output schemas for a graph](https://docs.langchain.com/oss/python/langgraph/use-graph-api#define-input-and-output-schemas).
    In these cases, we often define an "internal" schema that contains *all* keys relevant to graph operations.
    But we use specific `input` and `output` schemas to constrain the input and output.
    First, let's just run the graph with a single schema.
    """

    class OverallState(TypedDict):
        question: str
        answer: str
        notes: str

    def thinking_node(state: OverallState):
        return {"answer": "bye", "notes": "... his name is Matthew"}

    def answer_node(state: OverallState):
        return {"answer": "bye Matthew"}

    graph = StateGraph(OverallState)
    graph.add_node("answer_node", answer_node)
    graph.add_node("thinking_node", thinking_node)
    graph.add_edge(START, "thinking_node")
    graph.add_edge("thinking_node", "answer_node")
    graph.add_edge("answer_node", END)

    graph = graph.compile()

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
        output_name="input_output_schema.png",
    )

    """Notice that the output of invoke contains all keys in `OverallState`."""

    final_state = graph.invoke({"question": "hi"})
    logger.info("Final state (json):\n%s", json.dumps(final_state, indent=2))
    return graph


##############################################
# Specific Input / Output Schema
##############################################
def specific_input_output_schema_example():
    """
    Now, let's use a specific `input` and `output` schema with our graph.
    Here, `input` / `output` schemas perform *filtering* on what keys are permitted on the input and output of the graph.
    In addition, we can use a type hint `state: InputState` to specify the input schema of each of our nodes.
    This is important when the graph is using multiple schemas.
    We use type hints below to, for example, show that the output of `answer_node` will be filtered to `OutputState`.
    """

    class InputState(TypedDict):
        question: str

    class OutputState(TypedDict):
        answer: str

    class OverallState(TypedDict):
        question: str
        answer: str
        notes: str

    def thinking_node(state: InputState):
        return {"answer": "bye", "notes": "... his is name is Matthew"}

    def answer_node(state: OverallState) -> OutputState:
        return {"answer": "bye Matthew"}

    graph = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
    graph.add_node("answer_node", answer_node)
    graph.add_node("thinking_node", thinking_node)
    graph.add_edge(START, "thinking_node")
    graph.add_edge("thinking_node", "answer_node")
    graph.add_edge("answer_node", END)

    graph = graph.compile()

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
        output_name="input_output_schema.png",
    )

    final_state = graph.invoke({"question": "hi"})
    logger.info("Final state (json):\n%s", json.dumps(final_state, indent=2))

    """We can see the `output` schema constrains the output to only the `answer` key."""
    return graph


##############################################
# Build App
##############################################
def build_app():

    init_runtime()
    init_langsmith()

    graph = private_state_example()
    graph = input_output_schema_example()
    graph = specific_input_output_schema_example()

    return graph


graph = build_app()

# Only run demos when you execute the file directly (NOT when Studio imports it).
if __name__ == "__main__":

    print("Program Done.")
