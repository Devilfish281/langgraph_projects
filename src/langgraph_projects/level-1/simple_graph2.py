# -*- coding: utf-8 -*-
# Langchain-acadeemy/src/langchain_academy/level-1/simple_graph2.py
# import logging
# import os
# import threading
# from pathlib import Path
# from typing import Literal

# from IPython.display import display
# from langchain_core.messages import HumanMessage
# from langchain_openai import ChatOpenAI
# from langchain_tavily import TavilySearch
# from langgraph.graph import END, START, MessagesState, StateGraph

# # from langgraph.graph import END, START, StateGraph
# from langgraph.prebuilt import ToolNode
# from langsmith import utils
# from PIL import Image

# %pip install --quiet -U langgraph


###########################################################
## Supporting Code
###########################################################
import logging
import os
import random
import threading
from pathlib import Path
from typing import Literal, NotRequired

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from PIL import Image

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


#####################################################################
### END
#####################################################################


"""## Nodes

[Nodes](https://docs.langchain.com/oss/python/langgraph/graph-api/#nodes) are just python functions.

The first positional argument is the state, as defined above.

Because the state is a `TypedDict` with schema as defined above, each node can access the key, `graph_state`, with `state['graph_state']`.

Each node returns a new value of the state key `graph_state`.
  
By default, the new value returned by each node [will override](https://docs.langchain.com/oss/python/langgraph/graph-api/#reducers) the prior state value.
"""


def node_1(state):
    print("---Node 1---")
    state["random_value"] = random.random()
    return {
        "graph_state": state["graph_state"] + " I am",
        "random_value": state["random_value"],
    }


def node_2(state):
    print("---Node 2---")
    return {"graph_state": state["graph_state"] + " happy!"}


def node_3(state):
    print("---Node 3---")
    return {"graph_state": state["graph_state"] + " sad!"}


"""## Edges

[Edges](https://docs.langchain.com/oss/python/langgraph/graph-api/#edges) connect the nodes.

Normal Edges are used if you want to *always* go from, for example, `node_1` to `node_2`.

[Conditional Edges](https://docs.langchain.com/oss/python/langgraph/graph-api/#conditional-edges) are used if you want to *optionally* route between nodes.

Conditional edges are implemented as functions that return the next node to visit based on some logic.
"""


from typing import Literal


def decide_mood(state) -> Literal["node_2", "node_3"]:

    # Often, we will use state to decide on the next node to visit
    user_input = state["graph_state"]

    # Here, let's just do a 50 / 50 split between nodes 2, 3
    # if random.random() < 0.5:
    if state["random_value"] < 0.5:

        # 50% of the time, we return Node 2
        return "node_2"

    # 50% of the time, we return Node 3
    return "node_3"


def build_app():
    init_runtime()
    init_langsmith()

    """
    ## State

    First, define the [State](https://docs.langchain.com/oss/python/langgraph/graph-api#state) of the graph.

    The State schema serves as the input schema for all Nodes and Edges in the graph.

    Let's use the `TypedDict` class from python's `typing` module as our schema, which provides type hints for the keys.
    """

    from typing_extensions import TypedDict

    class State(TypedDict):
        graph_state: str
        random_value: NotRequired[float]

    """
    ## Graph Construction

    Now, we build the graph from our components defined above.

    The [StateGraph class](https://docs.langchain.com/oss/python/langgraph/graph-api/#stategraph) is the graph class that we can use.

    First, we initialize a StateGraph with the `State` class we defined above.

    Then, we add our nodes and edges.

    We use the  [`START` Node, a special node](https://docs.langchain.com/oss/python/langgraph/graph-api/#start-node) that sends user input to the graph, to indicate where to start our graph.

    The [`END` Node](https://docs.langchain.com/oss/python/langgraph/graph-api/#end-node) is a special node that represents a terminal node.

    Finally, we [compile our graph](https://docs.langchain.com/oss/python/langgraph/graph-api/#compiling-your-graph) to perform a few basic checks on the graph structure.

    We can visualize the graph as a [Mermaid diagram](https://github.com/mermaid-js/mermaid).
    """

    # Build graph
    builder = StateGraph(State)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2)
    builder.add_node("node_3", node_3)

    # Logic
    builder.add_edge(START, "node_1")
    builder.add_conditional_edges("node_1", decide_mood)
    builder.add_edge("node_2", END)
    builder.add_edge("node_3", END)

    # Add
    graph = builder.compile()

    return graph


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
        output_name="simple_graph_image.png",
    )
    #######################################################

    """
    ## Graph Invocation

    The compiled graph implements the [runnable](https://reference.langchain.com/python/langchain_core/runnables/?h=runnables) protocol.

    This provides a standard way to execute LangChain components.

    `invoke` is one of the standard methods in this interface.

    The input is a dictionary `{"graph_state": "Hi, this is lance."}`, which sets the initial value for our graph state dict.

    When `invoke` is called, the graph starts execution from the `START` node.

    It progresses through the defined nodes (`node_1`, `node_2`, `node_3`) in order.

    The conditional edge will traverse from node `1` to node `2` or `3` using a 50/50 decision rule.

    Each node function receives the current state and returns a new value, which overrides the graph state.

    The execution continues until it reaches the `END` node.
    """
    logger.debug("Invoking model...")
    final_state = graph.invoke({"graph_state": "Hi, this is Lance."})
    logger.debug("Model returned.")

    """
    `invoke` runs the entire graph synchronously.

    This waits for each step to complete before moving to the next.

    It returns the final state of the graph after all nodes have executed.

    In this case, it returns the state after `node_3` has completed:

    ```
    {'graph_state': 'Hi, this is Lance. I am sad!'}
    ```
    """

    logger.info(
        "Final state: graph_state=%s | random_value=%s",
        final_state.get("graph_state"),
        final_state.get("random_value"),
    )

    print(f"Final graph_state: {final_state['graph_state']}")
    print(f"Final random_value: {final_state.get('random_value')}")
    print("Program Done.")
    # langgraph dev
