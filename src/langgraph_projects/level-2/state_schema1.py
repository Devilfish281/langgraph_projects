# -*- coding: utf-8 -*-
# langgraph_projects/level-2/state_schema1.py
"""
In this module, we're going to build a deeper understanding of both state and memory.

Lets see few different ways to define your state schema.
"""


# %pip install --quiet -U langgraph
###########################################################
## Supporting Code
###########################################################
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


"""## Schema

When we define a LangGraph `StateGraph`, we use a [state schema](https://docs.langchain.com/oss/python/langgraph/graph-api/#state).

The state schema represents the structure and types of data that our graph will use.

All nodes are expected to communicate with that schema.

LangGraph offers flexibility in how you define your state schema, accommodating various Python [types](https://docs.python.org/3/library/stdtypes.html#type-objects) and validation approaches!

## TypedDict

As we mentioned in Module 1, we can use the `TypedDict` class from python's `typing` module.

It allows you to specify keys and their corresponding value types.

But, note that these are type hints.

They can be used by static type checkers (like [mypy](https://github.com/python/mypy)) or IDEs to catch potential type-related errors before the code is run.

But they are not enforced at runtime!
"""

from typing_extensions import TypedDict


class TypedDictState(TypedDict):
    foo: str
    bar: str


"""For more specific value constraints, you can use things like the `Literal` type hint.

Here, `mood` can only be either "happy" or "sad".
"""

from typing import Literal


class TypedDictState(TypedDict):
    name: str
    mood: Literal["happy", "sad"]
    random_value: NotRequired[float]


"""
We can use our defined state class (e.g., here `TypedDictState`) in LangGraph by simply passing it to `StateGraph`.

And, we can think about each state key as just a "channel" in our graph.

As discussed in Module 1, we overwrite the value of a specified key or "channel" in each node.
"""

import random


def node_1(state):
    logger.info("---Node 1---")
    rv = random.random()
    return {
        "name": state["name"] + " is ... ",
        "random_value": rv,
    }


def node_1_new(state):
    logger.info("---Node 1 new---")
    rv = random.random()
    return {"name": state.name + " is ... ", "random_value": rv}


def node_2(state):
    logger.info("---Node 2---")
    return {"mood": "happy"}


def node_3(state):
    logger.info("---Node 3---")
    return {"mood": "sad"}


def decide_mood(state) -> Literal["node_2", "node_3"]:

    # Here, let's just do a 50 / 50 split between nodes 2, 3
    # if random.random() < 0.5:
    if state["random_value"] < 0.5:

        # 50% of the time, we return Node 2
        return "node_2"

    # 50% of the time, we return Node 3
    return "node_3"


def decide_mood_new(state) -> Literal["node_2", "node_3"]:

    # Here, let's just do a 50 / 50 split between nodes 2, 3
    # if random.random() < 0.5:
    if state.random_value < 0.5:
        # 50% of the time, we return Node 2
        return "node_2"

    # 50% of the time, we return Node 3
    return "node_3"


# from langgraph.graph import END, START, StateGraph
def build_app():
    init_runtime()
    init_langsmith()

    # Build graph
    builder = StateGraph(TypedDictState)
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


from dataclasses import dataclass, field


@dataclass
class DataclassState:
    name: str
    mood: Literal["happy", "sad"]
    random_value: float | None = None


def build_dataclasses_app():
    init_runtime()
    init_langsmith()

    # Build graph
    builder = StateGraph(DataclassState)
    builder.add_node("node_1", node_1_new)
    builder.add_node("node_2", node_2)
    builder.add_node("node_3", node_3)

    # Logic
    builder.add_edge(START, "node_1")
    builder.add_conditional_edges("node_1", decide_mood_new)
    builder.add_edge("node_2", END)
    builder.add_edge("node_3", END)

    # Add
    graph = builder.compile()
    return graph


from pydantic import BaseModel, ValidationError, field_validator


# If you don’t even want callers to be able to provide it during initialization,
class PydanticState(BaseModel):
    name: str
    mood: str  # "happy" or "sad"
    random_value: float = field(init=False)  # not accepted in __init__

    @field_validator("mood")
    @classmethod
    def validate_mood(cls, value):
        # Ensure the mood is either "happy" or "sad"
        if value not in ["happy", "sad"]:
            raise ValueError("Each mood must be either 'happy' or 'sad'")
        return value


PydanticState


def build_pydantic_state_app():

    init_runtime()
    init_langsmith()

    """We can use `PydanticState` in our graph seamlessly."""
    # Build graph
    builder = StateGraph(PydanticState)
    builder.add_node("node_1", node_1_new)
    builder.add_node("node_2", node_2)
    builder.add_node("node_3", node_3)

    # Logic
    builder.add_edge(START, "node_1")
    builder.add_conditional_edges("node_1", decide_mood_new)
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
    """Because our state is a dict, we simply invoke the graph with a dict to set an initial value of the `name` key in our state."""
    logger.debug("Invoking model...")
    final_state = graph.invoke({"name": "Lance"})
    logger.debug("Model returned.")

    logger.info(
        "Final state: name=%s | mood=%s | random_value=%s",
        final_state.get("name"),
        final_state.get("mood"),
        final_state.get("random_value"),
    )

    """## Dataclass

    Python's [dataclasses](https://docs.python.org/3/library/dataclasses.html) provide [another way to define structured data](https://www.datacamp.com/tutorial/python-data-classes).

    Dataclasses offer a concise syntax for creating classes that are primarily used to store data.
    """

    # from dataclasses import dataclass, field

    # # If you don’t even want callers to be able to provide it during initialization,
    # @dataclass
    # class DataclassState:
    #     name: str
    #     mood: Literal["happy", "sad"]
    #     random_value: float = field(init=False)  # not accepted in __init__

    """To access the keys of a `dataclass`, we just need to modify the subscripting used in `node_1`:

    * We use `state.name` for the `dataclass` state rather than `state["name"]` for the `TypedDict` above

    You'll notice something a bit odd: in each node, we still return a dictionary to perform the state updates.

    This is possible because LangGraph stores each key of your state object separately.

    The object returned by the node only needs to have keys (attributes) that match those in the state!

    In this case, the `dataclass` has key `name` so we can update it by passing a dict from our node, just as we did when state was a `TypedDict`.

    def node_1_new(state):
        logger.info("---Node 1 new---")
        state.random_value = random.random()
        return {"name": state.name + " is ... "}
    """
    graph = build_dataclasses_app()
    """We invoke with a `dataclass` to set the initial values of each key / channel in our state!"""

    logger.debug("Invoking model...")
    final_state = graph.invoke(DataclassState(name="Lance", mood="sad"))
    logger.debug("Model returned.")

    logger.info(
        "Final state: name=%s | mood=%s | random_value=%s",
        final_state.get("name"),
        final_state.get("mood"),
        final_state.get("random_value"),
    )

    """
    ## Pydantic

    As mentioned, `TypedDict` and `dataclasses` provide type hints but they don't enforce types at runtime.

    This means you could potentially assign invalid values without raising an error!

    For example, we can set `mood` to `mad` even though our type hint specifies `mood: list[Literal["happy","sad"]]`.
    """

    dataclass_instance = DataclassState(name="Lance", mood="mad")

    """[Pydantic](https://docs.pydantic.dev/latest/api/base_model/) is a data validation and settings management library using Python type annotations.

    It's particularly well-suited [for defining state schemas in LangGraph](https://docs.langchain.com/oss/python/langgraph/use-graph-api#use-pydantic-models-for-graph-state) due to its validation capabilities.

    Pydantic can perform validation to check whether data conforms to the specified types and constraints at runtime.
    """

    # # If you don’t even want callers to be able to provide it during initialization,
    # class PydanticState(BaseModel):
    #     name: str
    #     mood: str  # "happy" or "sad"
    #     random_value: float = field(init=False)  # not accepted in __init__
    #     @field_validator("mood")
    #     @classmethod
    #     def validate_mood(cls, value):
    #         # Ensure the mood is either "happy" or "sad"
    #         if value not in ["happy", "sad"]:
    #             raise ValueError("Each mood must be either 'happy' or 'sad'")
    #         return value

    state = None  # Added Code
    try:
        state = PydanticState(name="John Doe", mood="mad")
    except ValidationError as exc:
        logger.exception(
            "PydanticState validation failed; continuing with fallback."
        )  # Added Code
        logger.error(
            "Validation details: %s", exc.errors()
        )  # Added Code (structured details)
        state = PydanticState(name="John Doe", mood="sad")  # Added Code (fallback)

    # Add
    graph = build_pydantic_state_app()

    logger.debug("Invoking model...")
    final_state = graph.invoke(PydanticState(name="Lance", mood="sad"))
    logger.debug("Model returned.")

    logger.info(
        "Final state: name=%s | mood=%s | random_value=%s",
        final_state.name,
        final_state.mood,
        final_state.random_value,
    )

    print("Program Done.")
