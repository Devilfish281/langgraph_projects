# -*- coding: utf-8 -*-
# source: src/langgraph_projects/level-2/state_reducers2.py


"""
# State Reducers
Now, we're going to dive into reducers, which specify how state updates are performed on specific keys / channels in the state schema.
"""
# %pip install --quiet -U langchain_core langgraph

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


########################################################
# Default overwriting state
########################################################
def Default_overwriting_state():
    """
    ## Default overwriting state
    Let's use a `TypedDict` as our state schema.
    """

    class State(TypedDict):
        foo: int

    def node_1(state):
        logger.info("---Node 1---")
        return {"foo": state["foo"] + 1}

    # Build graph
    builder = StateGraph(State)
    builder.add_node("node_1", node_1)

    # Logic
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", END)

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
        output_name="Default_overwriting_state.png",
    )

    logger.debug("Starting graph invocation...")
    final_state = graph.invoke({"foo": 1})
    logger.debug("Ending graph invocation.")
    """ 
    look at the state update, `return {"foo": state['foo'] + 1}`.
    by default LangGraph doesn't know the preferred way to update the state.
    So, it will just overwrite the value of `foo` in `node_1`:
    return {"foo": state['foo'] + 1}
    If we pass `{'foo': 1}` as input, the state returned from the graph is `{'foo': 2}`.
    """
    logger.info("Final state (json):\n%s", json.dumps(final_state, indent=2))

    logger.info(
        "Final state: foo=%s ",
        final_state.get("foo"),
    )

    logger.info(" Default Overwriting State Graph Done")


from langgraph.errors import InvalidUpdateError


#######################################################
# Branching
#######################################################
def Branching():
    class State(TypedDict):
        foo: int

    def node_1(state):
        logger.info("---Node 1---")
        return {"foo": state["foo"] + 1}

    def node_2(state):
        logger.info("---Node 2---")
        return {"foo": state["foo"] + 1}

    def node_3(state):
        logger.info("---Node 3---")
        return {"foo": state["foo"] + 1}

    # Build graph
    builder = StateGraph(State)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2)
    builder.add_node("node_3", node_3)

    # Logic
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")
    builder.add_edge("node_1", "node_3")
    builder.add_edge("node_2", END)
    builder.add_edge("node_3", END)

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
        output_name="Branching.png",
    )

    """
    We see a problem!
    Node 1 branches to nodes 2 and 3.
    Nodes 2 and 3 run in parallel, which means they run in the same step of the graph.
    They both attempt to overwrite the state *within the same step*.
    This is ambiguous for the graph! Which state should it keep?
    """
    #     from langgraph.errors import InvalidUpdateError
    try:
        final_state = graph.invoke({"foo": 1})
    except InvalidUpdateError as e:
        logger.error("InvalidUpdateError occurred: %s", e)
        # raise

    logger.info(" Branching Graph Done")


#######################################################
# Reducers
#######################################################
from operator import add
from typing import Annotated


def Reducers():
    """
    ##
    # Reducers
    [Reducers](https://docs.langchain.com/oss/python/langgraph/graph-api/#reducers) give us a general way to address this problem.
    They specify how to perform updates.
    We can use the `Annotated` type to specify a reducer function.
    For example, in this case let's append the value returned from each node rather than overwriting them.
    We just need a reducer that can perform this: `operator.add` is a function from Python's built-in operator module.
    When `operator.add` is applied to lists, it performs list concatenation.
    """

    class State(TypedDict):
        foo: Annotated[list[int], add]

    def node_1(state):
        logger.info("---Node 1---")
        return {"foo": [state["foo"][0] + 1]}

    # Build graph
    builder = StateGraph(State)
    builder.add_node("node_1", node_1)

    # Logic
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", END)

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
        output_name="Reducers1.png",
    )

    final_state = graph.invoke({"foo": [1]})
    foo_list = final_state.get("foo", [])
    for i, val in enumerate(foo_list):
        logger.info("foo[%d]=%s", i, val)

    ##################################################
    # Branching with reducers
    ##################################################
    """
    Now, our state key `foo` is a list.
    This `operator.add` reducer function will append updates from each node to this list.
    """

    def node_1(state):
        logger.info("---Node 1---")
        return {"foo": [state["foo"][-1] + 1]}

    def node_2(state):
        logger.info("---Node 2---")
        return {"foo": [state["foo"][-1] + 1]}

    def node_3(state):
        logger.info("---Node 3---")
        return {"foo": [state["foo"][-1] + 1]}

    # Build graph
    builder = StateGraph(State)
    builder.add_node("node_1", node_1)
    builder.add_node("node_2", node_2)
    builder.add_node("node_3", node_3)

    # Logic
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")
    builder.add_edge("node_1", "node_3")
    builder.add_edge("node_2", END)
    builder.add_edge("node_3", END)

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
        output_name="Reducers2.png",
    )

    # We can see that updates in nodes 2 and 3 are performed concurrently because they are in the same step.

    final_state = graph.invoke({"foo": [1]})
    foo_list = final_state.get("foo", [])
    for i, val in enumerate(foo_list):
        logger.info("foo[%d]=%s", i, val)
    """
    Now, let's see what happens if we pass `None` to `foo`.
    We see an error because our reducer, `operator.add`, attempts to concatenate `NoneType` pass as input to list in `node_1`.
    """

    try:
        final_state = graph.invoke({"foo": None})
    except TypeError as e:
        logger.error(f"TypeError occurred: {e}")

    logger.info(" Reducers Graph Done")


######################################################
# Custom Reducers
######################################################
def custom_reducers():
    """
    ## Custom Reducers
    To address cases like this,[we can also define custom reducers](https://docs.langchain.com/oss/python/langgraph/use-graph-api#process-state-updates-with-reducers).
    For example, lets define custom reducer logic to combine lists and handle cases where either or both of the inputs might be `None`.
    """

    def reduce_list(left: list | None, right: list | None) -> list:
        """
        Safely combine two lists, handling cases where either or both inputs might be None.
        Args:
            left (list | None): The first list to combine, or None.
            right (list | None): The second list to combine, or None.
        Returns:
            list: A new list containing all elements from both input lists.
                If an input is None, it's treated as an empty list.
        """
        if not left:
            left = []
        if not right:
            right = []
        return left + right

    class DefaultState(TypedDict):
        foo: Annotated[list[int], add]

    class CustomReducerState(TypedDict):
        foo: Annotated[list[int], reduce_list]

    """In `node_1`, we append the value 2."""

    def node_1(state):
        print("---Node 1---")
        return {"foo": [2]}

    # Build graph
    builder = StateGraph(DefaultState)
    builder.add_node("node_1", node_1)

    # Logic
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", END)

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
        output_name="custom_reducers1.png",
    )

    try:
        print(graph.invoke({"foo": None}))
    except TypeError as e:
        logger.error(f"TypeError occurred: {e}")
    ##################################################
    # Now, try with our custom reducer. We can see that no error is thrown.
    ##################################################

    # Build graph
    builder = StateGraph(CustomReducerState)
    builder.add_node("node_1", node_1)

    # Logic
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", END)

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
        output_name="custom_reducers2.png",
    )

    try:
        final_state = graph.invoke({"foo": None})
        foo_list = final_state.get("foo", [])
        for i, val in enumerate(foo_list):
            logger.info("foo[%d]=%s", i, val)
    except TypeError as e:
        logger.error(f"TypeError occurred: {e}")

    logger.info(" Custom Reducers Graph Done")

    return graph


######################################################
# Messages and add_messages
######################################################
def Messages_and_add_messages():
    """
    ## Messages
    In module 1, we showed how to use a built-in reducer, `add_messages`, to handle messages in state.
    We also showed that [`MessagesState` is a useful shortcut if you want to work with messages](https://docs.langchain.com/oss/python/langgraph/use-graph-api#messagesstate).
    * `MessagesState` has a built-in `messages` key
    * It also has a built-in `add_messages` reducer for this key
    These two are equivalent.
    We'll use the `MessagesState` class via `from langgraph.graph import MessagesState` for brevity.
    """

    # from typing import Annotated

    # from langchain_core.messages import AnyMessage
    # from langgraph.graph import MessagesState
    # from langgraph.graph.message import add_messages

    # Define a custom TypedDict that includes a list of messages with add_messages reducer
    class CustomMessagesState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        added_key_1: str
        added_key_2: str
        # etc

    # Use MessagesState, which includes the messages key with add_messages reducer
    class ExtendedMessagesState(MessagesState):
        # Add any keys needed beyond messages, which is pre-built
        added_key_1: str
        added_key_2: str
        # etc

    # Let's talk a bit more about usage of the `add_messages` reducer.

    # from langchain_core.messages import AIMessage, HumanMessage
    # from langgraph.graph.message import add_messages

    # Initial state
    initial_messages = [
        AIMessage(content="Hello! How can I assist you?", name="Model"),
        HumanMessage(
            content="I'm looking for information on marine biology.", name="Matthew"
        ),
    ]

    # New message to add
    new_message = AIMessage(
        content="Sure, I can help with that. What specifically are you interested in?",
        name="Model",
    )

    # Test
    add_messages(initial_messages, new_message)
    logger.info(
        "Initial messages (model_dump): %s",
        json.dumps([m.model_dump() for m in initial_messages], indent=2),
    )

    """So we can see that `add_messages` allows us to append messages to the `messages` key in our state.

    ### Re-writing

    Let's show some useful tricks when working with the `add_messages` reducer.

    If we pass a message with the same ID as an existing one in our `messages` list, it will get overwritten!
    """

    # Initial state
    initial_messages = [
        AIMessage(content="Hello! How can I assist you?", name="Model", id="1"),
        HumanMessage(
            content="I'm looking for information on marine biology.",
            name="Matthew",
            id="2",
        ),
    ]

    # New message to add
    new_message = HumanMessage(
        content="I'm looking for information on whales, specifically",
        name="Matthew",
        id="2",
    )

    # Test
    messages = add_messages(initial_messages, new_message)
    logger.info(
        "For messages (model_dump): %s",
        json.dumps([m.model_dump() for m in messages], indent=2),
    )
    #############################################
    # We can see that the message with ID `2` is overwritten by the new message.
    #############################################
    """
    ### Removal
    We can remove messages by using [RemoveMessage](https://docs.langchain.com/oss/python/langgraph/add-memory#delete-messages).
    """

    from langchain_core.messages import RemoveMessage

    # Message list
    messages = [AIMessage("Hi.", name="Bot", id="1")]
    messages.append(HumanMessage("Hi.", name="Matthew", id="2"))
    messages.append(
        AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3")
    )
    messages.append(
        HumanMessage(
            "Yes, I know about whales. But what others should I learn about?",
            name="Matthew",
            id="4",
        )
    )

    #############################################
    # Now, let's remove some messages.
    #############################################
    logger.info("Now, let's remove some messages.")
    # Isolate messages to delete
    delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]
    logger.info("Delete messages " + str(delete_messages))

    messages = add_messages(messages, delete_messages)  # Changed Code

    logger.info(  # Changed Code
        "messages after delete (model_dump): %s",
        json.dumps([m.model_dump() for m in messages], indent=2),
    )
    """
    We can see that mesage IDs 1 and 2, as noted in `delete_messages` are removed by the reducer.
    We'll see this put into practice a bit later.
    """
    logger.info(" Messages and add_messages Graph Done")


######################################################
# build app
#######################################################
def build_app():

    init_runtime()
    init_langsmith()

    Default_overwriting_state()
    Branching()
    Reducers()
    graph = custom_reducers()
    Messages_and_add_messages()

    return graph


graph = build_app()

# Only run demos when you execute the file directly (NOT when Studio imports it).
if __name__ == "__main__":
    # Demo / manual run (kept under __main__ so imports remain side-effect-light).

    print("Program Done.")
