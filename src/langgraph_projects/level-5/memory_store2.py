# src/langgraph_projects/level-5/memory_store2.py
"""
Here, we'll introduce the [LangGraph Memory Store](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore) as a way to save and retrieve long-term memories.
We'll build a chatbot that uses both `short-term (within-thread)` and `long-term (across-thread)` memory.
We'll focus on long-term [semantic memory](https://docs.langchain.com/oss/python/concepts/memory#semantic-memory), which will be facts about the user.
These long-term memories will be used to create a personalized chatbot that can remember facts about the user.
It will save memory ["in the hot path"](https://docs.langchain.com/oss/python/concepts/memory#writing-memories), as the user is chatting with it.
"""


# %pip install -U langchain_openai langgraph langchain_core

# We'll use [LangSmith](https://docs.langchain.com/langsmith/home) for [tracing](https://docs.langchain.com/langsmith/observability-concepts).

###########################################################
## Supporting Code
###########################################################
import asyncio
import json
import logging
import operator
import os
import subprocess
import sys
import threading
import uuid

# from concurrent.futures import thread
from operator import add
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    NotRequired,
    Optional,
    TypedDict,
    Union,
)

from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string,
)
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, Send, interrupt
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown

from langgraph_projects.my_utils.configuration import configuration
from langgraph_projects.my_utils.load_env import load_dotenv_only, validate_environment
from langgraph_projects.my_utils.logger_setup import setup_logger

# from PIL import Image


# from pyexpat.errors import messages


logger = logging.getLogger(__name__)  # Reuse the global logger
_MODEL = None
_LANGSMITH_ENABLED: bool = False
_LANGSMITH_INITIALIZED = False
_RUNTIME_INITIALIZED = False


_INIT_LOCK = threading.Lock()
_MODEL_LOCK = threading.Lock()

LANGCHAIN_PROJECT_NAME = "BreakpointsProject"


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

        load_dotenv_only()
        logger = setup_logger()
        validate_environment(log=logger)
        logger.debug("my_langchain_logger Started!")
        logger.debug(f"Effective LOG_LEVEL from env: {os.getenv('LOG_LEVEL')}")
        logger.debug(f"Logger effective level: {logger.getEffectiveLevel()}")

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
    - ``LANGCHAIN_PROJECT`` = ``"BreakpointsProject"``

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

        # Set the LANGCHAIN_PROJECT_NAME at the top of the file
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
            # Open the ACTUAL saved file so the viewer shows messages_as_state.png
            if sys.platform.startswith("win"):  # added Line
                os.startfile(output_file_path)  # added Line
            elif sys.platform == "darwin":  # added Line
                subprocess.run(
                    ["open", str(output_file_path)], check=False
                )  # added Line
            else:  # Linux  #added Line
                subprocess.run(
                    ["xdg-open", str(output_file_path)], check=False
                )  # added Line
        except Exception:
            logger.exception("Saved graph PNG but failed to open viewer.")


from typing import Any, Mapping


def _to_jsonable_state(obj: Any) -> Any:
    """Convert state objects into JSON-serializable structures."""
    # Already JSON-friendly
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable_state(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable_state(v) for v in obj]

    # LangGraph local StateSnapshot-like objects
    # (values/next/config/metadata/tasks/interrupts)
    values = getattr(obj, "values", None)
    next_nodes = getattr(obj, "next", None)
    config = getattr(obj, "config", None)
    metadata = getattr(obj, "metadata", None)
    tasks = getattr(obj, "tasks", None)
    interrupts = getattr(obj, "interrupts", None)

    if any(
        x is not None for x in (values, next_nodes, config, metadata, tasks, interrupts)
    ):
        return {
            "values": _to_jsonable_state(values),
            "next": _to_jsonable_state(next_nodes),
            "config": _to_jsonable_state(config),
            "metadata": _to_jsonable_state(metadata),
            "tasks": _to_jsonable_state(tasks),
            "interrupts": _to_jsonable_state(interrupts),
        }

    # Fallback: stringify unknown types (messages, tool calls, etc.)
    return str(obj)


"""
    log_state_time_travel(
        to_replay, raw_flag=True, pretty_raw=True, label="Replaying from state"
    )
"""


def log_state_time_travel(
    state: Any,
    *,
    raw_flag: bool = True,
    pretty_raw: bool = True,  # <-- Added Code: better name than "compact" for raw formatting
    label: str = "Current state",
) -> None:
    print("\n" + "---" * 25)

    if raw_flag:
        if pretty_raw:
            safe = _to_jsonable_state(state)
            logger.info(
                "%s (raw):\n%s", label, json.dumps(safe, indent=2, ensure_ascii=False)
            )
        else:
            logger.info("%s (raw): %s", label, state)

    # -------------------------
    # API dict state path
    # -------------------------
    if isinstance(state, Mapping):
        values = state.get("values", None)
        next_nodes = state.get("next", None)
        config_like = state.get("checkpoint") or state.get("metadata") or None

        messages = None
        if isinstance(values, Mapping):
            messages = values.get("messages", None)

        logger.info("%s values: %s", label, values)
        logger.info("%s next: %s", label, next_nodes)
        logger.info("%s config: %s", label, config_like)
        logger.info("%s messages: %s", label, messages)

        print("\n" + "---" * 25)
        return

    # -------------------------
    # Local StateSnapshot path
    # -------------------------
    values = getattr(state, "values", None)
    next_nodes = getattr(state, "next", None)
    config = getattr(state, "config", None)

    messages = None
    if isinstance(values, Mapping):
        messages = values.get("messages", None)

    logger.info("%s values: %s", label, values)
    logger.info("%s next: %s", label, next_nodes)
    logger.info("%s config: %s", label, config)
    logger.info("%s messages: %s", label, messages)

    print("\n" + "---" * 25)


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


def simple_test_no_graph():
    """A simple test function to verify that the module runs without errors."""
    """
    ## Introduction to the LangGraph Store
    The  [LangGraph Memory Store](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore) provides a way to store and retrieve information *across threads* in LangGraph.
    This is an  [open source base class](https://blog.langchain.com/launching-long-term-memory-support-in-langgraph/) for persistent `key-value` stores.
    """

    # import uuid

    # from langgraph.store.memory import InMemoryStore

    in_memory_store = InMemoryStore()

    """
    When storing objects (e.g., memories) in the [Store](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore), we provide:
    - The `namespace` for the object, a tuple (similar to directories)
    - the object `key` (similar to filenames)
    - the object `value` (similar to file contents)
    We use the [put](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore.put) method to save an object to the store by `namespace` and `key`.
    """

    # Namespace for the memory to save
    user_id = "1"
    namespace_for_memory = (user_id, "memories")

    # Save a memory to namespace as key and value
    key = str(uuid.uuid4())

    # The value needs to be a dictionary
    value = {"food_preference": "I like pizza"}

    # Save the memory
    in_memory_store.put(namespace_for_memory, key, value)

    """
    We use [search](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore.search) to retrieve objects from the store by `namespace`.
    This returns a list.
    """

    # Search
    memories = in_memory_store.search(namespace_for_memory)
    # type(memories)
    logger.info("memories type=%s", type(memories))
    logger.info("memories count=%d", len(memories))
    if memories:
        first = memories[0]
        logger.info("first item type=%s", type(first))
        logger.info("first.key=%s", first.key)
        logger.info("first.namespace=%s", first.namespace)
        logger.info("first.value=%s", first.value)
        logger.info("first.score=%s", getattr(first, "score", None))

    # Metatdata
    # To see all fields quickly (great for debugging):
    logger.info("first item dict=%s", memories[0].dict())

    # The key, value
    logger.info("Memory key=%s | value=%s", memories[0].key, memories[0].value)

    """We can also use [get](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore.get) 
    to retrieve an object by `namespace` and `key`."""

    # Get the memory by namespace and key
    memory = in_memory_store.get(namespace_for_memory, key)

    logger.info("dict=%s", memory.dict())
    logger.info("Simple store test complete.")


"""
## Chatbot with long-term memory
We want a chatbot that [has two types of memory](https://docs.google.com/presentation/d/181mvjlgsnxudQI6S3ritg9sooNyu4AcLLFH1UK0kIuk/edit#slide=id.g30eb3c8cf10_0_156):
1. `Short-term (within-thread) memory`: Chatbot can persist conversational history and / or allow interruptions in a chat session.
2. `Long-term (cross-thread) memory`: Chatbot can remember information about a specific user *across all chat sessions*.
"""

"""
Fo
See Module 2 and our [conceptual docs](https://docs.langchain.com/oss/python/langgraph/persistence) for more on checkpointers, but in summary:
* They write the graph state at each step to a thread.
* They persist the chat history in the thread.
* They allow the graph to be interrupted and / or resumed from any step in the thread.
And, for `long-term memory`, we'll use the [LangGraph Store](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore) as introduced above.
"""

"""
The chat history will be saved to short-term memory using the checkpointer.
The chatbot will reflect on the chat history.
It will then create and save a memory to the [LangGraph Store](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore).
This memory is accessible in future chat sessions to personalize the chatbot's responses.
"""


# from IPython.display import Image, display
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_core.runnables.config import RunnableConfig
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import END, START, MessagesState, StateGraph
# from langgraph.store.base import BaseStore


#######################################################
# Chatbot instruction
#######################################################
# Chatbot instruction
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user.
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""

# Create new memory from the chat history and any existing memory
CREATE_MEMORY_INSTRUCTION = """"You are collecting information about the user to personalize your responses.

CURRENT USER INFORMATION:
{memory}

INSTRUCTIONS:
1. Review the chat history below carefully
2. Identify new information about the user, such as:
   - Personal details (name, location)
   - Preferences (likes, dislikes)
   - Interests and hobbies
   - Past experiences
   - Goals or future plans
3. Merge any new information with existing memory
4. Format the memory as a clear, bulleted list
5. If new information conflicts with existing memory, keep the most recent version

Remember: Only include factual information directly stated by the user. Do not make assumptions or inferences.

Based on the chat history below, please update the user information:"""


def call_model_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Load memory from the store and use it to personalize the chatbot's response."""
    # added for STUDIO-273: to handle both dict and StateSnapshot state shapes
    # Get configuration
    configurable = configuration.Configuration.from_runnable_config(config)

    # Get the user ID from the config
    user_id = configurable.user_id

    # # Get the user ID from the config
    # user_id = config["configurable"]["user_id"]

    # Retrieve memory from the store
    namespace = ("memory", user_id)
    key = "user_memory"
    existing_memory = store.get(namespace, key)

    # Extract the actual memory content if it exists and add a prefix
    if existing_memory:
        # Value is a dictionary with a memory key
        existing_memory_content = existing_memory.value.get("memory")
    else:
        existing_memory_content = "No existing memory found."

    # Format the memory in the system prompt
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=existing_memory_content)

    # Added Code: accept both state shapes
    if isinstance(state, Mapping) and "messages" in state:
        messages = state["messages"]
    else:
        messages = state.get("values", {}).get("messages", [])

    logger.info(
        "call_model_node messages_type=%s count=%s", type(messages), len(messages)
    )  # Added Code

    # Respond using memory as well as the chat history
    response = get_model().invoke([SystemMessage(content=system_msg)] + messages)
    # logger.info("Model response: %s", response)
    # return {"messages": response}
    return {"messages": [response]}  # added Line


def write_memory_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and save a memory to the store."""
    # added for STUDIO-273: to handle both dict and StateSnapshot state shapes
    # Get configuration
    configurable = configuration.Configuration.from_runnable_config(config)

    # Get the user ID from the config
    user_id = configurable.user_id

    # # Get the user ID from the config
    # user_id = config["configurable"]["user_id"]

    # Retrieve existing memory from the store
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")

    # Extract the memory
    if existing_memory:
        existing_memory_content = existing_memory.value.get("memory")
    else:
        existing_memory_content = "No existing memory found."

    # Added Code: accept both state shapes
    if isinstance(state, Mapping) and "messages" in state:  # added Line
        messages = state["messages"]  # added Line
    else:  # added Line
        messages = state.get("values", {}).get("messages", [])  # added Line

    system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=existing_memory_content)
    new_memory = get_model().invoke(
        [SystemMessage(content=system_msg)] + messages
    )  # added Line

    # Overwrite the existing memory in the store
    key = "user_memory"

    # Write value as a dictionary with a memory key
    store.put(namespace, key, {"memory": new_memory.content})
    return {}  # added Line


def generate_memory_graph(
    *,
    use_checkpointer: bool = False,
    return_builder: bool = False,
    cross_thread: bool = False,
) -> Union[StateGraph, object]:

    init_runtime()
    init_langsmith()

    # Define the graph
    # builder = StateGraph(MessagesState)
    builder = StateGraph(MessagesState, config_schema=configuration.Configuration)
    builder.add_node("call_model", call_model_node)
    builder.add_node("write_memory", write_memory_node)

    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", "write_memory")
    builder.add_edge("write_memory", END)

    # optionally return the uncompiled builder
    if return_builder:
        return builder

    if cross_thread:

        # Store for long-term (across-thread) memory
        across_thread_memory = InMemoryStore()

        # Checkpointer for short-term (within-thread) memory
        within_thread_memory = MemorySaver()
        return builder.compile(
            checkpointer=within_thread_memory, store=across_thread_memory
        )
    if use_checkpointer:
        # Set up memory
        memory = MemorySaver()
        # Compile the graph with memory
        return builder.compile(checkpointer=memory)
    else:
        return builder.compile()
    # langgraph dev
    # builder = StateGraph(MessagesState, config_schema=configuration.Configuration)
    # graph = builder.compile()


# analysts graph
def get_memory_graph_remote():
    return generate_memory_graph(
        use_checkpointer=False, return_builder=False, cross_thread=False
    )


memory_graph_remote = get_memory_graph_remote()
memory_graph = None


def test_call_model():

    state = MessagesState(
        values={"messages": [HumanMessage(content="Hi, my name is Matthew")]}
    )
    # Store for long-term (across-thread) memory
    across_thread_memory = InMemoryStore()

    # Checkpointer for short-term (within-thread) memory
    within_thread_memory = MemorySaver()

    config = RunnableConfig(configurable={"user_id": "test_user"})
    store = InMemoryStore()
    result = call_model_node(state, config, store)
    logger.info("Model result: %s", result)

    builder = generate_memory_graph(use_checkpointer=False, return_builder=True)

    # Compile the graph with the checkpointer fir and store
    graph = builder.compile(
        checkpointer=within_thread_memory, store=across_thread_memory
    )

    #######################################################
    ## Display the graph for Store
    #######################################################
    display_graph_if_enabled(
        graph,
        save_env="VIEW_GRAPH",
        open_env="VIEW_GRAPH_OPEN",
        default_save=True,
        default_open=True,
        xray=1,
        images_dir_name="images",
        output_name="store_graph.png",
    )

    """
    When we interact with the chatbot, we supply two things:
    1. `Short-term (within-thread) memory`: A `thread ID` for persisting the chat history.
    2. `Long-term (cross-thread) memory`: A `user ID` to namespace long-term memories to the user.
    Let's see how these work together in practice.
    """

    # We supply a thread ID for short-term (within-thread) memory
    # We supply a user ID for long-term (across-thread) memory
    config = {"configurable": {"thread_id": "1", "user_id": "1"}}

    # User input
    input_messages = [HumanMessage(content="Hi, my name is Matthew")]

    # Run the graph
    for chunk in graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    # User input
    input_messages = [HumanMessage(content="I like to bike around San Francisco")]

    # Run the graph
    for chunk in graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    """We're using the `MemorySaver` checkpointer for within-thread memory.
    This saves the chat history to the thread.
    We can look at the chat history saved to the thread.
    """
    #######################################################
    # short-term memory: look at the chat history saved to the thread
    #######################################################
    thread = {"configurable": {"thread_id": "1"}}
    state = graph.get_state(thread).values
    for m in state["messages"]:
        m.pretty_print()

    """Recall that we compiled the graph with our the store:
    across_thread_memory = InMemoryStore()
     And, we added a node to the graph (`write_memory`) that reflects on the chat history and saves a memory to the store.
    We can to see if the memory was saved to the store.
    """
    #######################################################
    # long-term memory: look at the memory saved to the store
    #######################################################
    # Namespace for the memory to save
    user_id = "1"
    namespace = ("memory", user_id)
    existing_memory = across_thread_memory.get(namespace, "user_memory")
    # existing_memory.dict()
    logger.info("Existing memory dict=%s", existing_memory.dict())
    """Now, let's kick off a *new thread* with the *same user ID*.

    We should see that the chatbot remembered the user's profile and used it to personalize the response.
    """
    #######################################################
    # New thread with same user ID to test long-term memory retrieval
    #######################################################
    # We supply a user ID for across-thread memory as well as a new thread ID
    config = {"configurable": {"thread_id": "2", "user_id": "1"}}

    # User input
    input_messages = [
        HumanMessage(content="Hi! Where would you recommend that I go biking?")
    ]

    # Run the graph
    for chunk in graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    # User input
    input_messages = [
        HumanMessage(
            content="Great, are there any bakeries nearby that I can check out? I like a croissant after biking."
        )
    ]

    # Run the graph
    for chunk in graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    logger.info("Tested call_model_node successfully.")


#######################################################
## LangGraph SDK Example
#######################################################
async def run_langgraph_sdk_example():
    """
    ## Using with LangGraph API
    langgraph dev
    see Readme for more details on how to set up and run the LangGraph API locally.
    """

    from langgraph_sdk import get_client

    pass


#######################################################
# MAIN
#######################################################
# Only run demos when you execute the file directly (NOT when Studio imports it).
if __name__ == "__main__":
    # Demo / manual run (kept under __main__ so imports remain side-effect-light).
    simple_test_flag = False
    test_call_flag = True

    init_runtime()
    init_langsmith()
    ############################################################
    ## Run the dynamic breakpoints example
    ############################################################

    langgraph_dev = False

    if langgraph_dev:
        import asyncio

        asyncio.run(run_langgraph_sdk_example())

    else:
        if simple_test_flag:
            simple_test_no_graph()

        if test_call_flag:
            memory_graph = generate_memory_graph(
                use_checkpointer=False, return_builder=False, cross_thread=True
            )
            #######################################################
            ## Display the graph for Analysts
            #######################################################
            display_graph_if_enabled(
                memory_graph,
                save_env="VIEW_GRAPH",
                open_env="VIEW_GRAPH_OPEN",
                default_save=True,
                default_open=True,
                xray=1,
                images_dir_name="images",
                output_name="analysts_graph.png",
            )
            test_call_model()

    logger.info("All done.")


"""
## Viewing traces in LangSmith
We can see that the memories are retrieved from the store and supplied as part of the system prompt, as expected:
https://smith.langchain.com/

## Studio
We can also interact with our chatbot in Studio.

"""
