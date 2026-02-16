# src/langgraph_projects/level-5/memoryschema_profile3.py
"""
Our chatbot saved memories as a string. In practice, we often want memories to have a structure.
For example, memories can be a [single, continuously updated schema](https://docs.langchain.com/oss/python/concepts/memory#profile).
In our case, we want this to be a single user profile.
We'll extend our chatbot to save semantic memories to a single [user profile](https://docs.langchain.com/oss/python/concepts/memory#profile).
We'll also introduce a library, [Trustcall](https://github.com/hinthornw/trustcall), to update this schema with new information.
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
from unittest import result

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
from trustcall import create_extractor

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
"""
## Defining a user profile schema
Python has many different types for [structured data](https://docs.langchain.com/oss/python/langchain/models#structured-outputs), such as TypedDict, Dictionaries, JSON, and [Pydantic](https://docs.pydantic.dev/latest/).
Let's start by using TypedDict to define a user profile schema.
"""


class UserProfile(TypedDict):
    user_name: str  # The user's preferred name
    interests: List[str]  # A list of the user's interests


def simple_test_no_graph():
    """
    ## Saving a schema to the store
    The  [LangGraph Store](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore) accepts any Python dictionary as the `value`.
    """

    # TypedDict instance
    user_profile: UserProfile = {
        "user_name": "Matthew",
        "interests": ["biking", "technology", "coffee"],
    }
    # user_profile
    logger.info("User profile: %s", user_profile)

    # Initialize the in-memory store (for testing purposes)
    in_memory_store = InMemoryStore()

    # Namespace for the memory to save
    user_id = "1"
    namespace_for_memory = (user_id, "memories")

    # Save a memory to namespace as key and value
    key = "user_profile"
    value = user_profile
    # Save the memory
    in_memory_store.put(namespace_for_memory, key, value)

    # Search
    for m in in_memory_store.search(namespace_for_memory):
        # print(m.dict())
        logger.info("Memory dict=%s", m.dict())

    # We can also use [get](https://reference.langchain.com/python/langgraph/store/?h=basestor#langgraph.store.base.BaseStore.get) to retrieve a specific object by namespace and key."""

    # Get the memory by namespace and key
    profile = in_memory_store.get(namespace_for_memory, "user_profile")
    # profile.value
    logger.info("Retrieved profile value=%s", profile.value)
    ##########################################################
    # Chatbot with profile schema
    ##########################################################
    """
    Now we know how to specify a schema for the memories and save it to the store.
    Now, how do we actually *create* memories with this particular schema?
    In our chatbot, we [want to create memories from a user chat](https://docs.langchain.com/oss/python/concepts/memory#profile).
    This is where the concept of [structured outputs](https://docs.langchain.com/oss/python/langchain/models#structured-outputs) is useful.
    LangChain's [chat model](https://docs.langchain.com/oss/python/langchain/models) interface has a [`with_structured_output`](https://docs.langchain.com/oss/python/langchain/models#structured-outputs) method to enforce structured output.
    This is useful when we want to enforce that the output conforms to a schema, and it parses the output for us.
    """
    """
    Let's pass the `UserProfile` schema we created to the `with_structured_output` method.
    We can then invoke the chat model with a list of [messages](https://docs.langchain.com/oss/python/langchain/messages) and get a structured output that conforms to our schema.
    """
    # from langchain_core.messages import HumanMessage
    # from langchain_openai import ChatOpenAI
    # from pydantic import BaseModel, Field
    # Bind schema to model
    model_with_structure = get_model().with_structured_output(UserProfile)

    # Invoke the model to produce structured output that matches the schema
    structured_output = model_with_structure.invoke(
        [HumanMessage("My name is Matthew, I like to bike.")]
    )
    # structured_output
    logger.info("Structured output: %s", structured_output)

    ##################################################################
    logger.info("Simple store test complete.")

    """
    Now, let's use this with our chatbot.
    This only requires minor changes to the `write_memory` function.
    We use `model_with_structure`, as defined above, to produce a profile that matches our schema.
    """
    # from IPython.display import Image, display
    # from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    # from langchain_core.runnables.config import RunnableConfig
    # from langgraph.checkpoint.memory import MemorySaver
    # from langgraph.graph import END, START, MessagesState, StateGraph
    # from langgraph.store.base import BaseStore


#######################################################
# Chatbot instruction
#######################################################
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user.
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""

# Create new memory from the chat history and any existing memory
CREATE_MEMORY_INSTRUCTION = """Create or update a user profile memory based on the user's chat history.
This will be saved for long-term memory. If there is an existing memory, simply update it.
Here is the existing memory (it may be empty): {memory}"""


def call_model_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Load memory from the store and use it to personalize the chatbot's response."""
    # added for STUDIO-273: to handle both dict and StateSnapshot state shapes
    # Get configuration

    # # Get the user ID from the config (STUDIO)
    # user_id = config["configurable"]["user_id"]
    configurable = configuration.Configuration.from_runnable_config(config)

    # Get the user ID from the config
    user_id = configurable.user_id

    # Retrieve memory from the store
    namespace = ("memory", user_id)
    key = "user_memory"
    existing_memory = store.get(namespace, key)

    # Format the memories for the system prompt
    if existing_memory and existing_memory.value:
        memory_dict = existing_memory.value
        formatted_memory = (
            f"Name: {memory_dict.get('user_name', 'Unknown')}\n"
            f"Interests: {', '.join(memory_dict.get('interests', []))}"
        )
    else:
        formatted_memory = None

    #########################################################################
    # Added Code: accept both state shapes
    #########################################################################
    if isinstance(state, Mapping) and "messages" in state:
        messages = state["messages"]
    else:
        messages = state.get("values", {}).get("messages", [])

    # Format the memory in the system prompt
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=formatted_memory)

    # Respond using memory as well as the chat history
    response = get_model().invoke([SystemMessage(content=system_msg)] + messages)
    # logger.info("Model response: %s", response)
    # return {"messages": response}
    return {"messages": [response]}  # added Line


def write_memory_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and save a memory to the store."""
    # added for STUDIO-273: to handle both dict and StateSnapshot state shapes
    # Get configuration

    # # Get the user ID from the config (STUDIO)
    # user_id = config["configurable"]["user_id"]
    configurable = configuration.Configuration.from_runnable_config(config)

    # Get the user ID from the config
    user_id = configurable.user_id

    # Retrieve existing memory from the store
    namespace = ("memory", user_id)
    key = "user_memory"
    existing_memory = store.get(namespace, key)

    # Format the memories for the system prompt
    if existing_memory and existing_memory.value:
        memory_dict = existing_memory.value
        formatted_memory = (
            f"Name: {memory_dict.get('user_name', 'Unknown')}\n"
            f"Interests: {', '.join(memory_dict.get('interests', []))}"
        )
    else:
        formatted_memory = None

    #########################################################################
    # Added Code: accept both state shapes
    #########################################################################
    if isinstance(state, Mapping) and "messages" in state:  # added Line
        messages = state["messages"]  # added Line
    else:  # added Line
        messages = state.get("values", {}).get("messages", [])  # added Line

    # Format the existing memory in the instruction
    system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=formatted_memory)

    model_with_structure = get_model().with_structured_output(UserProfile)

    # Invoke the model to produce structured output that matches the schema
    new_memory = model_with_structure.invoke(
        [SystemMessage(content=system_msg)] + messages
    )

    # Overwrite the existing use profile memory
    key = "user_memory"
    store.put(namespace, key, new_memory)

    return {}  # added Line


#######################################################
# Graph Definition
#######################################################
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
        graph = builder.compile(
            checkpointer=within_thread_memory, store=across_thread_memory
        )
        return graph, across_thread_memory

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


def test_call_model(across_thread_memory):
    # We supply a thread ID for short-term (within-thread) memory
    # We supply a user ID for long-term (across-thread) memory
    config = {"configurable": {"thread_id": "1", "user_id": "1"}}

    # We supply a thread ID for short-term (within-thread) memory
    # We supply a user ID for long-term (across-thread) memory
    config = {"configurable": {"thread_id": "1", "user_id": "1"}}

    # User input
    input_messages = [
        HumanMessage(
            content="Hi, my name is Matthew and I like to bike around San Jose."
        )
    ]
    # Run the graph
    for chunk in memory_graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    #######################################################
    # long-term memory: look at the memory saved to the store
    #######################################################
    # across_thread_memory = InMemoryStore()
    # Namespace for the memory to save
    user_id = "1"
    namespace = ("memory", user_id)
    existing_memory = across_thread_memory.get(namespace, "user_memory")

    if existing_memory is None:
        logger.info("No memory found yet for namespace=%s", namespace)
    else:
        logger.info("Existing memory dict=%s", existing_memory.dict())
        # existing_memory.value
        logger.info("Existing memory value=%s", existing_memory.value)

    #######################################################
    # Continue the conversation in a new thread with updated memory
    #######################################################
    # User input
    input_messages = [
        HumanMessage(
            content="Hi, my name is Lance and I like to bike around San Francisco and eat at bakeries."
        )
    ]

    # Run the graph
    for chunk in memory_graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    #######################################################
    # long-term memory: look at the memory saved to the store
    #######################################################
    # across_thread_memory = InMemoryStore()
    # Namespace for the memory to save
    user_id = "1"
    namespace = ("memory", user_id)
    existing_memory = across_thread_memory.get(namespace, "user_memory")

    if existing_memory is None:
        logger.info("No memory found yet for namespace=%s", namespace)
    else:
        logger.info("Existing memory dict=%s", existing_memory.dict())
        # existing_memory.value
        logger.info("Existing memory value=%s", existing_memory.value)


#######################################################
# When can this fail?
########################################################
class OutputFormat(BaseModel):
    preference: str
    sentence_preference_revealed: str


class TelegramPreferences(BaseModel):
    preferred_encoding: Optional[List[OutputFormat]] = None
    favorite_telegram_operators: Optional[List[OutputFormat]] = None
    preferred_telegram_paper: Optional[List[OutputFormat]] = None


class MorseCode(BaseModel):
    preferred_key_type: Optional[List[OutputFormat]] = None
    favorite_morse_abbreviations: Optional[List[OutputFormat]] = None


class Semaphore(BaseModel):
    preferred_flag_color: Optional[List[OutputFormat]] = None
    semaphore_skill_level: Optional[List[OutputFormat]] = None


class TrustFallPreferences(BaseModel):
    preferred_fall_height: Optional[List[OutputFormat]] = None
    trust_level: Optional[List[OutputFormat]] = None
    preferred_catching_technique: Optional[List[OutputFormat]] = None


class CommunicationPreferences(BaseModel):
    telegram: TelegramPreferences
    morse_code: MorseCode
    semaphore: Semaphore


class UserPreferences(BaseModel):
    communication_preferences: CommunicationPreferences
    trust_fall_preferences: TrustFallPreferences


class TelegramAndTrustFallPreferences(BaseModel):
    pertinent_user_preferences: UserPreferences


#######################################################
# Trustcall
########################################################
def test_can_fail(across_thread_memory):
    """
    ## When can this fail?
    [`with_structured_output`](https://docs.langchain.com/oss/python/langchain/models#structured-outputs) is very useful, but what happens if we're working with a more complex schema?
    [Here's](https://github.com/hinthornw/trustcall?tab=readme-ov-file#complex-schema) an example of a more complex schema, which we'll test below.
    This is a [Pydantic](https://docs.pydantic.dev/latest/) model that describes a user's preferences for communication and trust fall.
    """

    # Now, let's try extraction of this schema using the `with_structured_output` method.

    from pydantic import ValidationError

    # Bind schema to model
    model_with_structure = get_model().with_structured_output(
        TelegramAndTrustFallPreferences
    )

    # Conversation
    conversation = """Operator: How may I assist with your telegram, sir?
    Customer: I need to send a message about our trust fall exercise.
    Operator: Certainly. Morse code or standard encoding?
    Customer: Morse, please. I love using a straight key.
    Operator: Excellent. What's your message?
    Customer: Tell him I'm ready for a higher fall, and I prefer the diamond formation for catching.
    Operator: Done. Shall I use our "Daredevil" paper for this daring message?
    Customer: Perfect! Send it by your fastest carrier pigeon.
    Operator: It'll be there within the hour, sir."""

    # Invoke the model
    try:
        result = model_with_structure.invoke(
            f"""Extract the preferences from the following conversation:
        <convo>
        {conversation}
        </convo>"""
        )
        # Extract the preferences
        # result["responses"][0]
        logger.info("Extracted preferences: %s", result)
    except ValidationError as e:
        print(e)

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
    test_call_flag = False
    test_can_fail_flag = True
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
            memory_graph, across_thread_memory = generate_memory_graph(
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
                output_name="long_memory_graph.png",
            )
            test_call_model(across_thread_memory)
        if test_can_fail_flag:
            memory_graph, across_thread_memory = generate_memory_graph(
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
                output_name="can_fail_graph.png",
            )
            test_can_fail(across_thread_memory)

    logger.info("All done.")


"""
## Viewing traces in LangSmith
We can see that the memories are retrieved from the store and supplied as part of the system prompt, as expected:
https://smith.langchain.com/

## Studio
We can also interact with our chatbot in Studio.

"""
