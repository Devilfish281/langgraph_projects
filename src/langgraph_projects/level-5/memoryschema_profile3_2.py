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


#######################################################
# Chatbot instruction
#######################################################
# MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user.
# If you have memory for this user, use it to personalize your responses.
# Here is the memory (it may be empty): {memory}"""

# # Create new memory from the chat history and any existing memory
# CREATE_MEMORY_INSTRUCTION = """Create or update a user profile memory based on the user's chat history.
# This will be saved for long-term memory. If there is an existing memory, simply update it.
# Here is the existing memory (it may be empty): {memory}"""


# Schema
# Profile of a user
class UserProfile(BaseModel):
    user_name: str = Field(description="The user's preferred name")
    user_location: str = Field(description="The user's location")
    interests: list[str] = Field(description="A list of the user's interests")


# Create the extractor
trustcall_extractor = create_extractor(
    get_model(),
    tools=[UserProfile],
    tool_choice="UserProfile",  # Enforces use of the UserProfile tool
)

# Chatbot instruction
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user.
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""

# Extraction instruction
TRUSTCALL_INSTRUCTION = """Create or update the memory (JSON doc) to incorporate information from the following conversation:"""


def call_model_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Load memory from the store and use it to personalize the chatbot's response."""
    # added for STUDIO-273: to handle both dict and StateSnapshot state shapes
    # Get configuration

    # # Get the user ID from the config
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
            f"Location: {memory_dict.get('user_location', 'Unknown')}\n"
            f"Interests: {', '.join(memory_dict.get('interests', []))}"
        )
    else:
        formatted_memory = None
    ###########################################################################
    # # Extract the actual memory content if it exists and add a prefix
    # if existing_memory:
    #     # Value is a dictionary with a memory key
    #     existing_memory_content = existing_memory.value.get("memory")
    # else:
    #     existing_memory_content = "No existing memory found."

    #########################################################################
    # Added Code: accept both state shapes
    #########################################################################
    if isinstance(state, Mapping) and "messages" in state:
        messages = state["messages"]
    else:
        messages = state.get("values", {}).get("messages", [])

    logger.info(
        "call_model_node messages_type=%s count=%s", type(messages), len(messages)
    )  # Added Code

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

    # # Get the user ID from the config
    # user_id = config["configurable"]["user_id"]
    configurable = configuration.Configuration.from_runnable_config(config)

    # Get the user ID from the config
    user_id = configurable.user_id

    # Retrieve existing memory from the store
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")

    # Get the profile as the value from the list, and convert it to a JSON doc
    existing_profile = (
        {"UserProfile": existing_memory.value} if existing_memory else None
    )

    # User input
    input_messages = [
        HumanMessage(
            content="Hi, my name is Matthew and I like to bike around San Jose and eat at bakeries."
        )
    ]

    #########################################################################
    # Added Code: accept both state shapes
    #########################################################################
    if isinstance(state, Mapping) and "messages" in state:  # added Line
        messages = state["messages"]  # added Line
    else:  # added Line
        messages = state.get("values", {}).get("messages", [])  # added Line

    # Invoke the extractor
    result = trustcall_extractor.invoke(
        {
            "messages": [SystemMessage(content=TRUSTCALL_INSTRUCTION)] + messages,
            "existing": existing_profile,
        }
    )

    # Get the updated profile as a JSON object
    updated_profile = result["responses"][0].model_dump()

    # Save the updated profile
    key = "user_memory"
    store.put(namespace, key, updated_profile)

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


def test_call_trustcall_model(across_thread_memory):

    # We supply a thread ID for short-term (within-thread) memory
    # We supply a user ID for long-term (across-thread) memory
    config = {"configurable": {"thread_id": "1", "user_id": "1"}}

    # User input
    input_messages = [HumanMessage(content="Hi, my name is Matthew.")]

    # Run the graph
    for chunk in memory_graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
    ########################################################################
    # Second turn of the conversation to generate memory
    ##########################################################################
    # User input
    input_messages = [HumanMessage(content="I like to bike around San Jose")]

    # Run the graph
    for chunk in memory_graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    ########################################################################
    # Save memory and check the store
    #########################################################################
    # Namespace for the memory to save
    user_id = "1"
    namespace = ("memory", user_id)
    existing_memory = across_thread_memory.get(namespace, "user_memory")
    # existing_memory.dict()
    logger.info("Existing memory dict=%s", existing_memory.dict())

    # The user profile saved as a JSON object
    # existing_memory.value
    logger.info("Existing memory value=%s", existing_memory.value)

    # User input
    input_messages = [HumanMessage(content="I also enjoy going to bakeries")]

    # Run the graph
    for chunk in memory_graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    """Continue the conversation in a new thread."""

    # We supply a thread ID for short-term (within-thread) memory
    # We supply a user ID for long-term (across-thread) memory
    config = {"configurable": {"thread_id": "2", "user_id": "1"}}

    # User input
    input_messages = [HumanMessage(content="What bakeries do you recommend for me?")]

    # Run the graph
    for chunk in memory_graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    # Namespace for the memory to save
    user_id = "1"
    namespace = ("memory", user_id)
    existing_memory = across_thread_memory.get(namespace, "user_memory")
    # existing_memory.dict()
    logger.info("Existing memory dict=%s", existing_memory.dict())
    # The user profile saved as a JSON object
    logger.info("Existing memory value=%s", existing_memory.value)
    """Trace:
    https://smith.langchain.com/
    ## Studio
    """


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
def test_trustcall(across_thread_memory):

    from pydantic import ValidationError

    """If we naively extract more complex schemas, even using high capacity model like `gpt-4o`, it is prone to failure.
    ## Trustcall for creating and updating profile schemas
    As we can see, working with schemas can be tricky.
    Complex schemas can be difficult to extract.
    In addition, updating even simple schemas can pose challenges.
    Consider our above chatbot.
    We regenerated the profile schema *from scratch* each time we chose to save a new memory.
    This is inefficient, potentially wasting model tokens if the schema contains a lot of information to re-generate each time.
    Worse, we may loose information when regenerating the profile from scratch.
    Addressing these problems is the motivation for [TrustCall](https://github.com/hinthornw/trustcall)!
    This is an open-source library for updating JSON schemas developed by one [Will Fu-Hinthorn](https://github.com/hinthornw) on the LangChain team.
    It's motivated by exactly these challenges while working on memory.
    Let's first show simple usage of extraction with TrustCall on this list of [messages](https://docs.langchain.com/oss/python/langchain/messages).
    """

    # Conversation
    conversation = [
        HumanMessage(content="Hi, I'm Matthew."),
        AIMessage(content="Nice to meet you, Matthew."),
        HumanMessage(content="I really like biking around San Jose."),
    ]

    """We use `create_extractor`, passing in the model as well as our schema as a [tool](https://docs.langchain.com/oss/python/langchain/tools).
    With TrustCall, can supply supply the schema in various ways.
    For example, we can pass a JSON object / Python dictionary or Pydantic model.
    Under the hood, TrustCall uses [tool calling](https://docs.langchain.com/oss/python/langchain/models#tool-calling) to produce [structured output](https://docs.langchain.com/oss/python/langchain/models#structured-outputs) from an input list of [messages](https://docs.langchain.com/oss/python/langchain/messages).
    To force Trustcall to produce structured output, we can include the schema name in the `tool_choice` argument.
    We can invoke the extractor with  the above conversation.
    """
    from trustcall import create_extractor

    # Schema
    # User profile schema with typed fields
    class UserProfile(BaseModel):
        user_name: str = Field(description="The user's preferred name")
        interests: List[str] = Field(description="A list of the user's interests")

    # Create the extractor
    trustcall_extractor = create_extractor(
        get_model(), tools=[UserProfile], tool_choice="UserProfile"
    )

    # Instruction
    system_msg = "Extract the user profile from the following conversation"

    # Invoke the extractor
    result = trustcall_extractor.invoke(
        {"messages": [SystemMessage(content=system_msg)] + conversation}
    )

    """
    When we invoke the extractor, we get a few things:
    * `messages`: The list of `AIMessages` that contain the tool calls.
    * `responses`: The resulting parsed tool calls that match our schema.
    * `response_metadata`: Applicable if updating existing tool calls. It says which of the responses correspond to which of the existing objects.
    """

    for m in result["messages"]:
        m.pretty_print()

    schema = result["responses"]
    # schema
    logger.info("Extracted schema: %s", schema)

    # schema[0].model_dump()
    logger.info("Extracted schema as dict: %s", schema[0].model_dump())

    # result["response_metadata"]
    logger.info("Response metadata: %s", result["response_metadata"])

    """
    Let's see how we can use it to *update* the profile.
    For updating, TrustCall takes a set of messages as well as the existing schema.
    The central idea is that it prompts the model to produce a [JSON Patch](https://jsonpatch.com/) to update only the relevant parts of the schema.
    This is less error-prone than naively overwriting the entire schema.
    It's also more efficient since the model only needs to generate the parts of the schema that have changed.
    We can save the existing schema as a dict.
    We can use `model_dump()` to serialize a Pydantic model instance into a dict.
    We pass it to the `"existing"` argument along with the schema name, `UserProfile`.
    """
    # Update the conversation
    updated_conversation = [
        HumanMessage(content="Hi, I'm Matthew."),
        AIMessage(content="Nice to meet you, Matthew."),
        HumanMessage(content="I really like biking around San Jose."),
        AIMessage(content="San Jose is a great city! Where do you go after biking?"),
        HumanMessage(content="I really like to go to a bakery after biking."),
    ]

    # Update the instruction
    system_msg = f"""Update the memory (JSON doc) to incorporate new information from the following conversation"""

    # Invoke the extractor with the updated instruction and existing profile with the corresponding tool name (UserProfile)
    result = trustcall_extractor.invoke(
        {"messages": [SystemMessage(content=system_msg)] + updated_conversation},
        {"existing": {"UserProfile": schema[0].model_dump()}},
    )

    for m in result["messages"]:
        m.pretty_print()

    # result["response_metadata"]
    logger.info("Response metadata for update: %s", result["response_metadata"])

    updated_schema = result["responses"][0]
    # updated_schema.model_dump()
    logger.info("Updated schema: %s", updated_schema.model_dump())

    """
    LangSmith trace:
    https://smith.langchain.com/public/229eae22-1edb-44c6-93e6-489124a43968/r
    Now, let's also test Trustcall on the [challenging schema](https://github.com/hinthornw/trustcall?tab=readme-ov-file#complex-schema) that we saw earlier.
    """
    #########################################################
    # Test on more complex schema
    #########################################################
    bound = create_extractor(
        get_model(),
        tools=[TelegramAndTrustFallPreferences],
        tool_choice="TelegramAndTrustFallPreferences",
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

    result = bound.invoke(
        f"""Extract the preferences from the following conversation:
    <convo>
    {conversation}
    </convo>"""
    )

    # Extract the preferences
    # result["responses"][0]
    logger.info("Extracted preferences with TrustCall: %s", result["responses"][0])

    """
    Trace:
    https://smith.langchain.com/public/5cd23009-3e05-4b00-99f0-c66ee3edd06e/r
    For more examples, you can see an overview video [here](https://www.youtube.com/watch?v=-H4s0jQi-QY).
    ## Chatbot with profile schema updating
    Now, let's bring Trustcall into our chatbot to create *and update* a memory profile.
    """

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
    test_trustcall_flag = False
    test_call_trustcall_flag = True
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

        if test_trustcall_flag:
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
                output_name="trustcall_graph.png",
            )
            test_trustcall(across_thread_memory)
        if test_call_trustcall_flag:
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
                output_name="call_trustcall_graph.png",
            )
            test_call_trustcall_model(across_thread_memory)
    logger.info("All done.")


"""
## Viewing traces in LangSmith
We can see that the memories are retrieved from the store and supplied as part of the system prompt, as expected:
https://smith.langchain.com/

## Studio
We can also interact with our chatbot in Studio.

"""
