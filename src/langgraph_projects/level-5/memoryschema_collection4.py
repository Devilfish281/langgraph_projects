# src/langgraph_projects/level-5/memoryschema_collection4.py
"""
Sometimes we want to save memories to a [collection](https://docs.google.com/presentation/d/181mvjlgsnxudQI6S3ritg9sooNyu4AcLLFH1UK0kIuk/edit#slide=id.g30eb3c8cf10_0_200) rather than single profile.
Here we'll update our chatbot to [save memories to a collection](https://docs.langchain.com/oss/python/concepts/memory#collection).
We'll also show how to use Trustcall to update this collection.
"""

# %pip install -U langchain_openai langgraph trustcall langchain_core

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
from xml.parsers.expat import model

from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string,
    merge_message_runs,
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

#######################################################
# Defining a collection schema
#######################################################
"""
Instead of storing user information in a fixed profile structure, we'll create a flexible collection schema to store memories about user interactions.
Each memory will be stored as a separate entry with a single `content` field for the main information we want to remember
This approach allows us to build an open-ended collection of memories that can grow and change as we learn more about the user.
We can define a collection schema as a [Pydantic](https://docs.pydantic.dev/latest/) object.
"""


# from pydantic import BaseModel, Field
class Memory(BaseModel):
    content: str = Field(
        description="The main content of the memory. For example: User expressed interest in learning about French."
    )


class MemoryCollection(BaseModel):
    memories: list[Memory] = Field(description="A list of memories about the user.")


# Create the Trustcall extractor
trustcall_extractor = create_extractor(
    get_model(),
    tools=[Memory],
    tool_choice="Memory",
    # This allows the extractor to insert new memories
    enable_inserts=True,
)

# Chatbot instruction
MODEL_SYSTEM_MESSAGE = """You are a helpful chatbot. You are designed to be a companion to a user.
You have a long term memory which keeps track of information you learn about the user over time.
Current Memory (may include updated memories from this conversation):
{memory}"""

# Trustcall instruction
TRUSTCALL_INSTRUCTION = """Reflect on following interaction.
Use the provided tools to retain any necessary memories about the user.
Use parallel tool calling to handle updates and insertions simultaneously:"""


def call_model_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Load memory from the store and use it to personalize the chatbot's response."""
    # added for STUDIO-273: to handle both dict and StateSnapshot state shapes
    # Get configuration

    # # Get the user ID from the config
    # user_id = config["configurable"]["user_id"]
    configurable = configuration.Configuration.from_runnable_config(config)

    # Get the user ID from the config
    user_id = configurable.user_id
    ##################################################################
    # Retrieve memory from the store
    namespace = ("memory", user_id)

    existing_memory = store.search(namespace)

    # Format the memories for the system prompt
    info = (
        "\n".join(f"- {mem.value['content']}" for mem in existing_memory)
        if existing_memory
        else "No memories yet."
    )
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=info)

    #########################################################################
    # Added Code: accept both state shapes
    #########################################################################
    if isinstance(state, Mapping) and "messages" in state:
        messages = state["messages"]
    else:
        messages = state.get("values", {}).get("messages", [])

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
    ###################################################################
    # Retrieve existing memory from the store
    namespace = ("memory", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "Memory"
    existing_memories = (
        [
            (existing_item.key, tool_name, existing_item.value)
            for existing_item in existing_items
        ]
        if existing_items
        else None
    )

    if isinstance(state, Mapping) and "messages" in state:  # Added Code
        messages = state["messages"]  # Added Code
    else:  # Added Code
        messages = state.get("values", {}).get("messages", [])  # Added Code

    updated_messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION)]
            + messages  # Changed Code
        )
    )

    # Invoke the extractor
    result = trustcall_extractor.invoke(
        {"messages": updated_messages, "existing": existing_memories}
    )

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )

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


def test_call_trustcall_model(across_thread_memory=None):

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
    existing_memory = across_thread_memory.search(namespace)

    for m in existing_memory:
        logger.info("Existing memory dict=%s", m.dict())

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

    """Trace:
    https://smith.langchain.com/
    ## Studio
    """
    existing_memory = across_thread_memory.search(namespace)

    for m in existing_memory:
        logger.info("Existing memory dict=%s", m.dict())
    logger.info("Test call trustcall model successfully.")


#######################################################
# Trustcall
########################################################
def test_trustcall():
    logger.info("SECTION ONE.")
    # Bind schema to model
    model_with_structure = get_model().with_structured_output(MemoryCollection)

    # Invoke the model to produce structured output that matches the schema
    #######################################################################
    # invoke
    memory_collection = model_with_structure.invoke(
        [HumanMessage("My name is Matthew. I like to bike.")]
    )
    # memory_collection.memories
    logger.info("Extracted memory collection: %s", memory_collection.memories)

    """We can use `model_dump()` to serialize a Pydantic model instance into a Python dictionary."""

    a_memmory = memory_collection.memories[0].model_dump()
    logger.info("Serialized memory: %s", a_memmory)

    """Save dictionary representation of each memory to the store."""
    # Initialize the in-memory store
    in_memory_store = InMemoryStore()

    # Namespace for the memory to save
    user_id = "1"
    namespace_for_memory = (user_id, "memories")

    # # Save a memory to namespace as key and value
    # key = str(uuid.uuid4())
    # value = memory_collection.memories[0].model_dump()
    # in_memory_store.put(namespace_for_memory, key, value)

    # key = str(uuid.uuid4())
    # value = memory_collection.memories[1].model_dump()
    # in_memory_store.put(namespace_for_memory, key, value)

    memories = memory_collection.memories or []
    logger.info("Generated memories: count=%d", len(memories))

    for i, mem in enumerate(memories):
        key = str(uuid.uuid4())
        value = mem.model_dump()
        in_memory_store.put(namespace_for_memory, key, value)
        logger.info("Saved memory[%d] key=%s value=%s", i, key, value)

    """Search for memories in the store."""
    # Search
    for m in in_memory_store.search(namespace_for_memory):
        logger.info("Searched memory: %s", m.dict())

    ###################################################################
    # Updating collection schema
    ####################################################################
    """
    We discussed the challenges with updating a profile schema in the last lesson.
    The same applies for collections!
    We want the ability to update the collection with new memories as well as update existing memories in the collection.
    Now we'll show that [Trustcall](https://github.com/hinthornw/trustcall) can be also used to update a collection.
    This enables both addition of new memories as well as [updating existing memories in the collection](https://github.com/hinthornw/trustcall?tab=readme-ov-file#simultanous-updates--insertions
    ).
    Let's define a new extractor with Trustcall.
    As before, we provide the schema for each memory, `Memory`.  
    But, we can supply `enable_inserts=True` to allow the extractor to insert new memories to the collection.
    """
    logger.info("SECTION TWO.")
    # Create the extractor
    trustcall_extractor = create_extractor(
        get_model(),
        tools=[Memory],
        tool_choice="Memory",
        enable_inserts=True,
    )
    # Instruction
    # Changed Code
    system_msg = """Update existing memories and create new ones based on the following conversation.

    Rules:
    - Create ONE Memory tool call per distinct fact.
    - Do NOT combine unrelated facts into one Memory.
    - If there are 2 facts, you MUST make 2 separate Memory tool calls.
    """

    # Conversation
    conversation = [
        HumanMessage(content="Hi, I'm Matthew."),
        AIMessage(content="Nice to meet you, Matthew."),
        HumanMessage(content="This morning I had a nice bike ride in San Jose."),
    ]

    # Invoke the extractor
    #####################################################################
    # invoke - result will contain the extracted memories as well as metadata about the tool calls
    result = trustcall_extractor.invoke(
        {"messages": [SystemMessage(content=system_msg)] + conversation}
    )

    # # Messages contain the tool calls
    # logger.info("TrustCall messages:")
    # for m in result["messages"]:
    #     m.pretty_print()

    # logger.info("TrustCall responses:")
    # # Responses contain the memories that adhere to the schema
    # for m in result["responses"]:
    #     print(m)

    # logger.info("TrustCall response metadata:")
    # # Metadata contains the tool call
    # for m in result["response_metadata"]:
    #     print(m)

    # Messages contain the tool calls
    msgs = result.get("messages", [])
    logger.info("TrustCall messages: count=%d", len(msgs))
    for i, m in enumerate(msgs):
        m.pretty_print()

    # Responses contain the memories that adhere to the schema
    responses = result.get("responses", [])
    logger.info("TrustCall responses: count=%d", len(responses))
    for i, m in enumerate(responses):
        print(m)

    # Metadata contains the tool call info / mapping (e.g., json_doc_id)
    metas = result.get("response_metadata", [])
    logger.info("TrustCall response_metadata: count=%d", len(metas))
    for i, meta in enumerate(metas):
        print(meta)

    logger.info("SECTION THREE.")
    # Update the conversation
    updated_conversation = [
        AIMessage(content="That's great, did you do after?"),
        HumanMessage(content="I went to Tartine and ate a croissant."),
        AIMessage(content="What else is on your mind?"),
        HumanMessage(
            content="I was thinking about my Japan, and going back this winter!"
        ),
    ]

    # Update the instruction
    # Changed Code
    system_msg = """Update existing memories and create new ones based on the following conversation.

    Rules:
    - Create ONE Memory tool call per distinct fact.
    - Do NOT combine unrelated facts into one Memory.
    - If there are 2 facts, you MUST make 2 separate Memory tool calls.
    """
    # We'll save existing memories, giving them an ID, key (tool name), and value
    tool_name = "Memory"
    prior_responses = result.get("responses", [])  # added Line
    existing_memories = (  # Changed Code
        [
            (str(i), tool_name, memory.model_dump())
            for i, memory in enumerate(prior_responses)
        ]  # added Line
        if prior_responses  # added Line
        else None  # added Line
    )
    # existing_memories
    logger.info("Existing memories for update: %s", existing_memories)

    # Changed Code
    update_messages = [
        SystemMessage(content=system_msg)
    ] + updated_conversation  # Added Code

    # Invoke the extractor with our updated conversation and existing memories
    #######################################################################
    # invoke - result will contain the extracted memories as well as metadata about the tool calls
    result = trustcall_extractor.invoke(
        {"messages": update_messages, "existing": existing_memories}  # Changed Code
    )

    # Messages from the model indicate two tool calls were made
    logger.info("TrustCall messages: count=%d", len(result.get("messages", [])))
    for i, m in enumerate(result.get("messages", [])):
        m.pretty_print()

    # Responses contain the memories that adhere to the schema
    logger.info("TrustCall responses: count=%d", len(result.get("responses", [])))
    for i, m in enumerate(result.get("responses", [])):
        print(m)

    # This tells us that we updated the first memory in the collection by specifying the `json_doc_id`.

    # Metadata contains the tool call
    logger.info(
        "TrustCall response_metadata: count=%d",
        len(result.get("response_metadata", [])),
    )
    for i, meta in enumerate(result.get("response_metadata", [])):
        print(meta)

    """
    LangSmith trace:
    https://smith.langchain.com/
    ## Chatbot with collection schema updating
    Now, let's bring Trustcall into our chatbot to create and update a memory collection.
    """

    logger.info("Test trustcall successfully.")


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
    test_trustcall_flag = True
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
            test_trustcall()

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
