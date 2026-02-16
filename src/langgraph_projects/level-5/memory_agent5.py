# src/langgraph_projects/level-5/memory_agent5.py
"""
Now, we're going to pull together the pieces we've learned to build an agent with long-term memory.
Our agent, `task_mAIstro`, will help us manage a ToDo list!
The chatbots we built previously *always* reflected on the conversation and saved memories.
`task_mAIstro` will decide *when* to save memories (items to our ToDo list).
The chatbots we built previously always saved one type of memory, a profile or a collection.
`task_mAIstro` can decide to save to either a user profile or a collection of ToDo items.
In addition semantic memory, `task_mAIstro` also will manage procedural memory.
This allows the user to update their preferences for creating ToDo items.
"""

# %pip install -U langchain_openai langgraph trustcall langchain_core

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
from datetime import datetime

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


######################################################################
# trustcall helper functions
######################################################################
######################################################################
def merge_existing_and_new(existing_memories, result):
    """
    Returns a combined list of Memory dicts:
      - keeps old ones
      - appends new ones
    (Does NOT handle “update-in-place” unless you add id mapping logic.)
    """
    existing_values = []
    if existing_memories:
        # existing_memories items look like: (id, "Memory", {"content": ...})
        existing_values = [val for (_id, _tool, val) in existing_memories]

    new_values = [m.model_dump() for m in (result.get("responses") or [])]

    return existing_values + new_values


def log_trustcall_result(  # added Line
    result: Mapping[str, Any],  # added Line
    logger: Any,  # added Line
    *,  # added Line
    label: str = "TrustCall",  # added Line
    show_messages: bool = True,  # added Line
    show_responses: bool = True,  # added Line
    show_metadata: bool = True,  # added Line
) -> None:  # added Line
    """Pretty-print Trustcall extractor output (messages / responses / metadata)."""  # added Line

    if show_messages:  # added Line
        msgs = result.get("messages", []) or []  # added Line
        logger.info("%s messages: count=%d", label, len(msgs))  # added Line
        for i, m in enumerate(msgs):  # added Line
            try:  # added Line
                m.pretty_print()  # added Line
            except Exception:  # added Line
                logger.info("%s message[%d]: %r", label, i, m)  # added Line

    if show_responses:  # added Line
        responses = result.get("responses", []) or []  # added Line
        logger.info("%s responses: count=%d", label, len(responses))  # added Line
        for i, m in enumerate(responses):  # added Line
            logger.info("%s response[%d]: %s", label, i, m)  # added Line

    if show_metadata:  # added Line
        metas = result.get("response_metadata", []) or []  # added Line
        logger.info("%s response_metadata: count=%d", label, len(metas))  # added Line
        for i, meta in enumerate(metas):  # added Line
            logger.info("%s meta[%d]: %s", label, i, meta)  # added Line


#####################################################################
### END
#####################################################################

#####################################################################
### Visibility into Trustcall updates
#####################################################################

"""
Trustcall creates and updates JSON schemas.
What if we want visibility into the *specific changes* made by Trustcall?
For example, we saw before that Trustcall has some of its own tools to:
* Self-correct from validation failures -- [see trace example here](https://smith.langchain.com/public/5cd23009-3e05-4b00-99f0-c66ee3edd06e/r/9684db76-2003-443b-9aa2-9a9dbc5498b7)
* Update existing documents -- [see trace example here](https://smith.langchain.com/public/f45bdaf0-6963-4c19-8ec9-f4b7fe0f68ad/r/760f90e1-a5dc-48f1-8c34-79d6a3414ac3)
Visibility into these tools can be useful for the agent we're going to build.
"""

# from pydantic import BaseModel, Field


class Memory(BaseModel):
    content: str = Field(
        description="The main content of the memory. For example: User expressed interest in learning about French."
    )


class MemoryCollection(BaseModel):
    memories: list[Memory] = Field(description="A list of memories about the user.")


"""
We can add a [listener](https://python.langchain.com/docs/how_to/lcel_cheatsheet/#add-lifecycle-listeners) <!-- broken, but cannot find better linke --> to the Trustcall extractor.
This will pass runs from the extractor's execution to a class, `Spy`, that we will define.
Our `Spy` class will extract information about what tool calls were made by Trustcall.
"""

# from trustcall import create_extractor
# from langchain_openai import ChatOpenAI


# Inspect the tool calls made by Trustcall
class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        # Collect information about the tool calls made by the extractor.
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )


# Initialize the spy
spy = Spy()


#######################################################
# Helper function to extract tool call information from Trustcall runs
# This function processes the tool calls made by Trustcall during the extraction process.
# It identifies calls to the "PatchDoc" tool (used for updating existing documents) and the schema tool (e.g., "Memory") for new document creation.
# The function compiles a list of changes, categorizing them as updates or new creations, and formats this information into a readable string output.
# The output includes details about document updates (planned edits and added content) and new document creations, providing insights into the specific modifications made by Trustcall.
#########################################################
# New check for empyt patches + better formatting of results
def extract_tool_info(tool_calls, schema_name="Memory"):
    """Extract information from tool calls for both patches and new memories.

    Args:
        tool_calls: List of tool calls from the model
        schema_name: Name of the schema tool (e.g., "Memory", "ToDo", "Profile")
    """
    # Initialize list of changes
    changes = []

    for call_group in tool_calls:
        for call in call_group:
            name = call.get("name")

            if name == "PatchDoc":
                args = call.get("args", {})
                patches = args.get("patches") or []  # added Line

                # Trustcall may emit PatchDoc with no patches when no changes are needed  # added Line
                patch_values = []  # added Line
                for p in patches:  # added Line
                    if isinstance(p, dict) and "value" in p:  # added Line
                        patch_values.append(p["value"])  # added Line

                changes.append(
                    {
                        "type": "update",
                        "doc_id": args.get("json_doc_id"),
                        "planned_edits": args.get("planned_edits", ""),
                        "values": patch_values,  # added Line (could be empty)
                    }
                )

            elif name == schema_name:
                changes.append({"type": "new", "value": call.get("args", {})})
    # Format results as a single string
    result_parts = []
    for change in changes:
        if change["type"] == "update":
            if change["values"]:
                added_content = "\n- " + "\n- ".join(
                    map(str, change["values"])
                )  # added Line
            else:
                added_content = "\n(no patches returned)"  # added Line

            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Added content: {added_content}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\nContent: {change['value']}"
            )

    return "\n\n".join(result_parts)


#######################################################
# Chatbot instruction
#######################################################
# Creating an agent
#######################################################
"""
There are many different agent architectures to choose from.
Here, we'll implement something simple, a [ReAct](https://docs.langchain.com/oss/python/langgraph/workflows-agents#agents) agent.
This agent will be a helpful companion for creating and managing a ToDo list.
This agent can make a decision to update three types of long-term memory:
(a) Create or update a user `profile` with general user information
(b) Add or update items in a ToDo list `collection`
(c) Update its own `instructions` on how to update items to the ToDo list
"""


# Update memory tool
# "Decision on what memory type to update"
class UpdateMemory(TypedDict):
    update_type: Literal["user", "todo", "instructions"]


#######################################################
# Graph definition
#######################################################
""" 
We add a simple router, `route_message`, that makes a binary decision to save memories.
The memory collection updating is handled by `Trustcall` in the `write_memory` node, as before!
"""


# User profile schema
# This is the profile of the user you are chatting with
class Profile(BaseModel):
    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    job: Optional[str] = Field(description="The user's job", default=None)
    connections: list[str] = Field(
        description="Personal connection of the user, such as family members, friends, or coworkers",
        default_factory=list,
    )
    interests: list[str] = Field(
        description="Interests that the user has", default_factory=list
    )


class ToDo(BaseModel):
    task: str = Field(description="The task to be completed.")
    time_to_complete: Optional[int] = Field(
        description="Estimated time to complete the task (minutes)."
    )
    deadline: Optional[datetime] = Field(
        description="When the task needs to be completed by (if applicable)",
        default=None,
    )
    solutions: list[str] = Field(
        description="List of specific, actionable solutions (e.g., specific ideas, service providers, or concrete options relevant to completing the task)",
        min_items=1,
        default_factory=list,
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        description="Current status of the task", default="not started"
    )


# Create the Trustcall extractor for updating the user profile
profile_extractor = create_extractor(
    get_model(),
    tools=[Profile],
    tool_choice="Profile",
)


#######################################################
# Prompts
#######################################################
# Chatbot instruction for choosing what to update and what tools to call
MODEL_SYSTEM_MESSAGE = """You are a helpful chatbot.

You are designed to be a companion to a user, helping them keep track of their ToDo list.

You have a long term memory which keeps track of three things:
1. The user's profile (general information about them)
2. The user's ToDo list
3. General instructions for updating the ToDo list

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>

Here is the current ToDo List (may be empty if no tasks have been added yet):
<todo>
{todo}
</todo>

Here are the current user-specified preferences for updating the ToDo list (may be empty if no preferences have been specified yet):
<instructions>
{instructions}
</instructions>

Here are your instructions for reasoning about the user's messages:

1. Reason carefully about the user's messages as presented below.

2. Decide whether any of the your long-term memory should be updated:
- If personal information was provided about the user, update the user's profile by calling UpdateMemory tool with type `user`
- If tasks are mentioned, update the ToDo list by calling UpdateMemory tool with type `todo`
- If the user has specified preferences for how to update the ToDo list, update the instructions by calling UpdateMemory tool with type `instructions`

3. Tell the user that you have updated your memory, if appropriate:
- Do not tell the user you have updated the user's profile
- Tell the user them when you update the todo list
- Do not tell the user that you have updated instructions

4. Err on the side of updating the todo list. No need to ask for explicit permission.

5. Respond naturally to user user after a tool call was made to save memories, or if no tool call was made."""

# Trustcall instruction
TRUSTCALL_INSTRUCTION = """Reflect on following interaction.

Use the provided tools to retain any necessary memories about the user.

Use parallel tool calling to handle updates and insertions simultaneously.

System Time: {time}"""

# Instructions for updating the ToDo list
CREATE_INSTRUCTIONS = """Reflect on the following interaction.

Based on this interaction, update your instructions for how to update ToDo list items.

Use any feedback from the user to update how they like to have items added, etc.

Your current instructions are:

<current_instructions>
{current_instructions}
</current_instructions>"""


#######################################################
# Node definitions
#######################################################
def task_mAIstro_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Load memories from the store and use them to personalize the chatbot's response."""

    # Get the user ID from the config
    # user_id = config["configurable"]["user_id"]
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Retrieve profile memory from the store
    namespace = ("profile", user_id)
    memories = store.search(namespace)
    if memories:
        user_profile = memories[0].value
    else:
        user_profile = None

    # Retrieve task memory from the store
    namespace = ("todo", user_id)
    memories = store.search(namespace)
    todo = "\n".join(f"{mem.value}" for mem in memories)

    # Retrieve custom instructions
    namespace = ("instructions", user_id)
    memories = store.search(namespace)
    if memories:
        instructions = memories[0].value
    else:
        instructions = ""

    system_msg = MODEL_SYSTEM_MESSAGE.format(
        user_profile=user_profile, todo=todo, instructions=instructions
    )

    # Respond using memory as well as the chat history
    response = (
        get_model()
        .bind_tools([UpdateMemory], parallel_tool_calls=False)
        .invoke([SystemMessage(content=system_msg)] + state["messages"])
    )

    return {"messages": [response]}


def update_profile_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update the memory collection."""
    """
    super simple explation
        Load existing profile from store
        Send conversation + existing profile to Trustcall
        Trustcall returns an updated Profile
        Save Profile back into the store
        Reply to the tool call so the agent can continue    
    """
    # Get the user ID from the config
    # user_id = config["configurable"]["user_id"]
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Define the namespace for the memories
    namespace = ("profile", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "Profile"
    existing_memories = (
        [
            (existing_item.key, tool_name, existing_item.value)
            for existing_item in existing_items
        ]
        if existing_items
        else None
    )

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(
        time=datetime.now().isoformat()
    )
    updated_messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)]
            + state["messages"][:-1]
        )
    )

    # Invoke the extractor
    result = profile_extractor.invoke(
        {"messages": updated_messages, "existing": existing_memories}
    )

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )
    tool_calls = state["messages"][-1].tool_calls
    # Return a Tool response to the original UpdateMemory call
    return {
        "messages": [
            {
                "role": "tool",
                "content": "updated profile",
                "tool_call_id": tool_calls[0]["id"],
            }
        ]
    }


def update_todos_node(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update the memory collection."""
    """
    simple examplation:
        Load old ToDos from memory store
        Ask Trustcall: “Given the recent chat, update these ToDos or add new ones”
        Save the updated ToDos back into memory store
        Return a tool-result message so the chatbot can continue    
    """
    # Get the user ID from the config
    # user_id = config["configurable"]["user_id"]
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Define the namespace for the memories
    namespace = ("todo", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "ToDo"
    existing_memories = (
        [
            (existing_item.key, tool_name, existing_item.value)
            for existing_item in existing_items
        ]
        if existing_items
        else None
    )

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(
        time=datetime.now().isoformat()
    )
    updated_messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)]
            + state["messages"][:-1]
        )
    )

    # Initialize the spy for visibility into the tool calls made by Trustcall
    spy = Spy()

    # Create the Trustcall extractor for updating the ToDo list
    todo_extractor = create_extractor(
        get_model(), tools=[ToDo], tool_choice=tool_name, enable_inserts=True
    ).with_listeners(on_end=spy)

    # Invoke the extractor
    result = todo_extractor.invoke(
        {"messages": updated_messages, "existing": existing_memories}
    )

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )

    # Respond to the tool call made in task_mAIstro, confirming the update
    tool_calls = state["messages"][-1].tool_calls

    # Extract the changes made by Trustcall and add the the ToolMessage returned to task_mAIstro
    todo_update_msg = extract_tool_info(spy.called_tools, tool_name)
    return {
        "messages": [
            {
                "role": "tool",
                "content": todo_update_msg,
                "tool_call_id": tool_calls[0]["id"],
            }
        ]
    }


def update_instructions_node(
    state: MessagesState, config: RunnableConfig, store: BaseStore
):
    """Reflect on the chat history and update the memory collection."""
    """
    Why you “lose” the old instructions (and why that’s okay)
    This node intentionally does overwrite behavior:
        It reads old instructions (optional)
        Generates new ones
        Saves new ones under the same key "user_instructions"
    So you don't keep versions unless you add versioning yourself (example: store under timestamps or incrementing keys).    
    """
    # Get the user ID from the config
    # user_id = config["configurable"]["user_id"]
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    namespace = ("instructions", user_id)

    existing_memory = store.get(namespace, "user_instructions")

    # Format the memory in the system prompt
    system_msg = CREATE_INSTRUCTIONS.format(
        current_instructions=existing_memory.value if existing_memory else None
    )
    new_memory = get_model().invoke(
        [SystemMessage(content=system_msg)]
        + state["messages"][:-1]
        + [
            HumanMessage(
                content="Please update the instructions based on the conversation"
            )
        ]
    )

    # Overwrite the existing memory in the store
    key = "user_instructions"
    store.put(namespace, key, {"memory": new_memory.content})
    tool_calls = state["messages"][-1].tool_calls
    # returns a ToolMessage-like payload back into the graph.
    return {
        "messages": [
            {
                "role": "tool",
                "content": "updated instructions",
                "tool_call_id": tool_calls[0]["id"],
            }
        ]
    }


#######################################################
# Conditional edge
#######################################################
# Conditional edge
def route_message(
    state: MessagesState, config: RunnableConfig, store: BaseStore
) -> Literal[END, "update_todos", "update_instructions", "update_profile"]:
    """Reflect on the memories and chat history to decide whether to update the memory collection."""
    message = state["messages"][-1]
    if len(message.tool_calls) == 0:
        return END
    else:
        tool_call = message.tool_calls[0]
        if tool_call["args"]["update_type"] == "user":
            return "update_profile"
        elif tool_call["args"]["update_type"] == "todo":
            return "update_todos"
        elif tool_call["args"]["update_type"] == "instructions":
            return "update_instructions"
        else:
            raise ValueError


# def route_message(
#     state: MessagesState, config: RunnableConfig, store: BaseStore
# ) -> Literal[END, "update_todos", "update_instructions", "update_profile"]:

#     tool_calls = getattr(message, "tool_calls", None) or []
#     if not tool_calls:
#         return END
#     tool_call = tool_calls[0]
#     # Reflect on the memories and chat history to decide whether to update the memory collection.
#     message = state["messages"][-1]
#     if len(message.tool_calls) == 0:
#         return END
#     else:
#         # tool_call = message.tool_calls[0]
#         if tool_call["args"]["update_type"] == "user":
#             return "update_profile"
#         elif tool_call["args"]["update_type"] == "todo":
#             return "update_todos"
#         elif tool_call["args"]["update_type"] == "instructions":
#             return "update_instructions"
#         else:
#             raise ValueError(
#                 f"Unknown update_type: {tool_call.get('args', {}).get('update_type')}"
#             )


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
    builder.add_node("task_mAIstro", task_mAIstro_node)
    builder.add_node("update_todos", update_todos_node)
    builder.add_node("update_profile", update_profile_node)
    builder.add_node("update_instructions", update_instructions_node)

    builder.add_edge(START, "task_mAIstro")
    builder.add_conditional_edges("task_mAIstro", route_message)
    builder.add_edge("update_todos", "task_mAIstro")
    builder.add_edge("update_profile", "task_mAIstro")
    builder.add_edge("update_instructions", "task_mAIstro")

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
"""langsmith STUDIO Raw Mode
Set user_id to "Matthew" 
{
  "messages": [
    { "role": "user", "content": "My name is Matthew. I live in SF with my wife. I have a 1 year old daughter." }
  ]
}

{
  "messages": [
    { "role": "user", "content": "My wife asked me to book swim lessons for the baby." }
  ]
}

{
  "messages": [
    { "role": "user", "content": "When creating or updating ToDo items, include specific local businesses / vendors." }
  ]
}

{
  "messages": [
    { "role": "user", "content": "I need to fix the jammed electric Yale lock on the door." }
  ]
}


{
  "messages": [
    { "role": "user", "content": "For the swim lessons, I need to get that done by end of November." }
  ]
}

"""


#######################################################
# Test function to call the memory_graph and see the memory updates in action
#######################################################
def test_call_mAIstro_model(across_thread_memory=None):

    # logger.info("PatchDoc args: %s", call.get("args", {}))  # added Line

    logger.info(" starting test_call_mAIstro_model.")
    # We supply a thread ID for short-term (within-thread) memory
    # We supply a user ID for long-term (across-thread) memory
    config = {"configurable": {"thread_id": "1", "user_id": "Matthew"}}

    # User input to create a profile memory
    input_messages = [
        HumanMessage(
            content="My name is Matthew. I live in SF with my wife. I have a 1 year old daughter."
        )
    ]

    # Run the graph
    logger.info("Running the graph with input to create a profile memory.")
    for chunk in memory_graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
    print("\n" + "---" * 25)

    # User input for a ToDo
    input_messages = [
        HumanMessage(content="My wife asked me to book swim lessons for the baby.")
    ]

    # Run the graph
    logger.info("Running the graph with input for a ToDo.")
    for chunk in memory_graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
    print("\n" + "---" * 25)

    # User input to update instructions for creating ToDos
    input_messages = [
        HumanMessage(
            content="When creating or updating ToDo items, include specific local businesses / vendors."
        )
    ]

    # Run the graph
    logger.info(
        "Running the graph with input to update instructions for creating ToDos."
    )
    for chunk in memory_graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
    print("\n" + "---" * 25)

    # Check for updated instructions
    user_id = "Matthew"

    # Search
    print("\n" + "---" * 25)
    logger.info("Searching across-thread memory for instructions for user_id Matthew.")
    for memory in across_thread_memory.search(("instructions", user_id)):
        print(memory.value)
    print("\n" + "---" * 25)

    # User input for a ToDo
    input_messages = [
        HumanMessage(content="I need to fix the jammed electric Yale lock on the door.")
    ]

    # Run the graph
    logger.info("Running the graph with input for a ToDo.")
    for chunk in memory_graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
    print("\n" + "---" * 25)

    # Namespace for the memory to save
    user_id = "Matthew"

    # Search
    print("\n" + "---" * 25)
    logger.info("Searching across-thread memory for ToDos for user_id Matthew.")
    for memory in across_thread_memory.search(("todo", user_id)):
        print(memory.value)
    print("\n" + "---" * 25)

    # User input to update an existing ToDo
    input_messages = [
        HumanMessage(
            content="For the swim lessons, I need to get that done by end of November."
        )
    ]

    # Run the graph
    logger.info("Running the graph with input to update an existing ToDo.")
    for chunk in memory_graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
    print("\n" + "---" * 25)

    """We can see that Trustcall performs patching of the existing memory:
    https://smith.langchain.com/public/4ad3a8af-3b1e-493d-b163-3111aa3d575a/r
    """

    # User input for a ToDo
    input_messages = [
        HumanMessage(content="Need to call back City Toyota to schedule car service.")
    ]

    # Run the graph
    logger.info("Running the graph with input for a ToDo.")
    for chunk in memory_graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
    print("\n" + "---" * 25)

    # Namespace for the memory to save
    user_id = "Matthew"

    # Search
    print("\n" + "---" * 25)
    logger.info("Searching across-thread memory for ToDos for user_id Matthew.")
    for memory in across_thread_memory.search(("todo", user_id)):
        print(memory.value)
    print("\n" + "---" * 25)

    """Now we can create a new thread.

    This creates a new session.

    Profile, ToDos, and Instructions saved to long-term memory are accessed.
    """

    # We supply a thread ID for short-term (within-thread) memory
    # We supply a user ID for long-term (across-thread) memory
    config = {"configurable": {"thread_id": "2", "user_id": "Matthew"}}

    # Chat with the chatbot
    input_messages = [
        HumanMessage(content="I have 30 minutes, what tasks can I get done?")
    ]

    # Run the graph
    logger.info(
        "Running the graph with input to chat with the chatbot in a new thread."
    )
    for chunk in memory_graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
    print("\n" + "---" * 25)

    # Chat with the chatbot
    input_messages = [
        HumanMessage(content="Yes, give me some options to call for swim lessons.")
    ]

    # Run the graph
    logger.info(
        "Running the graph with input to chat with the chatbot in a new thread."
    )
    for chunk in memory_graph.stream(
        {"messages": input_messages}, config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
    print("\n" + "---" * 25)

    """Trace:

    https://smith.langchain.com/public/84768705-be91-43e4-8a6f-f9d3cee93782/r

    ## Studio

    ![Screenshot 2024-11-04 at 1.00.19 PM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/6732cfb05d9709862eba4e6c_Screenshot%202024-11-11%20at%207.46.40%E2%80%AFPM.png)
    """

    logger.info("Test mAIstro mode completed successfully.")


# from typing import Any, Mapping, Optional  # added Line


#######################################################
# Trustcall
########################################################
def test_trustcall():
    logger.info("SECTION ONE.")
    """
    First call: “extract from scratch”
    What this means:
    You pass only messages.
        Trustcall has no “existing memories” list to compare against.
        So it will mainly create new Memory tool calls from what it sees in conversation.
        Output structure typically includes:
            result["messages"] (contains the AIMessage with tool calls)
            result["responses"] (parsed Memory(...) objects)
            result["response_metadata"] (tool call ids, and sometimes insert/update ids depending on mode)    
    """
    # Create the extractor
    trustcall_extractor = create_extractor(
        get_model(),
        tools=[Memory],
        tool_choice="any",  # Changed Code (was "Memory")tool_choice="Memory",
        enable_inserts=True,
    )

    # Add the spy as a listener
    trustcall_extractor_see_all_tool_calls = trustcall_extractor.with_listeners(
        on_end=spy
    )

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    # Instruction
    # instruction = """Extract memories from the following conversation:"""
    system_msg = """Update existing memories and create new ones based on the following conversation.

    Rules:
    - You MUST create one Memory tool call for EACH distinct fact.
    - If there are 2 facts, you MUST make EXACTLY 2 Memory tool calls.
    - Facts in this conversation include food/places visited and travel plans.

    Example:
    Fact A: I ate a croissant at Tartine. -> Memory(content="User ate a croissant at Tartine.")
    Fact B: I plan to visit Japan this winter. -> Memory(content="User plans to visit Japan this winter.")
    """

    # Conversation
    conversation = [
        HumanMessage(content="Hi, I'm Matthew."),
        AIMessage(content="Nice to meet you, Matthew."),
        HumanMessage(content="I like to bike in San Jose."),
    ]

    # Invoke the extractor
    # it gives Trustcall a list of prior memories to update (or keep), and it can also insert new ones.

    #####################################################################
    # invoke -
    result = trustcall_extractor.invoke(
        {"messages": [SystemMessage(content=system_msg)] + conversation}
    )

    log_trustcall_result(result, logger, label="SECTION ONE")  # added Line

    logger.info("SECTION TWO.")
    # Update the conversation
    updated_conversation = [
        AIMessage(content="That's great, did you do after?"),
        HumanMessage(content="I went to Tartine and ate a croissant."),
        AIMessage(content="What else is on your mind?"),
        HumanMessage(
            content="I was thinking about my Japan, and going back this winter!"
        ),
    ]

    # We'll save existing memories, giving them an ID, key (tool name), and value
    tool_name = "Memory"
    existing_memories = (
        [
            (str(i), tool_name, memory.model_dump())
            for i, memory in enumerate(result["responses"])
        ]
        if result["responses"]
        else None
    )
    # existing_memories
    logger.info(f"Existing memories to update: {existing_memories}")

    # Invoke the extractor with our updated conversation and existing memories
    #####################################################################
    # invoke -
    # result = trustcall_extractor_see_all_tool_calls.invoke(
    #     {"messages": updated_conversation, "existing": existing_memories}
    # )
    result = trustcall_extractor_see_all_tool_calls.invoke(
        {
            "messages": [SystemMessage(content=system_msg)]
            + updated_conversation,  # Added Code
            "existing": existing_memories,
        }
    )
    log_trustcall_result(result, logger, label="SECTION TWO")

    all_memories_after_section_two = merge_existing_and_new(existing_memories, result)
    logger.info("ALL memories after SECTION TWO: %s", all_memories_after_section_two)

    #####################################################################
    #  Inspect the tool calls made by Trustcall using SPY
    logger.info("SECTION THREE.")
    # spy.called_tools
    logger.info(f"Tool calls made by Trustcall (raw): {spy.called_tools}")

    # Inspect spy.called_tools to see exactly what happened during the extraction
    schema_name = "Memory"
    changes = extract_tool_info(spy.called_tools, schema_name)
    logger.info(f"changes spy: {changes}")

    #######################################################
    logger.info("Test mAIstro successfully.")


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
    test_call_mAIstro_flag = True
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

        if test_call_mAIstro_flag:
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
                output_name="call_mAIstro_graph.png",
            )
            test_call_mAIstro_model(across_thread_memory)
    logger.info("All done.")


"""
## Viewing traces in LangSmith
We can see that the memories are retrieved from the store and supplied as part of the system prompt, as expected:
https://smith.langchain.com/

## Studio
We can also interact with our chatbot in Studio.

"""
