# src/langgraph_projects/level-3/time_travel5.py
"""
Now, let's show how LangGraph [supports debugging](https://docs.langchain.com/oss/python/langgraph/use-time-travel) by viewing, re-playing, and even forking from past states.
We call this `time travel`.
"""


# %pip install --quiet -U langgraph langchain_openai langgraph_sdk langgraph-prebuilt
###########################################################
## Supporting Code
###########################################################
import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from PIL import Image
from pyexpat.errors import messages

from langgraph_projects.my_utils.load_env import load_dotenv_only, validate_environment
from langgraph_projects.my_utils.logger_setup import setup_logger

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


# This will be a tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    value = a * b
    logger.info(f"Tool `multiply` called with a={a}, b={b}, result={value}")
    return value


# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    value = a + b
    logger.info(f"Tool `add` called with a={a}, b={b}, result={value}")
    return value


def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    value = a / b
    logger.info(f"Tool `divide` called with a={a}, b={b}, result={value}")
    return value

    return build_app(use_checkpointer=False)


def build_app(*, use_checkpointer: bool = False):
    init_runtime()
    init_langsmith()

    tools = [add, multiply, divide]
    llm = get_model()
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    # from langgraph.checkpoint.memory import MemorySaver
    # from langgraph.graph import END, START, MessagesState, StateGraph
    # from langgraph.prebuilt import ToolNode, tools_condition

    # System message
    sys_msg = SystemMessage(
        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
    )

    # Node
    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    # Graph
    builder = StateGraph(MessagesState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine the control flow
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    if use_checkpointer:
        # Set up memory
        memory = MemorySaver()
        # Compile the graph with memory
        graph = builder.compile(checkpointer=memory)
        return graph
    else:
        graph = builder.compile()
        return graph


def get_graph_remote():
    return build_app(use_checkpointer=False)


graph_remote = get_graph_remote()
graph = build_app(use_checkpointer=True)


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


def run_time_travel_example():
    init_runtime()
    init_langsmith()

    # Let's run it, as before.

    # Input
    initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}

    # Thread
    thread = {"configurable": {"thread_id": "1"}}

    # Run the graph with streaming updates to state.
    logger.info("Multiply 2 and 3 - Running agent with thread 1...")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    print("\n" + "---" * 25)
    logger.info("Graph done.")
    # Browsing History
    # We can use `get_state` to look at the **current** state of our graph, given the `thread_id`!
    current_state = graph.get_state({"configurable": {"thread_id": "1"}})
    log_state_time_travel(
        current_state, raw_flag=True, pretty_raw=True, label="Local Current state"
    )
    # We can also browse the state history of our agent.
    # `get_state_history` lets us get the state at all prior steps.

    all_states = [s for s in graph.get_state_history(thread)]
    logger.info(f"Total states in history: {len(all_states)}")
    # optional: print a quick view (StateSnapshot objects)
    for i, s in enumerate(all_states[:5], start=1):
        cfg = getattr(s, "config", {}) or {}
        configurable = cfg.get("configurable", {}) if isinstance(cfg, dict) else {}

        checkpoint_id = configurable.get("checkpoint_id")
        next_nodes = getattr(s, "next", None)
        metadata = getattr(s, "metadata", {}) or {}
        step = metadata.get("step") if isinstance(metadata, dict) else None

        logger.info(
            "%02d | checkpoint_id=%s | next=%s | step=%s",
            i,
            checkpoint_id,
            next_nodes,
            step,
        )

    print("\n" + "---" * 25)

    # The first element is the current state, just as we got from `get_state`.
    # all_states[-1] = oldest
    # all_states[-2] = second oldest
    back_state = all_states[-1]
    log_state_time_travel(back_state, raw_flag=True, label="First State ")
    # Let's look back at the step that recieved human input!

    to_replay = all_states[-2]
    log_state_time_travel(to_replay, raw_flag=True, label="Replaying from state")
    # Look at the state.
    logger.info(f"Replaying state values: {to_replay.values}")

    # We can see the next node to call.
    logger.info(f"Replaying next node: {to_replay.next}")

    # We also get the config, which tells us the `checkpoint_id` as well as the `thread_id`.
    logger.info(f"Replaying from config: {to_replay.config}")

    # To replay from here, we simply pass the config back to the agent!
    # The graph knows that this checkpoint has aleady been executed.
    # It just re-plays from this checkpoint!
    logger.info("Re-playing from checkpoint...")
    for event in graph.stream(None, to_replay.config, stream_mode="values"):
        event["messages"][-1].pretty_print()

    print("\n" + "---" * 25)
    ###################################################
    ## Forking
    ###################################################
    #   Now, we can see our current state after the agent re-ran.
    ## Forking
    # What if we want to run from that same step, but with a different input.
    # This is forking.

    to_fork = all_states[-2]
    log_state_time_travel(to_fork, raw_flag=True, label="Forking from state")
    # to_fork.values["messages"]
    logger.info(f"Forking from messages: {to_fork.values['messages']}")

    # Again, we have the config.
    logger.info(f"Forking from config: {to_fork.config}")

    """
    Let's modify the state at this checkpoint.
    We can just run `update_state` with the `checkpoint_id` supplied.
    Remember how our reducer on `messages` works:
    * It will append, unless we supply a message ID.
    * We supply the message ID to overwrite the message, rather than appending to state!
    So, to overwrite the the message, we just supply the message ID, which we have `to_fork.values["messages"].id`.
    """
    logger.info("Forking from checkpoint with new input...")
    fork_config = graph.update_state(
        to_fork.config,
        {
            "messages": [
                HumanMessage(
                    content="Multiply 5 and 3", id=to_fork.values["messages"][0].id
                )
            ]
        },
    )

    logger.info(f"Forked config: {fork_config}")

    """
    This creates a new, forked checkpoint.
    But, the metadata - e.g., where to go next - is perserved!
    We can see the current state of our agent has been updated with our fork.
    """
    #     logger.info(f"Total states in history: {len(all_states)}")
    all_states = [state for state in graph.get_state_history(thread)]
    logger.info(f"Total states in history: {len(all_states)}")
    state = all_states[0].values["messages"]
    logger.info(f"Forked state messages: {state}")

    config = graph.get_state({"configurable": {"thread_id": "1"}})
    # logger.info(f"Forked state configurable: {config}")
    log_state_time_travel(config, raw_flag=True, label="Forked Current state")

    """Now, when we stream, the graph knows this checkpoint has never been executed.

    So, the graph runs, rather than simply re-playing.
    """
    logger.info("Running forked checkpoint...")
    for event in graph.stream(None, fork_config, stream_mode="values"):
        event["messages"][-1].pretty_print()

    print("\n" + "---" * 25)
    # Now, we can see the current state is the end of our agent run.

    current_state = graph.get_state({"configurable": {"thread_id": "1"}})
    logger.info("Graph done.")
    log_state_time_travel(current_state, raw_flag=True, label="Forked Current state")
    # logger.info(f"Current state: {current_state}")
    # logger.info(
    #     "Thread state attrs: %s",
    #     ["values", "next", "tasks", "interrupts", "metadata", "config"],
    # )
    # logger.info("Next: %s", current_state.next)
    # logger.info("Tasks: %s", current_state.tasks)
    # logger.info("Interrupts: %s", current_state.interrupts)

    logger.info("Time travel example complete.")


async def run_time_travel_with_langgraph_sdk_example():
    """
    langgraph dev see README.md for instructions on running the local dev server.

    You should see the following output:
    - API: http://127.0.0.1:2024
    - Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
    - API Docs: http://127.0.0.1:2024/docs

    Open your browser and navigate to the **Studio UI** URL shown above.
    The LangGraph API [supports breakpoints](https://docs.langchain.com/langsmith/add-human-in-the-loop).
    """
    init_runtime()
    init_langsmith()

    from langgraph_sdk import get_client

    assistant_id = "time_travel_graph"
    client = get_client(url="http://127.0.0.1:2024")

    """#### Re-playing

    Let's run our agent streaming `updates` to the state of the graph after each node is called.
    """
    ######################################################
    # First run of agent.
    #######################################################
    initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}
    thread = await client.threads.create()

    logger.info("A full run of the agent with streaming updates...")
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id=assistant_id,
        input=initial_input,
        stream_mode="updates",
    ):
        if chunk.data:
            assisant_node = chunk.data.get("assistant", {}).get("messages", [])
            tool_node = chunk.data.get("tools", {}).get("messages", [])
            if assisant_node:
                print("-" * 20 + "Assistant Node" + "-" * 20)
                print(assisant_node[-1])
            elif tool_node:
                print("-" * 20 + "Tools Node" + "-" * 20)
                print(tool_node[-1])

    print("\n" + "---" * 25)
    states = await client.threads.get_history(thread["thread_id"])
    logger.info("Total states in API history: %d", len(states))

    # optional: print a quick view
    for i, s in enumerate(states[:5], start=1):
        logger.info(
            "%02d | checkpoint_id=%s | next=%s | step=%s",
            i,
            s.get("checkpoint_id"),
            s.get("next"),
            (s.get("metadata") or {}).get("step"),
        )
    print("\n" + "---" * 25)

    # Now, let's look at **replaying** from a specified checkpoint.
    # We simply need to pass the `checkpoint_id`.
    logger.info("Replaying from a prior checkpoint with streaming values...")
    states = await client.threads.get_history(thread["thread_id"])
    to_replay = states[-2]
    log_state_time_travel(
        to_replay, raw_flag=True, pretty_raw=True, label="Replaying from state"
    )
    logger.info(f"Replaying from state values: {to_replay}")

    # Let's stream with `stream_mode="values"` to see the full state at every node as we replay.
    logger.info(
        "Streaming with stream_mode='values' to see the full state at every node as we replay."
    )
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id=assistant_id,
        input=None,
        stream_mode="values",
        checkpoint_id=to_replay["checkpoint_id"],
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")
    print("\n" + "---" * 25)

    # We can all view this as streaming only `updates` to state made by the nodes that we reply.
    logger.info(
        "Streaming with stream_mode='updates' to see only the updates made by the nodes as we replay."
    )
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id=assistant_id,
        input=None,
        stream_mode="updates",
        checkpoint_id=to_replay["checkpoint_id"],
    ):
        if chunk.data:
            assisant_node = chunk.data.get("assistant", {}).get("messages", [])
            tool_node = chunk.data.get("tools", {}).get("messages", [])
            if assisant_node:
                print("-" * 20 + "Assistant Node" + "-" * 20)
                print(assisant_node[-1])
            elif tool_node:
                print("-" * 20 + "Tools Node" + "-" * 20)
                print(tool_node[-1])
    print("\n" + "---" * 25)
    ##########################################################
    # Forking
    ###########################################################
    """
    #### Forking
    Now, let's look at forking.
    Let's get the same step as we worked with above, the human input.
    Let's create a new thread with our agent.
    """

    initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}
    thread = await client.threads.create()
    logger.info("Multiply 2 and 3 - Running agent with new thread...")
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id=assistant_id,
        input=initial_input,
        stream_mode="updates",
    ):
        if chunk.data:
            assistant_node = chunk.data.get("assistant", {}).get("messages", [])
            tool_node = chunk.data.get("tools", {}).get("messages", [])
            if assisant_node:
                print("-" * 20 + "Assistant Node" + "-" * 20)
                print(assisant_node[-1])
            elif tool_node:
                print("-" * 20 + "Tools Node" + "-" * 20)
                print(tool_node[-1])
    print("\n" + "---" * 25)

    states = await client.threads.get_history(thread["thread_id"])
    to_fork = states[-2]

    logger.info("Forking from state:")
    logger.info(to_fork)
    # to_fork["values"]
    logger.info(f"Forking from messages: {to_fork['values']}")
    # to_fork["values"]["messages"][0]["id"]
    logger.info(f"Forking from message ID: {to_fork['values']['messages'][0]['id']}")
    # to_fork["next"]
    logger.info(f"Forking from next node: {to_fork['next']}")
    # to_fork["checkpoint_id"]
    logger.info(f"Forking from checkpoint ID: {to_fork['checkpoint_id']}")
    """
    Let's edit the state.
    Remember how our reducer on `messages` works:
    * It will append, unless we supply a message ID.
    * We supply the message ID to overwrite the message, rather than appending to state!
    """

    forked_input = {
        "messages": HumanMessage(
            content="Multiply 3 and 3", id=to_fork["values"]["messages"][0]["id"]
        )
    }

    forked_config = await client.threads.update_state(
        thread["thread_id"], forked_input, checkpoint_id=to_fork["checkpoint_id"]
    )
    # forked_config
    logger.info(f"Forked config: {forked_config}")

    states = await client.threads.get_history(thread["thread_id"])
    # states[0]
    logger.info(f"Forked state: {states[0]}")
    logger.info(f"Forked state messages: {states[0]['values']['messages']}")

    """To rerun, we pass in the `checkpoint_id`."""
    logger.info("Running forked checkpoint with streaming updates...")
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id=assistant_id,
        input=None,
        stream_mode="updates",
        checkpoint_id=forked_config["checkpoint_id"],
    ):
        if chunk.data:
            assisant_node = chunk.data.get("assistant", {}).get("messages", [])
            tool_node = chunk.data.get("tools", {}).get("messages", [])
            if assisant_node:
                print("-" * 20 + "Assistant Node" + "-" * 20)
                print(assisant_node[-1])
            elif tool_node:
                print("-" * 20 + "Tools Node" + "-" * 20)
                print(tool_node[-1])
    print("\n" + "---" * 25)
    logger.info("SDK Forking example complete.")
    """
    ### LangGraph Studio
    Let's look at forking in the Studio UI with our `agent`, which uses `level-1/studio/agent.py` set in `level-1/studio/langgraph.json`.
    """


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
        output_name="time_travel.png",
    )
    ############################################################
    ## Run the dynamic breakpoints example
    ############################################################
    langgraph_dev = False
    if langgraph_dev:
        import asyncio

        asyncio.run(run_time_travel_with_langgraph_sdk_example())

    else:
        run_time_travel_example()

    logger.info("All done.")
