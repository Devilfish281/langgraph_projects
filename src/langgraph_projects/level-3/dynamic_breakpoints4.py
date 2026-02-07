# src/langgraph_projects/level-3/dynamic_breakpoints4.py
"""
Breakpoints are set by the developer on a specific node during graph compilation.
But, sometimes it is helpful to allow the graph **dynamically interrupt** itself!
This is an internal breakpoint, and can be achieved using `interrupt()`.
This has a few specific benefits:
(1) You can do it conditionally (from inside a node based on developer-defined logic).
(2) You can communicate to the user why it's interrupted (by passing whatever you want to `interrupt()`).
Let's create a graph where a `interrupt()` is thrown based on the length of the input.
"""


# %pip install --quiet -U langgraph langchain_openai langgraph_sdk
###########################################################
## Supporting Code
###########################################################
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
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.errors import NodeInterrupt
# from langgraph.graph import END, START, StateGraph
# from typing_extensions import TypedDict


class State(TypedDict):
    input: str


def step_1(state: State) -> State:
    logger.info("---Step 1---")
    return state


def step_2(state: State) -> State:
    # Dynamically pause if the input is longer than 5 characters.
    if len(state["input"]) > 5:
        interrupt(
            {
                "reason": "Input too long (> 5 chars).",
                "input": state["input"],
                "max_len": 5,
                "hint": "Update state (or resume with a shorter value) and re-run.",
            }
        )

    logger.info("---Step 2---")
    return state


def step_3(state: State) -> State:
    logger.info("---Step 3---")
    return state


def get_graph_remote():
    return build_app(use_checkpointer=False)


def build_app(*, use_checkpointer: bool = False):
    init_runtime()
    init_langsmith()

    builder = StateGraph(State)
    builder.add_node("step_1", step_1)
    builder.add_node("step_2", step_2)
    builder.add_node("step_3", step_3)

    builder.add_edge(START, "step_1")
    builder.add_edge("step_1", "step_2")
    builder.add_edge("step_2", "step_3")
    builder.add_edge("step_3", END)

    if use_checkpointer:
        # Set up memory
        memory = MemorySaver()
        # Compile the graph with memory
        graph = builder.compile(checkpointer=memory)
        return graph
    else:
        graph = builder.compile()
        return graph


graph_remote = get_graph_remote()
graph = build_app(use_checkpointer=True)


async def dynamic_breakpoints_with_langgraph_api():
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

    # This is the URL of the local development server
    URL = "http://127.0.0.1:2024"
    client = get_client(url=URL)

    # Search all hosted graphs
    assistants = await client.assistants.search()
    logger.info(f"Found {len(assistants)} assistants on the server.")
    for i, a in enumerate(assistants, start=1):
        print(
            f"{i:02d} | graph_id={a.get('graph_id')} | assistant_id={a.get('assistant_id')} | name={a.get('name')}"
        )

    thread = await client.threads.create()
    input_dict = {"input": "hello world"}

    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id="dynamic_breakpoints_graph",
        input=input_dict,
        stream_mode="values",
    ):

        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")

    print("\n" + "---" * 25)
    current_state = await client.threads.get_state(thread["thread_id"])
    logger.info("Graph interrupted at breakpoint.")
    logger.info(f"Current state: {current_state}")
    logger.info("Thread state keys: %s", list(current_state.keys()))  # Added Code
    logger.info("Next: %s", current_state.get("next"))  # Added Code
    logger.info("Tasks: %s", current_state.get("tasks"))  # Added Code
    logger.info("Interrupts: %s", current_state.get("interrupts"))  # Added Code

    logger.info("Updating state with new input...")
    await client.threads.update_state(thread["thread_id"], {"input": "hi!"})
    logger.info("Resuming graph execution from breakpoint...")
    print("\n" + "---" * 25)
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id="dynamic_breakpoints_graph",
        input=None,
        stream_mode="values",
    ):

        print(f"Receiving new event of type: {chunk.event}...")
        print(chunk.data)
        print("\n\n")

    print("\n" + "---" * 25)

    current_state = await client.threads.get_state(thread["thread_id"])
    logger.info("Graph interrupted at breakpoint.")
    logger.info(f"Current state: {current_state}")
    logger.info("Thread state keys: %s", list(current_state.keys()))  # Added Code
    logger.info("Next: %s", current_state.get("next"))  # Added Code
    logger.info("Tasks: %s", current_state.get("tasks"))  # Added Code
    logger.info("Interrupts: %s", current_state.get("interrupts"))  # Added Code

    step_num = current_state.get("metadata", {}).get("step")  # Added Code
    logger.info("step': %s", step_num)  # Added Code

    logger.info("Dynamic breakpoints with LangGraph API example complete.")


def dynamic_breakpoints_example():
    logger.info("Running dynamic breakpoints example...")

    # Let's run the graph with an input that's longer than 5 characters.
    initial_input = {"input": "hello world"}
    thread_config = {"configurable": {"thread_id": "1"}}

    # Run the graph until the first interruption
    logger.info("Streaming graph execution until interruption...")
    for event in graph.stream(initial_input, thread_config, stream_mode="values"):
        print(event)
    print("\n" + "---" * 25)
    # If we inspect the graph state at this point, we the node set to execute next (`step_2`).
    state = graph.get_state(thread_config)
    logger.info("Graph interrupted at breakpoint.")
    logger.info(f"Current state: {state}")
    logger.info(f"Next node to execute: {state.next}")

    # We can see that the `Interrupt` is logged to state.
    logger.info(f"Tasks at breakpoint: {state.tasks}")

    """
    We can try to resume the graph from the breakpoint.
    But, this just re-runs the same node!
    Unless state is changed we will be stuck here.
    """
    print("\n" + "---" * 25)
    logger.info("Resuming graph execution from breakpoint...")
    for event in graph.stream(None, thread_config, stream_mode="values"):
        print(event)
    print("\n" + "---" * 25)

    state = graph.get_state(thread_config)
    logger.info("Graph interrupted at breakpoint.")
    logger.info(f"Current state: {state}")
    logger.info(f"Next node to execute: {state.next}")
    # Now, we can update state.

    logger.info("Updating state with new input...")
    graph.update_state(
        thread_config,
        {"input": "hi"},
    )
    logger.info("Resuming graph execution from breakpoint...")
    for event in graph.stream(None, thread_config, stream_mode="values"):
        print(event)
    print("\n" + "---" * 25)

    logger.info("Dynamic breakpoints example complete.")


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
        output_name="agent_breakpoints.png",
    )
    ############################################################
    ## Run the dynamic breakpoints example
    ############################################################
    langgraph_dev = True
    if langgraph_dev:
        import asyncio

        asyncio.run(dynamic_breakpoints_with_langgraph_api())
    else:
        dynamic_breakpoints_example()

    logger.info("All done.")
