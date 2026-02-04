# src/langgraph_projects/level-3/breakpoints2.py
"""
For `human-in-the-loop`, we often want to see our graph outputs as its running.
We laid the foundations for this with streaming.

Now, let's talk about the motivations for `human-in-the-loop`:
(1) `Approval` - We can interrupt our agent, surface state to a user, and allow the user to accept an action
(2) `Debugging` - We can rewind the graph to reproduce or avoid issues
(3) `Editing` - You can modify the state

LangGraph offers several ways to get or update agent state to support various `human-in-the-loop` workflows.
First, I'll introduce [breakpoints](https://docs.langchain.com/oss/python/langgraph/interrupts#debugging-with-interrupts), which provide a simple way to stop the graph at specific steps.

I'll show how this enables user `approval`.
"""

# %pip install --quiet -U langgraph langchain_openai langgraph_sdk langgraph-prebuilt

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
from PIL import Image
from pyexpat.errors import messages
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

"""
## Breakpoints for human approval
Let's re-consider the simple agent that we worked with in Module 1.
Let's assume that are concerned about tool use: we want to approve the agent to use any of its tools.
All we need to do is simply compile the graph with `interrupt_before=["tools"]` where `tools` is our tools node.
This means that the execution will be interrupted before the node `tools`, which executes the tool call.
"""


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


def build_app():
    init_runtime()
    init_langsmith()

    tools = [add, multiply, divide]

    # For this ipynb we set parallel tool calling to false as math generally is done sequentially, and this time we have 3 tools that can do math
    # the OpenAI model specifically defaults to parallel tool calling for efficiency, see https://python.langchain.com/docs/how_to/tool_calling_parallel/
    # play around with it and see how the model behaves with math equations!
    llm_with_tools = get_model().bind_tools(tools, parallel_tool_calls=False)

    """Let's create our LLM and prompt it with the overall desired agent behavior."""

    # System message
    sys_msg = SystemMessage(
        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
    )

    # Node
    def assistant_node(state: MessagesState):
        logger.debug("Invoking LLM with tool calling capabilities...")
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    """
    As before, we use `MessagesState` and define a `Tools` node with our list of tools.
    The `Assistant` node is just our model with bound tools.
    We create a graph with `Assistant` and `Tools` nodes.
    We add `tools_condition` edge, which routes to `End` or to `Tools` based on  whether the `Assistant` calls a tool.
    
    Now, we add one new step:
    We connect the `Tools` node *back* to the `Assistant`, forming a loop.
    * After the `assistant` node executes, `tools_condition` checks if the model's output is a tool call.
    * If it is a tool call, the flow is directed to the `tools` node.
    * The `tools` node connects back to `assistant`.
    * This loop continues as long as the model decides to call tools.
    * If the model response is not a tool call, the flow is directed to END, terminating the process.
    """

    # Graph
    builder = StateGraph(MessagesState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    # (NEW) Loop back from tools to assistant
    builder.add_edge("tools", "assistant")
    memory = MemorySaver()
    # graph = builder.compile()
    # graph = builder.compile(interrupt_before=["tools"], checkpointer=memory)
    graph = builder.compile(interrupt_before=["assistant"], checkpointer=memory)
    return graph


# from langchain_core.messages import HumanMessage

graph = build_app()


def prompt_yes_no(prompt: str) -> bool:
    while True:
        sys.stdout.write(prompt)
        sys.stdout.flush()
        resp = sys.stdin.readline()  #  <-- terminal STDIN
        if not resp:  #  EOF (e.g., piped input ended)
            return False
        resp = resp.strip().lower()
        if resp in {"y", "yes"}:
            return True
        if resp in {"n", "no"}:
            return False
        print("Please type yes or no.")


import asyncio


async def breakpoints_with_langgraph_api():
    """
    Breakpoints with LangGraph API
    ---------------------------------
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

    # This is the URL of the local development server (You need to change)
    client = get_client(url="http://127.0.0.1:2024")

    # shown above, we can add `interrupt_before=["node"]` when compiling the graph that is running in Studio.
    # However, with the API, you can also pass `interrupt_before` to the stream method directly.
    initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}
    thread = await client.threads.create()
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id="agent",
        input=initial_input,
        stream_mode="values",
        interrupt_before=["tools"],
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        messages = chunk.data.get("messages", [])
        if messages:
            print(messages[-1])
        print("-" * 50)

    # We can get the current state
    current_state = await client.threads.get_state(thread["thread_id"])
    logger.info("Graph interrupted at breakpoint.")
    logger.info(f"Current state at breakpoint: {current_state}")

    """We can look at the last message in state."""

    last_message = current_state["values"]["messages"][-1]
    logger.info("Last message at breakpoint:")
    logger.info(f"last Message: {last_message}")

    # We can edit it!
    last_message["content"] = "No, actually multiply 3 and 3!"
    logger.info("Updated last message at breakpoint:")
    logger.info(f"Updated last Message: {last_message}")

    """
    Remember, as we said before, updates to the `messages` key will use the same `add_messages` reducer.
    If we want to over-write the existing message, then we can supply the message `id`.
    Here, we did that. We only modified the message `content`, as shown above.
    """
    await client.threads.update_state(thread["thread_id"], {"messages": last_message})
    #################################################################

    # Now, we can proceed from the breakpoint just like we did before by passing the `thread_id` and `None` as the input!
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id="agent",
        input=None,
        stream_mode="values",
        interrupt_before=["assistant"],
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        messages = chunk.data.get("messages", [])
        if messages:
            print(messages[-1])
        print("-" * 50)

    # We get the result of the tool call as `9`, as expected.
    async for chunk in client.runs.stream(
        thread["thread_id"],
        assistant_id="agent",
        input=None,
        stream_mode="values",
        interrupt_before=["assistant"],
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        messages = chunk.data.get("messages", [])
        if messages:
            print(messages[-1])
        print("-" * 50)

    logger.info("Part 1 Graph execution Done.")

    """Editing State at Breakpoints with Human Feedback
    ---------------------------------
    In the previous example, we interrupted the graph before the `tools` node.
    We got the current state, modified it programmatically, and updated the graph state.

    So, it's clear that we can edit our agent state after a breakpoint.
    Now, what if we want to allow for human feedback to perform this state update?
    We'll add a node that serves as a placeholder for human feedback within our agent.
    This `human_feedback` node allow the user to add feedback directly to state.
    We specify the breakpoint using `interrupt_before` our `human_feedback` node.
    We set up a checkpointer to save the state of the graph up until this node.
    """
    tools = [add, multiply, divide]
    llm_with_tools = get_model().bind_tools(tools, parallel_tool_calls=False)

    # System message
    sys_msg = SystemMessage(
        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
    )

    # no-op node that should be interrupted on
    def human_feedback(state: MessagesState):
        pass

    # Assistant node
    def assistant(state: MessagesState):
        return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

    # Graph
    builder = StateGraph(MessagesState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("human_feedback", human_feedback)

    # Define edges: these determine the control flow
    builder.add_edge(START, "human_feedback")
    builder.add_edge("human_feedback", "assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("tools", "human_feedback")

    memory = MemorySaver()
    graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)
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
        output_name="api_agent_breakpoints.png",
    )

    """
    We will get feedback from the user.
    We use `.update_state` to update the state of the graph with the human response we get, as before.
    We use the `as_node="human_feedback"` parameter to apply this state update as the specified node, `human_feedback`.
    """

    # Input
    initial_input = {"messages": "Multiply 2 and 3"}

    # Thread
    thread = {"configurable": {"thread_id": "5"}}

    # Run the graph until the first interruption
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()
    print("-" * 50)
    # Get user input
    user_input = input("Tell me how you want to update the state: ")

    # We now update the state as if we are the human_feedback node
    graph.update_state(thread, {"messages": user_input}, as_node="human_feedback")

    # Continue the graph execution
    logger.info("Resuming from breakpoint after human feedback...")
    for event in graph.stream(None, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    # Continue the graph execution
    logger.info("Resuming from second breakpoint after human feedback...")
    for event in graph.stream(None, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

    logger.info("Part 2 Graph execution Done.")


def breakpoints_example():
    # Demo / manual run (kept under __main__ so imports remain side-effect-light).
    # Input
    initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}

    # Thread
    thread = {"configurable": {"thread_id": "1"}}

    # Run the graph until the first interruption
    logger.info("Run the graph until the first interruption...")
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()
    print("\n" + "---" * 25)
    # We can get the state and look at the next node to call.
    # This is a nice way to see that the graph has been interrupted.

    state = graph.get_state(thread)
    logger.info("Graph interrupted at breakpoint.")
    logger.info(f"Current state: {state}")

    # state.next
    logger.info(f"Next node to execute: {state.next}")

    """
    Now, we can directly apply a state update.
    Remember, updates to the `messages` key will use the `add_messages` reducer:
    * If we want to over-write the existing message, we can supply the message `id`.
    * If we simply want to append to our list of messages, then we can pass a message without an `id` specified, as shown below.
    """
    graph.update_state(
        thread,
        {"messages": [HumanMessage(content="No, actually multiply 3 and 3!")]},
    )
    print("\n" + "---" * 25)
    """
    Let's have a look.
    We called `update_state` with a new message.
    The `add_messages` reducer appends it to our state key, `messages`.
    """
    logger.info("Updated state after user correction:")
    new_state = graph.get_state(thread).values
    for m in new_state["messages"]:
        m.pretty_print()
    print("\n" + "---" * 25)
    # Now, let's proceed with our agent, simply by passing `None` and allowing it proceed from the current state.
    # We emit the current and then proceed to execute the remaining nodes.
    logger.info("Resuming from breakpoint...")
    for event in graph.stream(None, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()
    print("\n" + "---" * 25)
    # Now, we're back at the `assistant`, which has our `breakpoint`.
    # We can again pass `None` to proceed.
    logger.info("Resuming from second breakpoint...")
    for event in graph.stream(None, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()
    print("\n" + "---" * 25)
    logger.info("Part 1 Graph execution Done.")
    print("\n" + "---" * 25)
    ############################################################
    # Now, lets bring these together with a specific user approval step that accepts user input.
    ############################################################
    logger.info("specific user approval step that accepts user input.")
    # Input
    initial_input = {"messages": HumanMessage(content="Multiply 2 and 3")}

    # Thread
    thread = {"configurable": {"thread_id": "2"}}

    # Run the graph until the first interruption
    for event in graph.stream(initial_input, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()
    print("\n" + "---" * 25)
    # Get user feedback
    # user_approval = input("Do you want to call the tool? (yes/no): ")
    # Get user feedback  # Existing Comment
    approved = prompt_yes_no("Do you want to call the tool? (yes/no): ")

    # Check approval
    if approved:
        # If approved, continue the graph execution
        for event in graph.stream(None, thread, stream_mode="values"):
            event["messages"][-1].pretty_print()
        print("\n" + "---" * 25)
    else:
        print("Operation cancelled by user.")

    logger.info("Graph execution Done.")
    print("\n" + "---" * 25)


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
    ## Breakpoints with LangGraph API
    ############################################################
    breakpoints_example()
    # asyncio.run(breakpoints_with_langgraph_api())
    logger.info("All done.")
