# src/langgraph_projects/level-3/streaming_interruption1.py
"""
This module I will dive into `human-in-the-loop`, which builds on memory and allows users to interact directly with graphs in various ways.
To set the stage for `human-in-the-loop`, we'll first dive into streaming, which provides several ways to visualize graph output (e.g., node state or chat model tokens) over the course of execution.
"""

"""
### Streaming Full Graph State in LangGraph

LangGraph supports streaming during graph execution so you can monitor how 
the state evolves after each node runs. There are two primary streaming approaches:

.stream - Synchronous method to yield states step by step.
.astream - Asynchronous version for async environments.

## Streaming Modes LangGraph offers two distinct streaming modes: (values and updates).
###########################
mode="values"` - Use `values` when you need the full picture at each step.
############################
Streams the entire state of the graph after each node execution.
Each step shows the cumulative state so far.

After node 1:
{ "messages": ["a"] }

After node 2:
{ "messages": ["a", "b"] }

After node 3:
{ "messages": ["a", "b", "c"] }
###########################
mode="updates"` - Use `updates` when you're only interested in what changed.
###########################
Streams only the changes (deltas) made to the state after each node call.
Each step shows just what was newly added or modified.

 After node 1:
{ "messages": ["a"] }

After node 2:
{ "messages": ["b"] }

After node 3:
{ "messages": ["c"] }
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


"""#
# Streaming
LangGraph is built with [first class support for streaming](https://docs.langchain.com/oss/python/langgraph/streaming).
Let's set up our Chatbot from Level 2, and show various way to stream outputs from the graph during execution.
"""


"""Note that we use `RunnableConfig` with `call_model` to enable token-wise streaming. This is [only needed with python < 3.11](https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/). We include in case you are running this notebook in CoLab, which will use python 3.x."""
from langchain_core.runnables import RunnableConfig

# from typing import Literal

# from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
# from langchain_core.runnables import RunnableConfig
# from langchain_openai import ChatOpenAI
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import END, START, MessagesState, StateGraph


def messages_with_summary():
    # State
    class State(MessagesState):
        summary: str

    # Define the logic to call the model
    def call_model_node(state: State, config: RunnableConfig):

        # Get summary if it exists
        summary = state.get("summary", "")

        # If there is summary, then we add it
        if summary:

            # Add summary to system message
            system_message = f"Summary of conversation earlier: {summary}"

            # Append summary to any newer messages
            messages = [SystemMessage(content=system_message)] + state["messages"]

        else:
            messages = state["messages"]

        response = get_model().invoke(messages, config)
        return {"messages": response}

    def summarize_conversation_node(state: State):

        # First, we get any existing summary
        summary = state.get("summary", "")

        # Create our summarization prompt
        if summary:

            # A summary already exists
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )

        else:
            summary_message = "Create a summary of the conversation above:"

        # Add prompt to our history
        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = get_model().invoke(messages)

        # Delete all but the 2 most recent messages
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"summary": response.content, "messages": delete_messages}

    # Determine whether to end or summarize the conversation
    def should_continue(state: State) -> Literal["summarize_conversation", END]:
        """Return the next node to execute."""

        messages = state["messages"]

        # If there are more than six messages, then we summarize the conversation
        if len(messages) > 6:
            return "summarize_conversation"

        # Otherwise we can just end
        return END

    # Define a new graph
    workflow = StateGraph(State)
    workflow.add_node("conversation", call_model_node)
    workflow.add_node("summarize_conversation", summarize_conversation_node)

    # Set the entrypoint as conversation
    workflow.add_edge(START, "conversation")
    workflow.add_conditional_edges("conversation", should_continue)
    workflow.add_edge("summarize_conversation", END)

    # Compile
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    return graph


##############################################
# Build App
##############################################
def build_app():

    init_runtime()
    init_langsmith()

    graph = messages_with_summary()

    return graph


graph = build_app()

import asyncio

"""
### Streaming Full Graph State in LangGraph

LangGraph supports streaming during graph execution so you can monitor how 
the state evolves after each node runs. There are two primary streaming approaches:

.stream - Synchronous method to yield states step by step.
.astream - Asynchronous version for async environments.

## Streaming Modes LangGraph offers two distinct streaming modes: (values and updates).
###########################
mode="values"` - Use `values` when you need the full picture at each step.
############################
Streams the entire state of the graph after each node execution.
Each step shows the cumulative state so far.

After node 1:
{ "messages": ["a"] }

After node 2:
{ "messages": ["a", "b"] }

After node 3:
{ "messages": ["a", "b", "c"] }
###########################
mode="updates"` - Use `updates` when you're only interested in what changed.
###########################
Streams only the changes (deltas) made to the state after each node call.
Each step shows just what was newly added or modified.

 After node 1:
{ "messages": ["a"] }

After node 2:
{ "messages": ["b"] }

After node 3:
{ "messages": ["c"] }
"""


# Only run demos when you execute the file directly (NOT when Studio imports it).
async def main():
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
        output_name="streaming_interruption.png",
    )

    """
    ### Streaming full state
    Now, let's talk about ways to [stream our graph state](https://docs.langchain.com/oss/python/langgraph/streaming#supported-stream-modes).
    `.stream` and `.astream` are sync and async methods for streaming back results.
    LangGraph supports a few [different streaming modes](https://docs.langchain.com/oss/python/langgraph/streaming#stream-graph-state) for graph state.
    * `values`: This streams the full state of the graph after each node is called.
    * `updates`: This streams updates to the state of the graph after each node is called.
    ![values_vs_updates.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbaf892d24625a201744e5_streaming1.png)
    Let's look at `stream_mode="updates"`.
    Because we stream with `updates`, we only see updates to the state after node in the graph is run.
    Each `chunk` is a dict with `node_name` as the key and the updated state as the value.

    stream and astream
    """

    """
    ###########################
    mode="updates"` - Use `updates` when you're only interested in what changed.
    ###########################
    Streams only the changes (deltas) made to the state after each node call.
    Each step shows just what was newly added or modified.

    After node 1:
    { "messages": ["a"] }

    After node 2:
    { "messages": ["b"] }

    After node 3:
    { "messages": ["c"] }
    """
    # Start conversation
    # stream returns an iterator of state updates
    print("\n" + "---" * 25)
    logger.info("stream mode=updates thread 1.....")
    # Create a thread
    config = {"configurable": {"thread_id": "1"}}
    for chunk in graph.stream(
        {"messages": [HumanMessage(content="hi! I'm Matthew")]},
        config,
        stream_mode="updates",
    ):
        print(chunk)

    # Let's now just print the state update.
    # Start conversation
    print("\n" + "---" * 25)
    logger.info("stream mode=updates conversation...")
    for chunk in graph.stream(
        {"messages": [HumanMessage(content="hi! I'm Matthew")]},
        config,
        stream_mode="updates",
    ):
        chunk["conversation"]["messages"].pretty_print()

    """
    ###########################
    mode="values"` - Use `values` when you need the full picture at each step.
    ############################
    Streams the entire state of the graph after each node execution.
    Each step shows the cumulative state so far.

    After node 1:
    { "messages": ["a"] }

    After node 2:
    { "messages": ["a", "b"] }

    After node 3:
    { "messages": ["a", "b", "c"] }
    """
    # Now, we can see `stream_mode="values"`.
    # This is the `full state` of the graph after the `conversation` node is called.
    print("\n" + "---" * 25)
    logger.info("stream mode=values thread 2...")
    # Start conversation, again
    config = {"configurable": {"thread_id": "2"}}

    # Start conversation
    input_message = HumanMessage(content="hi! I'm Matthew")
    for event in graph.stream(
        {"messages": [input_message]}, config, stream_mode="values"
    ):
        for m in event["messages"]:
            m.pretty_print()
        print("\n" + "---" * 25)

    ##########################################################
    ### Streaming tokens
    ##########################################################
    """
    We often want to stream more than graph state.
    In particular, with chat model calls it is common to stream the tokens as they are generated.
    We can do this [using the `.astream_events` method](https://docs.langchain.com/oss/python/langchain/models#advanced-streaming-topics:streaming-events), which streams back events as they happen inside nodes!
    Each event is a dict with a few keys:
    * `event`: This is the type of event that is being emitted.
    * `name`: This is the name of event.
    * `data`: This is the data associated with the event.
    * `metadata`: Contains`langgraph_node`, the node emitting the event.
    """
    print("\n" + "---" * 25)
    logger.info("async astream_events  thread 3...")
    config = {"configurable": {"thread_id": "3"}}
    input_message = HumanMessage(content="Tell me about the 49ers NFL team")
    async for event in graph.astream_events(
        {"messages": [input_message]}, config, version="v2"
    ):
        print(
            f"Node: {event['metadata'].get('langgraph_node','')}. Type: {event['event']}. Name: {event['name']}"
        )

    """
    The central point is that tokens from chat models within your graph have the `on_chat_model_stream` type.
    We can use `event['metadata']['langgraph_node']` to select the node to stream from.
    And we can use `event['data']` to get the actual data for each event, which in this case is an `AIMessageChunk`.
    """
    print("\n" + "---" * 25)
    logger.info("async  astream_events thread 4...")
    node_to_stream = "conversation"
    config = {"configurable": {"thread_id": "4"}}
    input_message = HumanMessage(content="Tell me about the 49ers NFL team")
    async for event in graph.astream_events(
        {"messages": [input_message]}, config, version="v2"
    ):
        # Get chat model tokens from a particular node
        if (
            event["event"] == "on_chat_model_stream"
            and event["metadata"].get("langgraph_node", "") == node_to_stream
        ):
            print(event["data"])

    """As you see above, just use the `chunk` key to get the `AIMessageChunk`."""
    print("\n" + "---" * 25)
    logger.info("async  astream_events thread 5...")
    config = {"configurable": {"thread_id": "5"}}
    input_message = HumanMessage(content="Tell me about the 49ers NFL team")
    async for event in graph.astream_events(
        {"messages": [input_message]}, config, version="v2"
    ):
        # Get chat model tokens from a particular node
        if (
            event["event"] == "on_chat_model_stream"
            and event["metadata"].get("langgraph_node", "") == node_to_stream
        ):
            data = event["data"]
            print(data["chunk"].content, end="|")
    print("\n" + "---" * 25)
    print("End of async main().")


async def main_langgraph_dev():
    """
    ### Streaming with LangGraph API
    langgraph dev

    You should see the following output:
    - API: http://127.0.0.1:2024
    - Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
    - API Docs: http://127.0.0.1:2024/docs
    Open your browser and navigate to the Studio UI URL shown above.
    The LangGraph API  [supports editing graph state](https://docs.langchain.com/langsmith/add-human-in-the-loop).
    """

    from langgraph_sdk import get_client

    # This is the URL of the local development server
    URL = "http://127.0.0.1:2024"
    client = get_client(url=URL)

    # Search all hosted graphs
    assistants = await client.assistants.search()

    """Let's [stream `values`](https://docs.langchain.com/oss/python/langgraph/streaming#stream-graph-state), like before."""

    # Create a new thread
    thread = await client.threads.create()
    # Input message
    input_message = HumanMessage(content="Multiply 2 and 3")
    async for event in client.runs.stream(
        thread["thread_id"],
        assistant_id="agent",
        input={"messages": [input_message]},
        stream_mode="values",
    ):
        print(event)

    """The streamed objects have:

    * `event`: Type
    * `data`: State
    """

    from langchain_core.messages import convert_to_messages

    thread = await client.threads.create()
    input_message = HumanMessage(content="Multiply 2 and 3")
    async for event in client.runs.stream(
        thread["thread_id"],
        assistant_id="agent",
        input={"messages": [input_message]},
        stream_mode="values",
    ):
        messages = event.data.get("messages", None)
        if messages:
            print(convert_to_messages(messages)[-1])
        print("=" * 25)

    """
    There are some new streaming mode that are only supported via the API.
    For example, we can  [use `messages` mode](https://docs.langchain.com/oss/python/langgraph/streaming#supported-stream-modes) to better handle the above case!
    This mode currently assumes that you have a `messages` key in your graph, which is a list of messages.
    All events emitted using `messages` mode have two attributes:
    * `event`: This is the name of the event
    * `data`: This is data associated with the event
    """

    thread = await client.threads.create()
    input_message = HumanMessage(content="Multiply 2 and 3")
    async for event in client.runs.stream(
        thread["thread_id"],
        assistant_id="agent",
        input={"messages": [input_message]},
        stream_mode="messages",
    ):
        print(event.event)

    """
    We can see a few events:
    * `metadata`: metadata about the run
    * `messages/complete`: fully formed message
    * `messages/partial`: chat model tokens
    <!--You can dig further into the types [~here~](https://langchain-ai.github.io/langgraph/cloud/concepts/api/#modemessages) [here](https://docs.langchain.com/oss/python/langgraph/concepts/langgraph_server). -->
    Now, let's show how to stream these messages.
    We'll define a helper function for better formatting of the tool calls in messages.
    """

    thread = await client.threads.create()
    input_message = HumanMessage(content="Multiply 2 and 3")

    def format_tool_calls(tool_calls):
        """
        Format a list of tool calls into a readable string.

        Args:
            tool_calls (list): A list of dictionaries, each representing a tool call.
                Each dictionary should have 'id', 'name', and 'args' keys.

        Returns:
            str: A formatted string of tool calls, or "No tool calls" if the list is empty.

        """

        if tool_calls:
            formatted_calls = []
            for call in tool_calls:
                formatted_calls.append(
                    f"Tool Call ID: {call['id']}, Function: {call['name']}, Arguments: {call['args']}"
                )
            return "\n".join(formatted_calls)
        return "No tool calls"

    async for event in client.runs.stream(
        thread["thread_id"],
        assistant_id="agent",
        input={"messages": [input_message]},
        stream_mode="messages",
    ):

        # Handle metadata events
        if event.event == "metadata":
            print(f"Metadata: Run ID - {event.data['run_id']}")
            print("-" * 50)

        # Handle partial message events
        elif event.event == "messages/partial":
            for data_item in event.data:
                # Process user messages
                if "role" in data_item and data_item["role"] == "user":
                    print(f"Human: {data_item['content']}")
                else:
                    # Extract relevant data from the event
                    tool_calls = data_item.get("tool_calls", [])
                    invalid_tool_calls = data_item.get("invalid_tool_calls", [])
                    content = data_item.get("content", "")
                    response_metadata = data_item.get("response_metadata", {})

                    if content:
                        print(f"AI: {content}")

                    if tool_calls:
                        print("Tool Calls:")
                        print(format_tool_calls(tool_calls))

                    if invalid_tool_calls:
                        print("Invalid Tool Calls:")
                        print(format_tool_calls(invalid_tool_calls))

                    if response_metadata and response_metadata.get("finish_reason"):
                        print(
                            f"Response Metadata: Finish Reason - {response_metadata['finish_reason']}"
                        )
            print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(main_langgraph_dev())
    logger.info("Program Done.")
