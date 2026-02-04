# src/langgraph_projects/level-2/chatbot_summarization5.py

"""
Rather than just trimming or filtering messages, we'll show how to use LLMs to produce a running summary of the conversation.
This allows us to retain a compressed representation of the full conversation, rather than just removing it with trimming or filtering.
We'll incorporate this summarization into a simple Chatbot.
And we'll equip that Chatbot with memory, supporting long-running conversations without incurring high token cost / latency.
"""


# %pip install --quiet -U langchain_core langgraph langchain_openai

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


def messages_with_summary():
    """
    We'll use `MessagesState`, as before.
    In addition to the built-in `messages` key, we'll now include a custom key (`summary`).
    """

    # from langgraph.graph import MessagesState
    class State(MessagesState):
        summary: str

    # We'll define a node to call our LLM that incorporates a summary, if it exists, into the prompt.
    # from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
    # Define the logic to call the model
    def call_model_node(state: State):

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

        response = get_model().invoke(messages)
        return {"messages": response}

    """
    We'll define a node to produce a summary.
    Note, here we'll use `RemoveMessage` to filter our state after we've produced the summary.
    """

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

    """We'll add a conditional edge to determine whether to produce a summary based on the conversation length."""
    # from langgraph.graph import END
    # from typing_extensions import Literal

    # Determine whether to end or summarize the conversation
    def should_continue(state: State) -> Literal["summarize_conversation", END]:
        """Return the next node to execute."""

        messages = state["messages"]

        # If there are more than six messages, then we summarize the conversation
        if len(messages) > 6:
            return "summarize_conversation"

        # Otherwise we can just end
        return END

    """
    ## Adding memory
    Recall that [state is transient](https://github.com/langchain-ai/langgraph/discussions/352#discussioncomment-9291220) to a single graph execution.
    This limits our ability to have multi-turn conversations with interruptions.
    As introduced at the end of Module 1, we can use [persistence](https://docs.langchain.com/oss/python/langgraph/persistence) to address this!
    LangGraph can use a checkpointer to automatically save the graph state after each step.
    This built-in persistence layer provides memory, allowing LangGraph to resume from the last state update.
    As we previously showed, one of the easiest to work with is `MemorySaver`, an in-memory key-value store for Graph state.
    All we need to do is compile the graph with a checkpointer, and our graph has memory!
    """
    # from langgraph.checkpoint.memory import MemorySaver
    # from langgraph.graph import START, StateGraph

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
        output_name="agent6_graph_image.png",
    )

    """
    ## Threads
    The checkpointer saves the state at each step as a checkpoint.
    These saved checkpoints can be grouped into a `thread` of conversation.
    Think about Slack as an analog: different channels carry different conversations.
    Threads are like Slack channels, capturing grouped collections of state (e.g., conversation).
    Below, we use `configurable` to set a thread ID.
    ![state.jpg](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbadf3b379c2ee621adfd1_chatbot-summarization1.png)
    """
    # Create a thread
    config = {"configurable": {"thread_id": "1"}}

    # Start conversation
    input_message = HumanMessage(content="hi! I'm Matthew")

    logger.info("Invoking model...")
    output = graph.invoke({"messages": [input_message]}, config)
    logger.info("Model returned.")
    for m in output["messages"][-1:]:
        m.pretty_print()

    input_message = HumanMessage(content="what's my name?")
    logger.info("Invoking model...")
    output = graph.invoke({"messages": [input_message]}, config)
    logger.info("Model returned.")
    for m in output["messages"][-1:]:
        m.pretty_print()

    input_message = HumanMessage(content="i like the 49ers!")
    logger.info("Invoking model...")
    output = graph.invoke({"messages": [input_message]}, config)
    logger.info("Model returned.")
    for m in output["messages"][-1:]:
        m.pretty_print()
    logger.info("--- All messages so far: ---")
    for m in output["messages"]:
        m.pretty_print()
    """
    Now, we don't yet have a summary of the state because we still have < = 6 messages.
    This was set in `should_continue`.
        # If there are more than six messages, then we summarize the conversation
        if len(messages) > 6:
            return "summarize_conversation"
    We can pick up the conversation because we have the thread.
    """

    summary_output = graph.get_state(config).values.get("summary", "")
    logger.info(
        f"Current summary: {summary_output}, summary length: {len(summary_output)}  characters"
    )

    # The `config` with thread ID allows us to proceed from the previously logged state!

    input_message = HumanMessage(
        content="i like Nick Bosa, isn't he the highest paid defensive player?"
    )
    logger.info("Invoking model...")
    output = graph.invoke({"messages": [input_message]}, config)
    logger.info("Model returned.")
    for m in output["messages"][-1:]:
        m.pretty_print()
    logger.info("--- All messages so far: ---")
    for m in output["messages"]:
        m.pretty_print()

    summary_output = graph.get_state(config).values.get("summary", "")
    logger.info(
        f"Updated summary: {summary_output}, summary length: {len(summary_output)}  characters"
    )

    logger.info("Program Done.")
