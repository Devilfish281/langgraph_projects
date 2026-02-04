# src/langgraph_projects/level-2/trim_filter_messages4.py
"""
Now, I can start using these concepts with models in LangGraph!
In the next few examples, I'll build towards a chatbot that has long-term memory.
Because my chatbot will use messages, let's first talk a bit more about advanced ways to work with messages in graph state.
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
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from PIL import Image
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


###################################
# Messages as state
###################################
def messages_example():
    """
    ## Messages as state
    First, let's define some messages.
    """

    # from pprint import pprint
    # from langchain_core.messages import AIMessage, HumanMessage
    messages = [
        AIMessage(f"So you said you Ire researching ocean mammals?", name="Bot")
    ]
    messages.append(
        HumanMessage(
            f"Yes, I know about whales. But what others should I learn about?",
            name="Matthew",
        )
    )

    for m in messages:
        m.pretty_print()

    """Recall I can pass them to a chat model."""
    logger.info("Invoking model...")
    response = get_model().invoke(messages)
    logger.info("Model returned.")
    logger.debug("Model output messages: %s", response)
    # If you want to print/log the whole “conversation so far”
    messages_out = messages + [response]
    for m in messages_out:
        m.pretty_print()


###################################
# Messages as State in Graph
###################################
def messages_as_state():
    """We can run our chat model in a simple graph with `MessagesState`."""
    messages = [
        AIMessage(f"So you said you Ire researching ocean mammals?", name="Bot")
    ]
    messages.append(
        HumanMessage(
            f"Yes, I know about whales. But what others should I learn about?",
            name="Matthew",
        )
    )

    # Node
    def chat_model_node(state: MessagesState):
        return {"messages": get_model().invoke(state["messages"])}

    # Build graph
    builder = StateGraph(MessagesState)
    builder.add_node("chat_model", chat_model_node)
    builder.add_edge(START, "chat_model")
    builder.add_edge("chat_model", END)
    graph = builder.compile()

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
        output_name="messages_as_state.png",
    )
    logger.info("Invoking model...")
    output = graph.invoke({"messages": messages})
    logger.info("Model returned.")
    # logger.debug("Model output messages: %s", output)
    for m in output["messages"]:
        m.pretty_print()


################################
# Reducers
################################
def messages_with_reducer():
    """## Reducer

    A practical challenge when working with messages is managing long-running conversations.

    Long-running conversations result in high token usage and latency if I are not careful, because I pass a growing list of messages to the model.

    We have a few ways to address this.

    First, recall the trick I saw using `RemoveMessage` and the `add_messages` reducer.
    """

    from langchain_core.messages import RemoveMessage

    # Nodes
    def filter_messages(state: MessagesState):
        # Delete all but the 2 most recent messages
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
        return {"messages": delete_messages}

    def chat_model_node(state: MessagesState):
        return {"messages": [get_model().invoke(state["messages"])]}

    # Build graph
    builder = StateGraph(MessagesState)
    builder.add_node("filter", filter_messages)
    builder.add_node("chat_model", chat_model_node)
    builder.add_edge(START, "filter")
    builder.add_edge("filter", "chat_model")
    builder.add_edge("chat_model", END)
    graph = builder.compile()

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
        output_name="reducers.png",
    )

    # Message list with a preamble
    messages = [AIMessage("Hi.", name="Bot", id="1")]
    messages.append(HumanMessage("Hi.", name="Matthew", id="2"))
    messages.append(
        AIMessage("So you said you Ire researching ocean mammals?", name="Bot", id="3")
    )
    messages.append(
        HumanMessage(
            "Yes, I know about whales. But what others should I learn about?",
            name="Matthew",
            id="4",
        )
    )

    # Invoke
    output = graph.invoke({"messages": messages})
    for m in output["messages"]:
        m.pretty_print()

    ##################################
    # Filtering messages
    ##################################

    """## Filtering messages

    If you don't need or want to modify the graph state, you can just filter the messages you pass to the chat model.

    For example, just pass in a filtered list: `get_model().invoke(messages[-1:])` to the model.
    """

    # Node
    def chat_model_node(state: MessagesState):
        return {"messages": [get_model().invoke(state["messages"][-1:])]}

    # Build graph
    builder = StateGraph(MessagesState)
    builder.add_node("chat_model", chat_model_node)
    builder.add_edge(START, "chat_model")
    builder.add_edge("chat_model", END)
    graph = builder.compile()
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
        output_name="filtering_messages.png",
    )

    """Let's take our existing list of messages, append the above LLM response, and append a follow-up question."""

    messages.append(output["messages"][-1])
    messages.append(HumanMessage(f"Tell me more about Narwhals!", name="Matthew"))

    for m in messages:
        m.pretty_print()

    # Invoke, using message filtering
    logger.info("Invoking model with message filtering...")
    output = graph.invoke({"messages": messages})
    logger.info("Model returned.")

    for m in output["messages"]:
        m.pretty_print()

    """The state has all of the mesages.

    But, let's look at the LangSmith trace to see that the model invocation only uses the last message:

    https://smith.langchain.com/public/75aca3ce-ef19-4b92-94be-0178c7a660d9/r
    """
    ##################################
    # Trimming messages
    ##################################
    """
    ##
    # Trim messages
    Another approach is to [trim messages](https://docs.langchain.com/oss/python/langgraph/add-memory#trim-messages), based upon a set number of tokens.
    This restricts the message history to a specified number of tokens.
    While filtering only returns a post-hoc subset of the messages betIen agents, trimming restricts the number of tokens that a chat model can use to respond.
    See the `trim_messages` below.
    """

    from langchain_core.messages import trim_messages

    # Node
    def chat_model_node(state: MessagesState):
        messages = trim_messages(
            state["messages"],
            max_tokens=100,
            strategy="last",
            token_counter=ChatOpenAI(model="gpt-5.1"),
            allow_partial=False,
        )
        return {"messages": get_model().invoke(messages)}

    # Build graph
    builder = StateGraph(MessagesState)
    builder.add_node("chat_model", chat_model_node)
    builder.add_edge(START, "chat_model")
    builder.add_edge("chat_model", END)
    graph = builder.compile()

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
        output_name="trimming_messages.png",
    )

    messages.append(output["messages"][-1])
    messages.append(HumanMessage(f"Tell me where Orcas live!", name="Matthew"))
    ##################################
    # Example of trimming messages
    ###################################
    messages_trim = trim_messages(
        messages,
        max_tokens=100,
        strategy="last",
        token_counter=ChatOpenAI(model="gpt-5.1"),
        allow_partial=False,
    )

    for m in messages_trim:
        m.pretty_print()

    # Invoke, using message trimming in the chat_model_node
    logger.info("Invoking model with message trimming...")
    messages_out_trim = graph.invoke({"messages": messages})
    logger.info("Model returned.")
    for m in messages_out_trim["messages"]:
        m.pretty_print()

    return graph


##############################################
# Build App
##############################################
def build_app():

    init_runtime()
    init_langsmith()

    messages_example()
    messages_as_state()
    graph = messages_with_reducer()

    return graph


graph = build_app()


# Only run demos when you execute the file directly (NOT when Studio imports it).
if __name__ == "__main__":

    print("Program Done.")
"""
Let's look at the LangSmith trace to see the model invocation:
https://smith.langchain.com/
"""
