# -*- coding: utf-8 -*-
# Langchain-acadeemy/src/langchain_academy/level-1/chain4.py


# import threading
# from pathlib import Path
# from typing import Literal

# from IPython.display import display  # (optional; safe to keep if you use it)
# from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
# from langchain_core.tools import tool
# from langchain_openai import ChatOpenAI
# from langgraph.graph import END, START, MessagesState, StateGraph
# from langgraph.prebuilt import ToolNode
# from langsmith import utils
# from PIL import Image


"""# # Level 1: Chain 4 - Chat Models, Messages, and Tool Calling

Now, let's build up to a simple chain that combines 4 concepts.

* Using [chat messages](https://docs.langchain.com/oss/python/langchain/messages) as our graph state
* Using [chat models](https://docs.langchain.com/oss/python/integrations/chat) in graph nodes
* [Binding tools](https://docs.langchain.com/oss/python/langchain/models#tool-calling) to our chat model
* [Executing tool calls](https://docs.langchain.com/oss/python/langchain/models#tool-execution-loop) in graph nodes


"""


# %pip install --quiet -U langchain_openai langchain_core langgraph

###########################################################
## Supporting Code
###########################################################
import logging
import os
import threading
from pathlib import Path
from typing import Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from PIL import Image

from langgraph_projects.my_utils.load_env import load_dotenv_only, validate_environment
from langgraph_projects.my_utils.logger_setup import setup_logger

logger = logging.getLogger(__name__)  # Reuse the global logger
_MODEL = None
_LANGSMITH_ENABLED: bool = False
_LANGSMITH_INITIALIZED = False
_RUNTIME_INITIALIZED = False


_INIT_LOCK = threading.Lock()
_MODEL_LOCK = threading.Lock()


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
# We will default to `gpt-4o` because it offers a good balance of quality, price, and speed.

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
        - ``OPENAI_MODEL`` (defaults to ``gpt-4o``)

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
                    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
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
        os.environ["LANGCHAIN_PROJECT"] = "Router Agent Project"

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
            Image.open(output_file_path).show()
        except Exception:
            logger.exception("Saved graph PNG but failed to open viewer.")


#####################################################################
### END
#####################################################################


##############################################################
##############################################################
def multiply(a: int, b: int) -> int:
    """
    Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    value = a * b
    logger.info(f"Tool `multiply` called with a={a}, b={b}, result={value}")
    return value


from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage


def test_messages():
    """
    ## Messages

    Chat models can use [messages](https://docs.langchain.com/oss/python/langchain/messages), which capture different roles within a conversation.

    LangChain supports various message types, including `HumanMessage`, `AIMessage`, `SystemMessage`, and `ToolMessage`.

    These represent a message from the user, from chat model, for the chat model to instruct behavior, and from a tool call.

    Let's create a list of messages.

    Each message can be supplied with a few things:

    * `content` - content of the message
    * `name` - optionally, a message author
    * `response_metadata` - optionally, a dict of metadata (e.g., often populated by model provider for `AIMessages`)
    """

    messages = [
        AIMessage(
            content=f"So you said you were researching ocean mammals?", name="Model"
        )
    ]
    messages.append(HumanMessage(content=f"Yes, that's right.", name="Lance"))
    messages.append(
        AIMessage(content=f"Great, what would you like to learn about.", name="Model")
    )
    messages.append(
        HumanMessage(
            content=f"I want to learn about the best place to see Orcas in the US.",
            name="Lance",
        )
    )

    for m in messages:
        m.pretty_print()

    logger.info("Invoking model...")
    result = get_model().invoke(messages)
    logger.info("Model returned.")
    logger.info(f"Model invocation result: {result}")
    logger.info(
        f"Model invocation result metadata: {getattr(result, 'response_metadata', {})}"
    )


def test_tool_calling():
    """
    This only showing that the LLM can:

    see that a tool exists (multiply(a:int, b:int))

    decide “I should use that tool”

    return a message that contains a tool call request (name + arguments)
    """

    """
    ## Tools

    Tools are useful whenever you want a model to interact with external systems.

    External systems (e.g., APIs) often require a particular input schema or payload, rather than natural language.

    When we bind an API, for example, as a tool we given the model awareness of the required input schema.

    The model will choose to call a tool based upon the natural language input from the user.

    And, it will return an output that adheres to the tool's schema.

    [Many LLM providers support tool calling](https://docs.langchain.com/oss/python/integrations/chat) and
    [tool calling interface](https://blog.langchain.com/improving-core-tool-interfaces-and-docs-in-langchain/) in LangChain is simple.

    You can simply pass any Python `function` into `ChatModel.bind_tools(function)`.


    Let's showcase a simple example of tool calling!

    The `multiply` function is our tool.
    """

    # ------------------------------------------------------------
    # Bind tools (model can now REQUEST tool calls)
    # ------------------------------------------------------------
    llm_with_tools = get_model().bind_tools([multiply])

    """If we pass an input - e.g., `"What is 2 multiplied by 3"` - we see a tool call returned.

    The tool call has specific arguments that match the input schema of our function along with the name of the function to call.

    ```
    {'arguments': '{"a":2,"b":3}', 'name': 'multiply'}
    ```
    """

    logger.info("Invoking tool-enabled model...")
    tool_call_msg = llm_with_tools.invoke(
        [HumanMessage(content="What is 2 multiplied by 3", name="Lance")]
    )
    logger.info("Model returned.")
    logger.info(f"Model invocation result: {tool_call_msg}")
    logger.info(f"Tool calls requested by model: {tool_call_msg.tool_calls}")


from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def test_messages_list():
    """
    ## Using messages as state

    With these foundations in place, we can now use  [messages](https://docs.langchain.com/oss/python/langchain/overview#messages) in our graph state.

    Let's define our state, `MessagesState`, as a `TypedDict` with a single key: `messages`.

    `messages` is simply a list of messages, as we defined above (e.g., `HumanMessage`, etc).
    """

    class MessagesState(TypedDict):
        messages: list[AnyMessage]

    """## Reducers

    Now, we have a minor problem!

    As we discussed, each node will return a new value for our state key `messages`.

    But, this new value will overwrite the prior `messages` value!

    As our graph runs, we want to **append** messages to our `messages` state key.

    We can use [reducer functions](https://docs.langchain.com/oss/python/langgraph/graph-api#reducers) to address this.

    Reducers specify how state updates are performed.

    If no reducer function is specified, then it is assumed that updates to the key should *override it* as we saw before.

    But, to append messages, we can use the pre-built `add_messages` reducer.

    This ensures that any messages are appended to the existing list of messages.

    We simply need to annotate our `messages` key with the `add_messages` reducer function as metadata.
    """

    class MessagesState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    """Since having a list of messages in graph state is so common, LangGraph has a pre-built  [`MessagesState`](https://docs.langchain.com/oss/python/langgraph/graph-api#messagesstate)!

    `MessagesState` is defined:

    * With a pre-build single `messages` key
    * This is a list of `AnyMessage` objects
    * It uses the `add_messages` reducer

    We'll usually use `MessagesState` because it is less verbose than defining a custom `TypedDict`, as shown above.
    """

    class MessagesState(MessagesState):
        # Add any keys needed beyond messages, which is pre-built
        pass

    """To go a bit deeper, we can see how the `add_messages` reducer works in isolation."""

    # Initial state
    initial_messages = [
        AIMessage(content="Hello! How can I assist you?", name="Model"),
        HumanMessage(
            content="I'm looking for information on marine biology.", name="Lance"
        ),
    ]

    # New message to add
    new_message = AIMessage(
        content="Sure, I can help with that. What specifically are you interested in?",
        name="Model",
    )

    # Test
    add_messages(initial_messages, new_message)
    logger.info(f"Messages after adding new message: {initial_messages}")


def build_app():
    init_runtime()
    init_langsmith()

    test_messages()
    test_tool_calling()
    test_messages_list()

    # ------------------------------------------------------------
    # LangGraph tool execution loop (THIS is what actually RUNS multiply)
    # ------------------------------------------------------------

    llm_with_tools = get_model().bind_tools([multiply])

    # Node: call the tool-enabled model
    def tool_calling_llm_node(state: MessagesState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # tool execution node (this is what actually calls multiply)
    tool_node = ToolNode([multiply])

    #  router decides whether to execute tools or stop
    # END is  "__end__"
    def should_continue(
        state: MessagesState,
    ) -> Literal["tool_node", "__end__"]:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tool_node"
        return END

    # Build graph
    builder = StateGraph(MessagesState)
    builder.add_node("tool_calling_llm", tool_calling_llm_node)  # (unchanged)
    builder.add_node("tool_node", tool_node)

    builder.add_edge(START, "tool_calling_llm")  # (unchanged)

    #  conditional route to tool execution when the model requests it
    builder.add_conditional_edges("tool_calling_llm", should_continue)

    #  after running the tool, go back to the LLM (tool loop)
    builder.add_edge("tool_node", "tool_calling_llm")

    graph = builder.compile()  # (unchanged)

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
        output_name="chain4_graph_image.png",
    )

    #######################################################
    """## Our graph

    Now, lets use `MessagesState` with a graph.
    """

    # ------------------------------------------------------------
    # Run graph: no tool call
    # ------------------------------------------------------------
    logger.debug("Invoking model...")
    messages = graph.invoke(
        {"messages": [HumanMessage(content="Hello!")]}
    )  #  always pass a list
    logger.debug("Model returned.")
    for m in messages["messages"]:
        m.pretty_print()

    # ------------------------------------------------------------
    # Run graph: SHOULD call multiply (ToolNode executes it)
    # ------------------------------------------------------------
    logger.debug("Invoking model...")
    messages = graph.invoke(
        {"messages": [HumanMessage(content="Multiply 2 and 3")]}
    )  #  always pass a list
    logger.debug("Model returned.")
    for m in messages["messages"]:
        m.pretty_print()

    print("Program Done.")
