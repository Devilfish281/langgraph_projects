# langgraph_projects/basics/basics.py
###########################################################
## Supporting Code
###########################################################
import logging
import os
import threading
from pathlib import Path
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
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
    """Create the LLM client lazily and return a cached instance.

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
    """Initialize LangSmith/LangChain tracing configuration (optional).

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

from langchain_core.messages import HumanMessage


def build_app():
    """Build and compile the LangGraph agent executor application.

    The graph follows a simple loop:

    1. ``llm_call``: The model decides whether to answer directly or emit tool calls.
    2. Conditional routing:
       - If tool calls exist, route to ``tool_node``.
       - Otherwise, route to ``__end__``.
    3. ``tool_node``: Executes tools and returns results to the message state.
    4. Loop back to ``llm_call``.

    Tools
    -----
    - :class:`langchain_tavily.TavilySearch` (requires ``TAVILY_API_KEY``)

    :returns: A compiled LangGraph application (executable graph).
    :rtype: typing.Any
    """
    init_runtime()
    init_langsmith()

    # Create a message
    msg = HumanMessage(content="Hello world", name="Lance")

    # Message list
    messages = [msg]

    # Chat models in LangChain have a number of [default methods](https://reference.langchain.com/python/langchain_core/runnables).
    # For the most part, we'll be using:
    # * [stream](https://docs.langchain.com/oss/python/langchain/models#stream): stream back chunks of the response
    # * [invoke](https://docs.langchain.com/oss/python/langchain/models#invoke): call the chain on an input

    # And, as mentioned, chat models take [messages](https://docs.langchain.com/oss/python/langchain/messages) as input.
    # Messages have a role (that describes who is saying the message) and a content property.
    # We'll be talking a lot more about this later, but here let's just show the basics.

    # Invoke the model with a list of messages
    logger.info("Invoking model...")
    result = get_model().invoke(messages)
    logger.info("Model returned.")
    logger.info(f"Model invocation result: {result}")

    # We get an `AIMessage` response. Also, note that we can just invoke a chat model with a string.
    # When a string is passed in as input, it is converted to a `HumanMessage` and then passed to the underlying model.
    logger.info("Invoking model...")
    result = get_model().invoke("hello world")
    logger.info("Model returned.")
    logger.info(f"Model invocation result: {result}")

    # Search Tools

    # You'll also see [Tavily](https://tavily.com/) in the README, which is a search engine optimized for LLMs and RAG, aimed at efficient,
    # quick, and persistent search results. As mentioned, it's easy to sign up and offers a generous free tier.
    # Some lessons (in Module 4) will use Tavily by default but, of course,
    # other search tools can be used if you want to modify the code for yourself.
    tavily_search = TavilySearch(max_results=3)

    data = tavily_search.invoke({"query": "What is LangGraph?"})
    search_docs = data.get("results", data)
    logger.info(f"Tavily Search returned {len(search_docs)} results.")
    for i, doc in enumerate(search_docs, start=1):
        logger.info(f"Result {i}: {doc}")

    print("app Done.")


if __name__ == "__main__":
    # Demo / manual run (kept under __main__ so imports remain side-effect-light).
    app = build_app()
    print("Program Done.")
