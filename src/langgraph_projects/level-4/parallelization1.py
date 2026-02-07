# src/langgraph_projects/level-4/parallelization1.py
"""
This code will build on `human-in-the-loop` as well as the `memory` concepts discussed in level 2.
I will dive into `multi-agent` workflows and build up to a multi-agent research assistant that ties together all of the modules from this code.
To build this multi-agent research assistant, I'll first discuss a few LangGraph controllability topics.
I'll start with [parallelization](https://docs.langchain.com/oss/python/langgraph/how-tos/graph-api#create-branches).
## Fan out and fan in
Let's build a simple linear graph that over-writes the state at each step.
"""

# %pip install -U  langgraph langgraph_tavily wikipedia langchain_openai langchain_community langgraph_sdk

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
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, NotRequired, TypedDict

from langchain_community.document_loaders import WikipediaLoader
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


# a simple linear graph that over-writes the state at each step.
def simple_linear_graph():
    class State(TypedDict):
        # Note, no reducer function.
        state: List[str]

    class ReturnNodeValue:
        def __init__(self, node_secret: str):
            self._value = node_secret

        def __call__(self, state: State) -> Any:
            print(f"Adding {self._value} to {state['state']}")
            return {"state": [self._value]}

    # Add nodes
    builder = StateGraph(State)

    # Initialize each node with node_secret
    builder.add_node("a", ReturnNodeValue("I'm A"))
    builder.add_node("b", ReturnNodeValue("I'm B"))
    builder.add_node("c", ReturnNodeValue("I'm C"))
    builder.add_node("d", ReturnNodeValue("I'm D"))

    # Flow
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", "c")
    builder.add_edge("c", "d")
    builder.add_edge("d", END)
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
        output_name="simple_linear_graph.png",
    )

    current_state = graph.invoke({"state": []})
    """We over-write state, as expected."""
    logger.info(f"Current state: {current_state}")
    logger.info("Simple Linear Graph done.")


def parallel_graph():
    class State(TypedDict):
        # Note, no reducer function.
        state: List[str]

    class ReturnNodeValue:
        def __init__(self, node_secret: str):
            self._value = node_secret

        def __call__(self, state: State) -> Any:
            print(f"Adding {self._value} to {state['state']}")
            return {"state": [self._value]}

    ############################################################
    ## Run a graph in parallel with fan-out and fan-in
    ############################################################
    """Now, let's run `b` and `c` in parallel.
    And then run `d`.
    We can do this easily with fan-out from `a` to `b` and `c`, and then fan-in to `d`.
    The the state updates are applied at the end of each step.
    Let's run it.
    """
    builder = StateGraph(State)

    # Initialize each node with node_secret
    builder.add_node("a", ReturnNodeValue("I'm A"))
    builder.add_node("b", ReturnNodeValue("I'm B"))
    builder.add_node("c", ReturnNodeValue("I'm C"))
    builder.add_node("d", ReturnNodeValue("I'm D"))

    # Flow
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("a", "c")
    builder.add_edge("b", "d")
    builder.add_edge("c", "d")
    builder.add_edge("d", END)
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
        output_name="parallel_graph.png",
    )

    """
    We see an error**!
    This is because both `b` and `c` are writing to the same state key / channel in the same step.
    """
    from langgraph.errors import InvalidUpdateError

    try:
        graph.invoke({"state": []})
    except InvalidUpdateError as e:
        logger.error(f"An error occurred: {e}")

    """
    When using fan out, we need to be sure that we are using a reducer if steps are writing to the same the channel / key.
    As we touched on in level 2, `operator.add` is a function from Python's built-in operator module.
    When `operator.add` is applied to lists, it performs list concatenation.
    """
    logger.info("Parallel Graph done.")


# import operator
# from typing import Annotated


def parallel_graph_with_reducer():
    class State(TypedDict):
        # The operator.add reducer fn makes this append-only
        state: Annotated[list, operator.add]

    class ReturnNodeValue:
        def __init__(self, node_secret: str):
            self._value = node_secret

        def __call__(self, state: State) -> Any:
            print(f"Adding {self._value} to {state['state']}")
            return {"state": [self._value]}

    # Add nodes
    builder = StateGraph(State)

    # Initialize each node with node_secret
    builder.add_node("a", ReturnNodeValue("I'm A"))
    builder.add_node("b", ReturnNodeValue("I'm B"))
    builder.add_node("c", ReturnNodeValue("I'm C"))
    builder.add_node("d", ReturnNodeValue("I'm D"))

    # Flow
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("a", "c")
    builder.add_edge("b", "d")
    builder.add_edge("c", "d")
    builder.add_edge("d", END)
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
        output_name="parallel_graph_with_reducer.png",
    )

    # Now we see that we append to state for the updates made in parallel by `b` and `c`.
    # Waiting for nodes to finish
    current_state = graph.invoke({"state": []})
    # We add to the state, as expected.
    logger.info(f"Current state: {current_state}")
    logger.info("Parallel Graph with reducer   done.")


# Now, lets consider a case where one parallel path has more steps than the other one.
def parallel_graph_with_more_steps():
    class State(TypedDict):
        # The operator.add reducer fn makes this append-only
        state: Annotated[list, operator.add]

    class ReturnNodeValue:
        def __init__(self, node_secret: str):
            self._value = node_secret

        def __call__(self, state: State) -> Any:
            print(f"Adding {self._value} to {state['state']}")
            return {"state": [self._value]}

    builder = StateGraph(State)

    # Initialize each node with node_secret
    builder.add_node("a", ReturnNodeValue("I'm A"))
    builder.add_node("b", ReturnNodeValue("I'm B"))
    builder.add_node("b2", ReturnNodeValue("I'm B2"))
    builder.add_node("c", ReturnNodeValue("I'm C"))
    builder.add_node("d", ReturnNodeValue("I'm D"))

    # Flow
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("a", "c")
    builder.add_edge("b", "b2")
    builder.add_edge(["b2", "c"], "d")
    builder.add_edge("d", END)
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
        output_name="parallel_graph_with_more_steps.png",
    )

    """In this case, `b`, `b2`, and `c` are all part of the same step.
    The graph will wait for all of these to be completed before proceeding to step `d`.
    """

    current_state = graph.invoke({"state": []})
    """
    ## Setting the order of state updates
    However, within each step we don't have specific control over the order of the state updates!
    In simple terms, it is a deterministic order determined by LangGraph based upon graph topology that **we do not control**.
    Above, we see that `c` is added before `b2`.
    """
    logger.info(f"Current state: {current_state}")
    logger.info("Parallel Graph with more steps   done.")


# However, we can use a custom reducer to customize this e.g., sort state updates.
def sorting_reducer(left, right):
    """Combines and sorts the values in a list"""
    if not isinstance(left, list):
        left = [left]

    if not isinstance(right, list):
        right = [right]

    return sorted(left + right, reverse=False)


def parallel_graph_with_sorting_reducer():
    class State(TypedDict):
        # sorting_reducer will sort the values in state
        state: Annotated[list, sorting_reducer]

    class ReturnNodeValue:
        def __init__(self, node_secret: str):
            self._value = node_secret

        def __call__(self, state: State) -> Any:
            print(f"Adding {self._value} to {state['state']}")
            return {"state": [self._value]}

    # Add nodes
    builder = StateGraph(State)

    # Initialize each node with node_secret
    builder.add_node("a", ReturnNodeValue("I'm A"))
    builder.add_node("b", ReturnNodeValue("I'm B"))
    builder.add_node("b2", ReturnNodeValue("I'm B2"))
    builder.add_node("c", ReturnNodeValue("I'm C"))
    builder.add_node("d", ReturnNodeValue("I'm D"))

    # Flow
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("a", "c")
    builder.add_edge("b", "b2")
    builder.add_edge(["b2", "c"], "d")
    builder.add_edge("d", END)
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
        output_name="parallel_graph_with_sorting_reducer.png",
    )

    current_state = graph.invoke({"state": []})
    """
    No
    The `sorting_reducer` example sorts all values globally. We can also:
    1. Write outputs to a separate field in the state during the parallel step
    2. Use a "sink" node after the parallel step to combine and order those outputs
    3. Clear the temporary field after combining
    <!-- See the [~docs~](https://langchain-ai.github.io/langgraph/how-tos/branching/#stable-sorting) [docs](https://docs.langchain.com/oss/python/langgraph/how-tos/graph-api#create-branches) for more details.-->
    """
    logger.info(f"Current state: {current_state}")
    logger.info("Parallel Graph with more steps done.")


#########################################################
# Working with LLMs
##########################################################
# from langchain_community.document_loaders import WikipediaLoader
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain_tavily import TavilySearch  # updated since filming
"""
Now, lets add a realistic example!
We want to gather context from two external sources (Wikipedia and Web-Search) and have an LLM answer a question.
"""


# You can try different web search tools. [Tavily](https://tavily.com/) is one nice option to consider, but ensure your `TAVILY_API_KEY` is set.


def search_web_node(state):
    """Retrieve docs from web search"""
    logger.info("search_web_node")
    # Search
    tavily_search = TavilySearch(max_results=3)
    data = tavily_search.invoke({"query": state["question"]})
    search_docs = data.get("results", data)

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


def search_wikipedia_node(state):
    # Retrieve docs from Wikipedia and return them as a single formatted string in `context`.
    logger.info("search_wikipedia_node")
    query = state["question"]
    # Search
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}">\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


def generate_answer_node(state):
    """Node to answer a question"""
    #
    logger.info("generate_answer_node")

    # Get state
    context = state["context"]
    question = state["question"]

    # Template
    answer_template = """Answer the question {question} using this context: {context}"""
    answer_instructions = answer_template.format(question=question, context=context)

    # Answer
    answer = get_model().invoke(
        [SystemMessage(content=answer_instructions)]
        + [HumanMessage(content=f"Answer the question.")]
    )

    # Append it to state
    # return {"answer": answer}
    return {"answer": answer.content}


def build_app(*, use_checkpointer: bool = False):

    class State(TypedDict):
        question: str
        answer: str
        context: Annotated[list, operator.add]

    init_runtime()
    init_langsmith()

    # Add nodes
    builder = StateGraph(State)

    # Initialize each node with node_secret
    builder.add_node("search_web", search_web_node)
    builder.add_node("search_wikipedia", search_wikipedia_node)
    builder.add_node("generate_answer", generate_answer_node)

    # Flow
    builder.add_edge(START, "search_wikipedia")
    builder.add_edge(START, "search_web")
    builder.add_edge("search_wikipedia", "generate_answer")
    builder.add_edge("search_web", "generate_answer")
    builder.add_edge("generate_answer", END)

    if use_checkpointer:
        # Set up memory
        memory = MemorySaver()
        # Compile the graph with memory
        return builder.compile(checkpointer=memory)
    else:
        return builder.compile()


def get_graph_remote():
    return build_app(use_checkpointer=False)


graph_remote = get_graph_remote()
graph = None


def run_example():
    global graph
    if graph is None:  # Added Code
        raise RuntimeError(
            "Local graph is not initialized. Run under __main__."
        )  # Added Code

    """Let's run the graph!"""

    # Thread
    thread = {"configurable": {"thread_id": "1"}}

    current_state = graph.invoke(  # Changed Code
        {"question": "How were Nvidia's Q2 2025 earnings?"},
        config=thread,  # Added Code
    )

    logger.info(f"Current state: {current_state}")
    # current_state["answer"].content
    logger.info(f"Answer: {current_state['answer']}")
    logger.info("Example done.")


async def run_langgraph_sdk_example():
    """
    ## Using with LangGraph API
    langgraph dev
    see Readme for more details on how to set up and run the LangGraph API locally.
    """

    from langgraph_sdk import get_client

    client = get_client(url="http://127.0.0.1:2024")

    thread = await client.threads.create()
    input_question = {"question": "How were Nvidia Q2 2025 earnings?"}
    async for event in client.runs.stream(
        thread["thread_id"],
        assistant_id="parallelization",
        input=input_question,
        stream_mode="values",
    ):
        # Check if answer has been added to state
        if event.data is not None:
            answer = event.data.get("answer", None)
            if answer:
                print(answer["content"])


# Only run demos when you execute the file directly (NOT when Studio imports it).
if __name__ == "__main__":
    # Demo / manual run (kept under __main__ so imports remain side-effect-light).

    graph = build_app(use_checkpointer=True)

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
        output_name="web_search.png",
    )
    ############################################################
    ## Run the dynamic breakpoints example
    ############################################################

    langgraph_dev = False
    if langgraph_dev:
        import asyncio

        asyncio.run(run_langgraph_sdk_example())

    else:
        simple_linear_graph()
        parallel_graph()
        parallel_graph_with_reducer()
        parallel_graph_with_more_steps()
        parallel_graph_with_sorting_reducer()
        run_example()

    logger.info("All done.")
