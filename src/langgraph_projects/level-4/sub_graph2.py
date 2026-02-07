# src/langgraph_projects/level-4/sub_graph2.py
"""

Now, we're [going to cover sub-graphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs).
## State
Sub-graphs allow you to create and manage different states in different parts of your graph.
This is particularly useful for multi-agent systems, with teams of agents that each have their own state.

Let's consider a toy example:
* I have a system that accepts logs
* It performs two separate sub-tasks by different agents (summarize logs, find failure modes)
* I want to perform these two operations in two different sub-graphs.

The most critical thing to understand is how the graphs communicate!
In short, communication is **done with over-lapping keys**:
* The sub-graphs can access `docs` from the parent
* The parent can access `summary/failure_report` from the sub-graphs
## Input
Let's define a schema for the logs that will be input to our graph.
"""


# %pip install -U  langgraph


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
from operator import add
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, NotRequired, Optional, TypedDict

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


#####################################################################
### END
#####################################################################

"""We'll use [LangSmith](https://docs.langchain.com/langsmith/home) for [tracing](https://docs.langchain.com/langsmith/observability-concepts)."""
# from operator import add
# from typing import Annotated, List, Optional
# from typing_extensions import TypedDict


# The structure of the logs
class Log(TypedDict):
    id: str
    question: str
    docs: Optional[List]
    answer: str
    grade: Optional[int]
    grader: Optional[str]
    feedback: Optional[str]


###############################################################
# Sub-Graphs
###############################################################
"""#
Sub graphs
Here is the failure analysis sub-graph, which uses `FailureAnalysisState`.
"""


###############################################################
# failure analysis sub-graph
###############################################################

# # Entry Graph
# class EntryGraphState(TypedDict):
#     raw_logs: List[Log]
#     cleaned_logs: List[Log]
#     fa_summary: str  # This will only be generated in the FA sub-graph
#     report: str  # This will only be generated in the QS sub-graph
#     processed_logs: Annotated[
#         List[int], add
#     ]  # This will be generated in BOTH sub-graphs


# Failure Analysis Sub-graph
class FailureAnalysisState(TypedDict):
    cleaned_logs: List[Log]  # EntryGraphState Input
    failures: List[Log]
    fa_summary: str  # EntryGraphState Output
    processed_logs: List[str]  # EntryGraphState Output


# The output state of the failure analysis sub-graph. Note that it contains ALL KEYS, even if some are not modified in the sub-graph.
class FailureAnalysisOutputState(TypedDict):
    fa_summary: str
    processed_logs: List[str]


def get_failures_node(state):
    """Get logs that contain a failure"""
    cleaned_logs = state["cleaned_logs"]
    failures = [log for log in cleaned_logs if "grade" in log]
    return {"failures": failures}


# def generate_summary_node(state):
def fa_generate_summary_node(state):
    """Generate summary of failures"""
    failures = state["failures"]
    # Add fxn: fa_summary = summarize(failures)
    fa_summary = "Poor quality retrieval of Chroma documentation."
    return {
        "fa_summary": fa_summary,
        "processed_logs": [
            f"failure-analysis-on-log-{failure['id']}" for failure in failures
        ],
    }


def build_failure_analysis_subgraph():
    fa_builder = StateGraph(
        state_schema=FailureAnalysisState, output_schema=FailureAnalysisOutputState
    )
    fa_builder.add_node("get_failures", get_failures_node)
    fa_builder.add_node("generate_summary", fa_generate_summary_node)
    fa_builder.add_edge(START, "get_failures")
    fa_builder.add_edge("get_failures", "generate_summary")
    fa_builder.add_edge("generate_summary", END)

    graph = fa_builder.compile()
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
        output_name="failure_analysis_subgraph.png",
    )
    return graph


###############################################################
# summarization sub-graph
###############################################################
# # Entry Graph
# class EntryGraphState(TypedDict):
#     raw_logs: List[Log]
#     cleaned_logs: List[Log]
#     fa_summary: str  # This will only be generated in the FA sub-graph
#     report: str  # This will only be generated in the QS sub-graph
#     processed_logs: Annotated[
#         List[int], add
#     ]  # This will be generated in BOTH sub-graphs


# Here is the question summarization sub-graph, which uses `QuestionSummarizationState`.
# Summarization subgraph
class QuestionSummarizationState(TypedDict):
    cleaned_logs: List[Log]  # EntryGraphState Input
    qs_summary: str
    report: str  # EntryGraphState Output
    processed_logs: List[str]  # EntryGraphState Output


class QuestionSummarizationOutputState(TypedDict):
    report: str
    processed_logs: List[str]


# def generate_summary_node(state):
def qs_generate_summary_node(state):
    cleaned_logs = state["cleaned_logs"]
    # Add fxn: summary = summarize(generate_summary)
    summary = "Questions focused on usage of ChatOllama and Chroma vector store."
    return {
        "qs_summary": summary,
        "processed_logs": [f"summary-on-log-{log['id']}" for log in cleaned_logs],
    }


def send_to_slack_node(state):
    qs_summary = state["qs_summary"]
    # Add fxn: report = report_generation(qs_summary)
    report = "foo bar baz"
    return {"report": report}


def build_summarization_subgraph():
    qs_builder = StateGraph(
        QuestionSummarizationState, output_schema=QuestionSummarizationOutputState
    )
    qs_builder.add_node("generate_summary", qs_generate_summary_node)
    qs_builder.add_node("send_to_slack", send_to_slack_node)

    qs_builder.add_edge(START, "generate_summary")
    qs_builder.add_edge("generate_summary", "send_to_slack")
    qs_builder.add_edge("send_to_slack", END)

    graph = qs_builder.compile()
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
        output_name="question_summarization_subgraph.png",
    )
    return graph


###############################################################
# Parent Graph
###############################################################
"""## Adding sub graphs to our parent graph
Now, we can bring it all together.
We create our parent graph with `EntryGraphState`.
And we add our sub-graphs as nodes!
entry_builder.add_node("question_summarization", qs_builder.compile())
entry_builder.add_node("failure_analysis", fa_builder.compile())
"""


# # Entry Graph
# class EntryGraphState(TypedDict):
#     raw_logs: List[Log]
#     cleaned_logs: Annotated[List[Log], add]  # This will be USED BY in BOTH sub-graphs
#     fa_summary: str  # This will only be generated in the FA sub-graph
#     report: str  # This will only be generated in the QS sub-graph
#     processed_logs: Annotated[
#         List[int], add
#     ]  # This will be generated in BOTH sub-graphs


"""
But, why does `cleaned_logs` have a reducer if it only goes *into* each sub-graph as an input? It is not modified.
```
cleaned_logs: Annotated[List[Log], add] # This will be USED BY in BOTH sub-graphs
```
This is because the output state of the subgraphs will contain **all keys**, even if they are unmodified.
The sub-graphs are run in parallel.
Because the parallel sub-graphs return the same key, it needs to have a reducer like `operator.add` to combine the incoming values from each sub-graph.
But, we can work around this by using another concept we talked about before.
We can simply create an output state schema for each sub-graph and ensure that the output state schema contains different keys to publish as output.
We don't actually need each sub-graph to output `cleaned_logs`.
"""


# Entry Graph
class EntryGraphState(TypedDict):
    raw_logs: List[Log]
    cleaned_logs: List[Log]
    fa_summary: str  # This will only be generated in the FA sub-graph
    report: str  # This will only be generated in the QS sub-graph
    processed_logs: Annotated[
        List[str], add  # change to str from int
    ]  # This will be generated in BOTH sub-graphs


def clean_logs(state):
    # Get logs
    raw_logs = state["raw_logs"]
    # Data cleaning raw_logs -> docs
    cleaned_logs = raw_logs
    return {"cleaned_logs": cleaned_logs}


def build_app(*, use_checkpointer: bool = False):

    init_runtime()
    init_langsmith()
    entry_builder = StateGraph(EntryGraphState)
    entry_builder.add_node("clean_logs", clean_logs)
    entry_builder.add_node("question_summarization", build_summarization_subgraph())
    entry_builder.add_node("failure_analysis", build_failure_analysis_subgraph())

    entry_builder.add_edge(START, "clean_logs")
    entry_builder.add_edge("clean_logs", "failure_analysis")
    entry_builder.add_edge("clean_logs", "question_summarization")
    entry_builder.add_edge("failure_analysis", END)
    entry_builder.add_edge("question_summarization", END)

    if use_checkpointer:
        # Set up memory
        memory = MemorySaver()
        # Compile the graph with memory
        return entry_builder.compile(checkpointer=memory)
    else:
        return entry_builder.compile()


def get_graph_remote():
    return build_app(use_checkpointer=False)


graph_remote = get_graph_remote()
graph = None


async def run_langgraph_sdk_example():
    """
    ## Using with LangGraph API
    langgraph dev
    see Readme for more details on how to set up and run the LangGraph API locally.
    """

    from langgraph_sdk import get_client

    pass


def run_example():
    global graph
    if graph is None:  # Added Code
        raise RuntimeError(
            "Local graph is not initialized. Run under __main__."
        )  # Added Code

    # Thread
    thread = {"configurable": {"thread_id": "1"}}

    # Dummy logs
    question_answer = Log(
        id="1",
        question="How can I import ChatOllama?",
        answer="To import ChatOllama, use: 'from langchain_community.chat_models import ChatOllama.'",
    )

    question_answer_feedback = Log(
        id="2",
        question="How can I use Chroma vector store?",
        answer="To use Chroma, define: rag_chain = create_retrieval_chain(retriever, question_answer_chain).",
        grade=0,
        grader="Document Relevance Recall",
        feedback="The retrieved documents discuss vector stores in general, but not Chroma specifically",
    )

    raw_logs = [question_answer, question_answer_feedback]
    current_state = graph.invoke(
        {"raw_logs": raw_logs},
        config=thread,  # Added Code
    )
    logger.info("Current state: %s", current_state)
    log_state_time_travel(
        current_state, raw_flag=True, pretty_raw=True, label="Return state"
    )
    logger.info(f"fa_summary: {current_state.get('fa_summary')}")
    logger.info(f"report: {current_state.get('report')}")
    logger.info(f"processed_logs: {current_state.get('processed_logs')}")

    """
    class EntryGraphState(TypedDict):
        raw_logs: List[Log]
        cleaned_logs: List[Log]
        fa_summary: str  # This will only be generated in the FA sub-graph
        report: str  # This will only be generated in the QS sub-graph
        processed_logs: Annotated[List[str], add]  # This will be generated in BOTH sub-graphs    
    """
    logger.info("Example done.")
    """
    ## LangSmith
    Let's look at the LangSmith trace:
    https://smith.langchain.com/public/f8f86f61-1b30-48cf-b055-3734dfceadf2/r
    """


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
        output_name="parent_with_subgraphs.png",
    )
    ############################################################
    ## Run the dynamic breakpoints example
    ############################################################

    langgraph_dev = False
    if langgraph_dev:
        import asyncio

        asyncio.run(run_langgraph_sdk_example())

    else:
        run_example()

    logger.info("All done.")
