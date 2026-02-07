# src/langgraph_projects/level-4/research_assistant4.py
"""
My goal is to build a lightweight, multi-agent system around chat models that customizes the research process.
`Source Selection`
* Users can choose any set of input sources for their research.
`Planning`
* Users provide a topic, and the system generates a team of AI analysts, each focusing on one sub-topic.
* `Human-in-the-loop` will be used to refine these sub-topics before research begins.
`LLM Utilization`
* Each analyst will conduct in-depth interviews with an expert AI using the selected sources.
* The interview will be a multi-turn conversation to extract detailed insights as shown in the [STORM](https://arxiv.org/abs/2402.14207) paper.
* These interviews will be captured in a using `sub-graphs` with their internal state.
`Research Process`
* Experts will gather information to answer analyst questions in `parallel`.
* And all interviews will be conducted simultaneously through `map-reduce`.
`Output Format`
* The gathered insights from each interview will be synthesized into a final report.
* We'll use customizable prompts for the report, allowing for a flexible output format.
"""


# %pip install --quiet -U langgraph langchain_openai langchain_community langchain_core tavily-python wikipedia

"""## Setup"""
# We'll use [LangSmith](https://docs.langchain.com/langsmith/home) for [tracing](https://docs.langchain.com/langsmith/observability-concepts).

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

# from concurrent.futures import thread
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
    get_buffer_string,
)
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown

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

#####################################################################
# Generate Analysts: Human-In-The-Loop
#####################################################################
# Create analysts and review them using human-in-the-loop.

# from typing import List
# from pydantic import BaseModel, Field
# from typing_extensions import TypedDict
#####################################################################
# State and supporting models for analyst generation


# This is the model for an individual analyst, which includes their name, affiliation, role, and a description of their focus and motives. This structured format allows the LLM to generate detailed and consistent analyst personas.
class Analyst(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the analyst.",
    )
    name: str = Field(description="Name of the analyst.")
    role: str = Field(
        description="Role of the analyst in the context of the topic.",
    )
    description: str = Field(
        description="Description of the analyst focus, concerns, and motives.",
    )

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


# This is the structured output format for the analyst generation node. It allows us to enforce that the LLM outputs a list of analysts with the specified fields.
class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )


# Graph state for analyst generation
class GenerateAnalystsState(TypedDict):
    topic: str  # Research topic
    max_analysts: int  # Number of analysts
    human_analyst_feedback: str  # Human feedback
    analysts: List[Analyst]  # Analyst asking questions


# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import END, START, StateGraph
#####################################################################
# Prompt for creating analysts
analyst_instructions = """You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}

2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts:

{human_analyst_feedback}

3. Determine the most interesting themes based upon documents and / or feedback above.

4. Pick the top {max_analysts} themes.

5. Assign one analyst to each theme."""


def create_analysts_node(state: GenerateAnalystsState):
    """Create analysts"""

    topic = state["topic"]
    max_analysts = state["max_analysts"]
    human_analyst_feedback = state.get("human_analyst_feedback", "")

    # Enforce structured output
    structured_llm = get_model().with_structured_output(Perspectives)

    # System message
    system_message = analyst_instructions.format(
        topic=topic,
        human_analyst_feedback=human_analyst_feedback,
        max_analysts=max_analysts,
    )

    # Generate question
    analysts = structured_llm.invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content="Generate the set of analysts.")]
    )

    # Write the list of analysis to state
    return {"analysts": analysts.analysts}


def human_feedback_node(state: GenerateAnalystsState):
    # No-op node that should be interrupted on
    pass


def should_continue(state: GenerateAnalystsState):
    # Return the next node to execute

    # Check if human feedback
    human_analyst_feedback = state.get("human_analyst_feedback", None)
    if human_analyst_feedback:
        return "create_analysts"

    # Otherwise end
    return END


def generate_Analysts_graph(*, use_checkpointer: bool = False):
    # Add nodes and edges
    builder = StateGraph(GenerateAnalystsState)
    builder.add_node("create_analysts", create_analysts_node)
    builder.add_node("human_feedback", human_feedback_node)

    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges(
        "human_feedback", should_continue, ["create_analysts", END]
    )

    if use_checkpointer:
        # Set up memory
        memory = MemorySaver()
        # Compile the graph with memory
        return builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)
    else:
        return builder.compile(interrupt_before=["human_feedback"])


#####################################################################
# Conduct Interview START
#####################################################################
"""
## Conduct Interview
### Generate Question
The analyst will ask questions to the expert.
"""
# import operator
# from typing import Annotated
# from langgraph.graph import MessagesState
#####################################################################
# State and supporting models for interview graph


class InterviewState(MessagesState):
    max_num_turns: int  # Number turns of conversation
    context: Annotated[list, operator.add]  # Source docs
    analyst: Analyst  # Analyst asking questions
    interview: str  # Interview transcript
    sections: list  # Final key we duplicate in outer state for Send() API


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")


#####################################################################
# Prompt for generating questions from the analyst to the expert.
# The prompt emphasizes that the analyst should aim for interesting and specific insights,
#  and it encourages a multi-turn conversation to drill down into the topic.
question_instructions = """You are an analyst tasked with interviewing an expert to learn about a specific topic.
Your goal is boil down to interesting and specific insights related to your topic.
1. Interesting: Insights that people will find surprising or non-obvious.
2. Specific: Insights that avoid generalities and include specific examples from the expert.
Here is your topic of focus and set of goals: {goals}
Begin by introducing yourself using a name that fits your persona, and then ask your question.
Continue to ask questions to drill down and refine your understanding of the topic.
When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"
Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""


def generate_question_node(state: InterviewState):
    # Node to generate a question
    # Get state
    analyst = state["analyst"]
    messages = state["messages"]

    # Generate question
    system_message = question_instructions.format(goals=analyst.persona)
    question = get_model().invoke([SystemMessage(content=system_message)] + messages)

    # Write messages to state
    return {"messages": [question]}


"""
### Generate Answer: Parallelization
The expert will gather information from multiple sources in parallel to answer questions.
For example, we can use:
* Specific web sites e.g., via [`WebBaseLoader`](https://docs.langchain.com/oss/python/integrations/document_loaders/web_base)
* Indexed documents e.g., via [RAG](https://docs.langchain.com/oss/python/langchain/retrieval)
* Web search
* Wikipedia search
You can try different web search tools, like [Tavily](https://tavily.com/).
"""


tavily_search = TavilySearch(max_results=3)

# Wikipedia search tool
# from langchain_community.document_loaders import WikipediaLoader

"""
Now, we create nodes to search the web and wikipedia.
We'll also create a node to answer analyst questions.
Finally, we'll create nodes to save the full interview and to write a summary ("section") of the interview.
"""

# from langchain_core.messages import get_buffer_string
#####################################################################
# Prompt for search query writing
# Search query writing
search_instructions = SystemMessage(
    content=f"""You will be given a conversation between an analyst and an expert.
Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
First, analyze the full conversation.
Pay particular attention to the final question posed by the analyst.
Convert this final question into a well-structured web search query"""
)


def search_web_node(state: InterviewState):
    """Retrieve docs from web search"""

    # Search query
    structured_llm = get_model().with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions] + state["messages"])

    # Search
    # search_docs = tavily_search.invoke(search_query.search_query) # updated 1.0
    data = tavily_search.invoke({"query": search_query.search_query})
    search_docs = data.get("results", data)

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


def search_wikipedia_node(state: InterviewState):
    """Retrieve docs from wikipedia"""

    # Search query
    structured_llm = get_model().with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions] + state["messages"])

    # Search
    search_docs = WikipediaLoader(
        query=search_query.search_query, load_max_docs=2
    ).load()

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}


#####################################################################
# Prompt for answering questions from the analyst to the expert.
answer_instructions = """You are an expert being interviewed by an analyst.
Here is analyst area of focus: {goals}.
You goal is to answer a question posed by the interviewer.
To answer question, use this context:
{context}
When answering questions, follow these guidelines:
1. Use only the information provided in the context.
2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.
3. The context contain sources at the topic of each individual document.
4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1].
5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc
6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list:
[1] assistant/docs/llama3_1.pdf, page 7
And skip the addition of the brackets as well as the Document source preamble in your citation."""


def generate_answer_node(state: InterviewState):
    """Node to answer a question"""

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    # Answer question
    system_message = answer_instructions.format(goals=analyst.persona, context=context)
    answer = get_model().invoke([SystemMessage(content=system_message)] + messages)

    # Name the message as coming from the expert
    answer.name = "expert"

    # Append it to state
    return {"messages": [answer]}


def save_interview_node(state: InterviewState):
    """Save interviews"""

    # Get messages
    messages = state["messages"]

    # Convert interview to a string
    interview = get_buffer_string(messages)

    # Save to interviews key
    return {"interview": interview}


# This router checks the number of turns and the content of the last question to determine
# whether to continue the interview "ask_question" or to end it and save the interview "save_interview".
# route_messages, ["ask_question", "save_interview"]
def route_messages(state: InterviewState, name: str = "expert"):
    """Route between question and answer"""

    # Get messages
    messages = state["messages"]
    max_num_turns = state.get("max_num_turns", 2)

    # Check the number of expert answers
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return "save_interview"

    # This router is run after each question - answer pair
    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]

    if "Thank you so much for your help" in last_question.content:
        return "save_interview"
    return "ask_question"


#####################################################################
# Prompt for writing a section of the report based on the interview and source documents.
section_writer_instructions = """You are an expert technical writer.
Your task is to create a short, easily digestible section of a report based on a set of source documents.
1. Analyze the content of the source documents:
- The name of each source document is at the start of the document, with the <Document tag.
2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for sub-section headers
3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)
4. Make your title engaging based upon the focus area of the analyst:
{focus}
5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Create a numbered list of source documents, as you use them
- Do not mention the names of interviewers or experts
- Aim for approximately 400 words maximum
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
6. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
- It will look like:
### Sources
[1] Link or Document name
[2] Link or Document name
7. Be sure to combine sources. For example this is not correct:
[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/
There should be no redundant sources. It should simply be:
[3] https://ai.meta.com/blog/meta-llama-3-1/
8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""


def write_section_node(state: InterviewState):
    """Node to answer a question"""

    # Get state
    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]

    # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
    system_message = section_writer_instructions.format(focus=analyst.description)
    section = get_model().invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content=f"Use this source to write your section: {context}")]
    )

    # Append it to state
    return {"sections": [section.content]}


# create the graph for conducting interviews with experts based on analyst questions,
# with parallel retrieval from web and wikipedia, and
# a final node to write a section of a report based on the interview and source documents.
def generate_interview_graph(*, use_checkpointer: bool = False):
    # Add nodes and edges
    interview_builder = StateGraph(InterviewState)
    interview_builder.add_node("ask_question", generate_question_node)
    interview_builder.add_node("search_web", search_web_node)
    interview_builder.add_node("search_wikipedia", search_wikipedia_node)
    interview_builder.add_node("answer_question", generate_answer_node)
    interview_builder.add_node("save_interview", save_interview_node)
    interview_builder.add_node("write_section", write_section_node)
    # Flow
    interview_builder.add_edge(START, "ask_question")
    interview_builder.add_edge("ask_question", "search_web")
    interview_builder.add_edge("ask_question", "search_wikipedia")
    interview_builder.add_edge("search_web", "answer_question")
    interview_builder.add_edge("search_wikipedia", "answer_question")
    interview_builder.add_conditional_edges(
        "answer_question", route_messages, ["ask_question", "save_interview"]
    )
    interview_builder.add_edge("save_interview", "write_section")
    interview_builder.add_edge("write_section", END)

    # Interview
    if use_checkpointer:
        # Set up memory
        memory = MemorySaver()
        # Compile the graph with memory
        return interview_builder.compile(checkpointer=memory).with_config(
            run_name="Conduct Interviews"
        )
    else:
        return interview_builder.compile().with_config(run_name="Conduct Interviews")


#####################################################################
# Conduct Interview END
#####################################################################


def get_graph_remote():
    return generate_Analysts_graph(use_checkpointer=False)


graph_remote = get_graph_remote()
graph = None


# Interview graph
def get_graph_interview_remote():
    return generate_interview_graph(use_checkpointer=False)


interview_graph_remote = get_graph_interview_remote()
interview_graph = None

topic = "The benefits of adopting LangGraph as an agent framework"


def test_generate_interview(analysts):
    # analysts[0]
    logger.info(f"Testing interview graph with analyst: {analysts[0]}")

    # Here, we run the interview passing an index of the llama3.1 paper, which is related to our topic.

    messages = [HumanMessage(f"So you said you were writing an article on {topic}?")]
    thread = {"configurable": {"thread_id": "1"}}
    interview = interview_graph.invoke(
        {"analyst": analysts[0], "messages": messages, "max_num_turns": 2}, thread
    )
    # Markdown(interview["sections"][0])

    section_md = interview["sections"][0]
    logger.info("Interview section markdown:\n%s", section_md)

    console = Console()

    # Later, when you have the section text:
    section_md = interview["sections"][0]
    console.print(Markdown(section_md))


def test_generate_analysts():
    # Input
    max_analysts = 3

    thread = {"configurable": {"thread_id": "1"}}

    # Run the graph until the first interruption
    for event in graph.stream(
        {
            "topic": topic,
            "max_analysts": max_analysts,
        },
        thread,
        stream_mode="values",
    ):
        # Review
        analysts = event.get("analysts", "")
        if analysts:
            for analyst in analysts:
                logger.info(f"Name: {analyst.name}")
                logger.info(f"Affiliation: {analyst.affiliation}")
                logger.info(f"Role: {analyst.role}")
                logger.info(f"Description: {analyst.description}")
                logger.info("-" * 50)
    #######################################################
    # Interrupt and provide human feedback to the graph
    #######################################################
    # Get state and look at next node
    state = graph.get_state(thread)

    log_state_time_travel(
        state, raw_flag=True, pretty_raw=True, label="Interrupt state"
    )

    # state.next
    logger.info(f"Next node to execute: {state.next}")

    # We now update the state as if we are the human_feedback node
    graph.update_state(
        thread,
        {
            "human_analyst_feedback": "Add in someone from a startup to add an entrepreneur perspective"
        },
        as_node="human_feedback",
    )
    #######################################################
    # Continue the graph execution
    #######################################################
    for event in graph.stream(None, thread, stream_mode="values"):
        # Review
        analysts = event.get("analysts", "")
        if analysts:
            for analyst in analysts:
                logger.info(f"Name: {analyst.name}")
                logger.info(f"Affiliation: {analyst.affiliation}")
                logger.info(f"Role: {analyst.role}")
                logger.info(f"Description: {analyst.description}")
                logger.info("-" * 50)
    #######################################################
    # Interrupt If we are satisfied, then we simply supply no feedback
    #######################################################
    further_feedack = None
    graph.update_state(
        thread, {"human_analyst_feedback": further_feedack}, as_node="human_feedback"
    )
    #######################################################
    # Continue the graph execution
    #######################################################
    # Continue the graph execution to end
    for event in graph.stream(None, thread, stream_mode="updates"):
        logger.info("--Node--")
        node_name = next(iter(event.keys()))
        logger.info(node_name)

    final_state = graph.get_state(thread)
    log_state_time_travel(
        final_state, raw_flag=True, pretty_raw=True, label="Final state"
    )
    analysts = final_state.values.get("analysts")

    # final_state.next
    logger.info(f"Next node to execute: {final_state.next}")

    for analyst in analysts:
        logger.info(f"Name: {analyst.name}")
        logger.info(f"Affiliation: {analyst.affiliation}")
        logger.info(f"Role: {analyst.role}")
        logger.info(f"Description: {analyst.description}")
        logger.info("-" * 50)

    logger.info("Generated Analysts complete.")

    return analysts


async def run_langgraph_sdk_example():
    """
    ## Using with LangGraph API
    langgraph dev
    see Readme for more details on how to set up and run the LangGraph API locally.
    """

    from langgraph_sdk import get_client

    pass


# Only run demos when you execute the file directly (NOT when Studio imports it).
if __name__ == "__main__":
    # Demo / manual run (kept under __main__ so imports remain side-effect-light).

    graph = generate_Analysts_graph(use_checkpointer=True)

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
    interview_graph = generate_interview_graph(use_checkpointer=True)
    display_graph_if_enabled(
        interview_graph,
        save_env="VIEW_GRAPH",
        open_env="VIEW_GRAPH_OPEN",
        default_save=True,
        default_open=True,
        xray=1,
        images_dir_name="images",
        output_name="interview_graph.png",
    )
    ############################################################
    ## Run the dynamic breakpoints example
    ############################################################

    langgraph_dev = False
    if langgraph_dev:
        import asyncio

        asyncio.run(run_langgraph_sdk_example())

    else:
        analysts = test_generate_analysts()
        test_generate_interview(analysts)

    logger.info("All done.")
"""
## Studio

langgraph dev
"""
