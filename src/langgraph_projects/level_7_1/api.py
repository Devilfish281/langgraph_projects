# src/langgraph_projects/level-7/api.py
import os
from contextlib import asynccontextmanager
from typing import Annotated, TypedDict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from pydantic import BaseModel


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def add_bot_message(state: State):
    last_human = None
    for msg in reversed(state["messages"]):
        if getattr(msg, "type", None) == "human":
            last_human = msg.content
            break

    if last_human and "name" in last_human.lower():
        content = "You said your name is Matthew."
    else:
        content = "Hello from the graph!"

    return {"messages": [{"role": "assistant", "content": content}]}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("bot", add_bot_message)
    builder.add_edge(START, "bot")
    builder.add_edge("bot", END)
    return builder


@asynccontextmanager
async def lifespan(app: FastAPI):
    # -------- STARTUP (runs once) --------
    builder = build_graph()

    db_uri = os.getenv(
        "DB_URI",
        "postgresql://postgres:postgres@localhost:5432/langgraph?sslmode=disable",
    )

    # Open PostgresSaver context manager and keep it for app lifetime
    app.state.checkpointer_cm = PostgresSaver.from_conn_string(db_uri)
    app.state.checkpointer = app.state.checkpointer_cm.__enter__()

    # Create tables (safe to run at startup)
    app.state.checkpointer.setup()

    # Compile graph once
    app.state.graph = builder.compile(checkpointer=app.state.checkpointer)

    # App is now "running"
    yield

    # -------- SHUTDOWN (runs once) --------
    app.state.checkpointer_cm.__exit__(None, None, None)


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",  # our tiny web server in Step 3
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatIn(BaseModel):
    thread_id: str
    message: str


@app.post("/chat")
def chat(payload: ChatIn):
    config = {"configurable": {"thread_id": payload.thread_id}}
    result = app.state.graph.invoke(
        {"messages": [{"role": "user", "content": payload.message}]},
        config=config,
    )

    last = result["messages"][-1].content
    return {
        "reply": last,
        "messages": [m.content for m in result["messages"]],
    }
