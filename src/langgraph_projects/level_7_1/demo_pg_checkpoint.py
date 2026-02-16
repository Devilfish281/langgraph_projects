# src/langgraph_projects/level-7/demo_pg_checkpoint.py
import os
from typing import Annotated, TypedDict

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages


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


if __name__ == "__main__":
    DB_URI = os.getenv(
        "DB_URI",
        "postgresql://postgres:postgres@localhost:5432/langgraph?sslmode=disable",
    )
    builder = build_graph()

    # ✅ from_conn_string returns a context manager, so you MUST use "with"
    """
    You used:
    postgresql://postgres:postgres@localhost:5432/langgraph?sslmode=disable
    Break it down:
        postgresql:// → protocol (Postgres)
        postgres:postgres → username:password
        localhost:5432 → “my own computer” port 5432 (which Docker forwards into the container)
        /langgraph → the database name
        ?sslmode=disable → SSL setting (fine for local dev; many cloud DBs use require)    
    """
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        # Create the checkpoint tables once (safe to call at startup)
        # NOTE: autocommit matters for setup when managing connections yourself;
        # using from_conn_string handles the connection lifecycle cleanly.
        """
        This creates the tables you saw in Postgres:
            checkpoints
            checkpoint_writes
            checkpoint_blobs
            checkpoint_migrations
            etc.
        """
        checkpointer.setup()

        app = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "student-demo-NEW"}}
        print("\n" + "---" * 25)
        result1 = app.invoke(
            {"messages": [{"role": "user", "content": "Hi this is Matthew"}]},
            config=config,
        )
        print("\n" + "---" * 25)
        print("RUN 1 messages:", result1["messages"])
        print("\n" + "---" * 25)
        result2 = app.invoke(
            {"messages": [{"role": "user", "content": "Hi again. What is my name?"}]},
            config=config,
        )
        print("RUN 2 messages:", result2["messages"])
        print("\n" + "---" * 25)
