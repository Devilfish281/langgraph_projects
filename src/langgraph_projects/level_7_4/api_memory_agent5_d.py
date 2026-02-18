# src/langgraph_projects/level_7_4/api_memory_agent5_d.py
import json
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.redis import (
    RedisSaver,
)  # add (exact import path depends on package version)
from langgraph.store.postgres import PostgresStore
from pydantic import BaseModel

# Import *functions*, not import-time graph instances
from langgraph_projects.level_7_4.memory_agent5_d import (
    get_builder_personal,
    get_builder_work,
    init_langsmith,
    init_runtime,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # -------- STARTUP (runs once) --------
    # Startup
    init_runtime()
    init_langsmith()

    db_uri = os.getenv(
        "DB_URI",
        "postgresql://postgres:postgres@localhost:5432/langgraph?sslmode=disable",
    )

    redis_uri = os.getenv("REDIS_URI", "redis://localhost:6379")

    # --- Checkpointer (short-term / per-thread) ---
    # Open RedisSaver context manager and keep it for app lifetime
    app.state.checkpointer_cm = RedisSaver.from_conn_string(redis_uri)

    # Open PostgresSaver context manager and keep it for app lifetime
    # app.state.checkpointer_cm = PostgresSaver.from_conn_string(db_uri)

    app.state.checkpointer = app.state.checkpointer_cm.__enter__()

    # Create tables (safe to run at startup)
    app.state.checkpointer.setup()

    # --- Store (long-term / cross-thread) ---
    app.state.store_cm = PostgresStore.from_conn_string(db_uri)
    app.state.store = app.state.store_cm.__enter__()
    app.state.store.setup()

    app.state.builders = {
        "personal": get_builder_personal(),
        "work": get_builder_work(),
    }
    # Compile graph once
    """
        # Store for long-term (across-thread) memory
        across_thread_memory = InMemoryStore()

        # Checkpointer for short-term (within-thread) memory
        within_thread_memory = MemorySaver()
        graph = builder.compile(
            checkpointer=within_thread_memory, store=across_thread_memory
        )
    """
    app.state.graphs = {
        kind: builder.compile(
            checkpointer=app.state.checkpointer,  # Redis
            store=app.state.store,  # Postgres
        )
        for kind, builder in app.state.builders.items()
    }

    # App is now "running"
    yield

    # Shutdown
    # -------- SHUTDOWN (runs once) --------
    app.state.store_cm.__exit__(None, None, None)
    app.state.checkpointer_cm.__exit__(None, None, None)


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",  # our tiny web server in Step 3
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


class ChatIn(BaseModel):
    thread_id: str
    user_id: str
    todo_kind: Literal["personal", "work"]
    message: str


def _pick_graph(todo_kind: str):
    graph = app.state.graphs.get(todo_kind)
    if graph is None:
        raise HTTPException(400, detail=f"Invalid todo_kind: {todo_kind}")
    return graph


@app.post("/chat")
def chat(payload: ChatIn):
    config = {
        "configurable": {"thread_id": payload.thread_id, "user_id": payload.user_id}
    }

    graph = _pick_graph(payload.todo_kind)

    result = graph.invoke(
        {"messages": [{"role": "user", "content": payload.message}]},
        config=config,
    )

    last = result["messages"][-1].content
    return JSONResponse(  # Added Code
        content={"reply": last, "todo_kind": payload.todo_kind},  # Added Code
        media_type="application/json; charset=utf-8",  # Added Code
    )  # Added Code
    # return {"reply": last, "todo_kind": payload.todo_kind}


@app.post("/stream")
def stream(payload: ChatIn):
    config = {
        "configurable": {"thread_id": payload.thread_id, "user_id": payload.user_id}
    }

    graph = _pick_graph(payload.todo_kind)

    def ndjson_generator():
        # stream_mode="values" emits the full state after each step :contentReference[oaicite:2]{index=2}
        for chunk in graph.stream(
            {"messages": [{"role": "user", "content": payload.message}]},
            config=config,
            stream_mode="values",
        ):
            # chunk is a state dict; messages are usually in chunk["messages"] for MessagesState graphs
            msg = chunk["messages"][-1]

            # Send one JSON object per line (NDJSON)
            yield json.dumps(
                {
                    "event": "chunk",
                    "todo_kind": payload.todo_kind,
                    "role": getattr(msg, "type", None),  # e.g., "ai", "human", "tool"
                    "content": getattr(msg, "content", ""),
                },
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(
        ndjson_generator(),
        media_type="application/x-ndjson; charset=utf-8",
    )
    # return StreamingResponse(ndjson_generator(), media_type="application/x-ndjson")


# @app.get("/todos/{user_id}")
# def get_todos(user_id: str):
#     namespace = ("todo", user_id)
#     items = app.state.store.search(namespace)

#     # items are StoreItem objects; value is what you stored (dict)
#     return {
#         "count": len(items),
#         "todos": [{"key": it.key, "value": it.value} for it in items],
#     }


@app.get("/todos/{todo_kind}/{user_id}")  # change Line
def get_todos(todo_kind: Literal["personal", "work"], user_id: str):  # change Line
    items = app.state.store.search(("todo", todo_kind, user_id))  # change Line
    return {
        "todo_kind": todo_kind,
        "count": len(items),
        "todos": [{"key": it.key, "value": jsonable_encoder(it.value)} for it in items],
    }


@app.get("/instructions/{todo_kind}/{user_id}")  # change Line
def get_instructions(  # change Line
    todo_kind: Literal["personal", "work"], user_id: str  # change Line
):
    namespace = ("instructions", todo_kind, user_id)  # change Line

    # writer uses: store.put(namespace, "user_instructions", {...})
    item = app.state.store.get(namespace, "user_instructions")
    if item:
        return {
            "todo_kind": todo_kind,
            "user_id": user_id,
            "key": item.key,
            "value": jsonable_encoder(item.value),
        }

    items = app.state.store.search(namespace)
    if not items:
        raise HTTPException(
            status_code=404, detail="No instructions found for this user_id/todo_kind"
        )

    return {
        "todo_kind": todo_kind,
        "user_id": user_id,
        "count": len(items),
        "instructions": [
            {"key": it.key, "value": jsonable_encoder(it.value)} for it in items
        ],
    }


@app.get("/profile/{user_id}")
def get_profile(user_id: str):
    namespace = ("profile", user_id)

    items = app.state.store.search(namespace)
    if not items:
        raise HTTPException(status_code=404, detail="No profile found for user_id")

    latest = items[0]

    return {
        "user_id": user_id,
        "count": len(items),
        "latest": {"key": latest.key, "value": jsonable_encoder(latest.value)},
        "profiles": [
            {"key": it.key, "value": jsonable_encoder(it.value)} for it in items
        ],
    }
