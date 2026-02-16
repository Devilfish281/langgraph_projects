# src/langgraph_projects/level_7_2/api_memory_agent5_d.py
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from pydantic import BaseModel

# Import *functions*, not import-time graph instances
from langgraph_projects.level_7_2.memory_agent5_d import (
    get_builder,
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

    # --- Checkpointer (short-term / per-thread) ---
    # Open PostgresSaver context manager and keep it for app lifetime
    app.state.checkpointer_cm = PostgresSaver.from_conn_string(db_uri)
    app.state.checkpointer = app.state.checkpointer_cm.__enter__()

    # Create tables (safe to run at startup)
    app.state.checkpointer.setup()

    # --- Store (long-term / cross-thread) ---
    app.state.store_cm = PostgresStore.from_conn_string(db_uri)
    app.state.store = app.state.store_cm.__enter__()
    app.state.store.setup()

    app.state.builder = get_builder()
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
    app.state.graph = app.state.builder.compile(
        checkpointer=app.state.checkpointer,
        store=app.state.store,
    )
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
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


class ChatIn(BaseModel):
    thread_id: str
    user_id: str
    message: str


@app.post("/chat")
def chat(payload: ChatIn):
    config = {
        "configurable": {"thread_id": payload.thread_id, "user_id": payload.user_id}
    }

    result = app.state.graph.invoke(
        {"messages": [{"role": "user", "content": payload.message}]},
        config=config,
    )

    last = result["messages"][-1].content
    return {"reply": last}


@app.post("/stream")
def stream(payload: ChatIn):
    config = {
        "configurable": {"thread_id": payload.thread_id, "user_id": payload.user_id}
    }

    def ndjson_generator():
        # stream_mode="values" emits the full state after each step :contentReference[oaicite:2]{index=2}
        for chunk in app.state.graph.stream(
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
                    "role": getattr(msg, "type", None),  # e.g., "ai", "human", "tool"
                    "content": getattr(msg, "content", ""),
                },
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(ndjson_generator(), media_type="application/x-ndjson")


# @app.get("/todos/{user_id}")
# def get_todos(user_id: str):
#     namespace = ("todo", user_id)
#     items = app.state.store.search(namespace)

#     # items are StoreItem objects; value is what you stored (dict)
#     return {
#         "count": len(items),
#         "todos": [{"key": it.key, "value": it.value} for it in items],
#     }


@app.get("/todos/{user_id}")
def get_todos(user_id: str):
    items = app.state.store.search(("todo", user_id))
    return {
        "count": len(items),
        "todos": [{"key": it.key, "value": jsonable_encoder(it.value)} for it in items],
    }


# added Line
@app.get("/instructions/{user_id}")  # added Line
def get_instructions(user_id: str):  # added Line
    namespace = ("instructions", user_id)  # added Line

    # Your writer uses this fixed key: store.put(namespace, "user_instructions", {...})
    item = app.state.store.get(namespace, "user_instructions")  # added Line
    if item:  # added Line
        return {  # added Line
            "user_id": user_id,  # added Line
            "key": item.key,  # added Line
            "value": jsonable_encoder(item.value),  # added Line
        }  # added Line

    # Fallback: if you ever stored instructions under different keys
    items = app.state.store.search(namespace)  # added Line
    if not items:  # added Line
        raise HTTPException(
            status_code=404, detail="No instructions found for this user_id"
        )  # added Line

    return {  # added Line
        "user_id": user_id,  # added Line
        "count": len(items),  # added Line
        "instructions": [  # added Line
            {"key": it.key, "value": jsonable_encoder(it.value)}
            for it in items  # added Line
        ],  # added Line
    }  # added Line


@app.get("/profile/{user_id}")
def get_profile(user_id: str):
    namespace = ("profile", user_id)

    # Your profile writes can create multiple docs (uuid keys),
    # so return the full list and also a "latest" convenience field.
    items = app.state.store.search(namespace)

    if not items:
        raise HTTPException(status_code=404, detail="No profile found for user_id")

    # NOTE: ordering depends on store implementation; this is a convenience.
    latest = items[0]

    return {
        "user_id": user_id,
        "count": len(items),
        "latest": {"key": latest.key, "value": jsonable_encoder(latest.value)},
        "profiles": [
            {"key": it.key, "value": jsonable_encoder(it.value)} for it in items
        ],
    }
