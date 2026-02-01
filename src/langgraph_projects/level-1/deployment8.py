# -*- coding: utf-8 -*-
# Langchain-acadeemy/src/langchain_academy/level-1/deployment8.py

"""
## Goals

Now, we'll cover how to actually deploy our agent locally to Studio and to `LangGraph Cloud`.
"""

# %pip install --quiet -U langgraph_sdk langchain_core

"""
## Concepts

There are a few central concepts to understand -

`LangGraph` —
- Python and JavaScript library
- Allows creation of agent workflows

`LangGraph API` —
- Bundles the graph code
- Provides a task queue for managing asynchronous operations
- Offers persistence for maintaining state across interactions

`LangSmith Deployment` (formerly `LangGraph Cloud`) --
- Hosted service for the LangGraph API
- Allows deployment of graphs from GitHub repositories
- Also provides monitoring and tracing for deployed graphs
- Accessible via a unique URL for each deployment

`LangSmith Studio` (formerly `LangGraph Studio`) --
- Integrated Development Environment (IDE) for LangGraph applications
- Uses the API as its back-end, allowing real-time testing and exploration of graphs
- Can be run locally or with cloud-deployment. See below.

`LangGraph SDK` --
- Python library for programmatically interacting with LangGraph graphs
- Provides a consistent interface for working with graphs, whether served locally or in the cloud
- Allows creation of clients, access to assistants, thread management, and execution of runs

## Running Locally
We can run our agent locally via the LangGraph API.

"""

import asyncio
import getpass
import json
import os
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph_sdk import get_client


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _set_env(var: str) -> None:
    """Prompt for an env var if it's missing."""
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


async def _assert_local_server_is_up(base_url: str) -> None:
    """Fail fast with a friendly message if the local server isn't reachable."""
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'httpx'. Install it in your Poetry env:\n"
            "  poetry add httpx\n"
            "Then re-run this script."
        ) from exc

    docs_url = f"{base_url.rstrip('/')}/docs"
    try:
        async with httpx.AsyncClient(timeout=2.5) as client:
            resp = await client.get(docs_url)
            resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            "Cannot reach your local LangGraph dev server.\n\n"
            f"Tried: {docs_url}\n\n"
            "Fix steps:\n"
            "1) Open a NEW terminal in your project and run `langgraph dev` (usually in the /studio directory).\n"
            "2) Wait until it prints the API URL (commonly http://127.0.0.1:2024).\n"
            "3) Paste that printed API URL into BASE_URL below.\n"
            "4) Confirm the docs page opens in a browser: http://127.0.0.1:2024/docs\n"
        ) from exc


def _print_stream_chunk(chunk: Any) -> None:
    """Print stream chunks safely for stream_mode='values'."""
    data = getattr(chunk, "data", None)
    if not isinstance(data, dict):
        print("chunk.data:", data)
        print("---")
        return

    # handle your simple_graph2.py shape
    if "graph_state" in data:
        print("graph_state:", data.get("graph_state"))
        print("random_value:", data.get("random_value"))
        print("---")
        return

    # Existing behavior: handle chat-style graphs
    msgs = data.get("messages")
    if isinstance(msgs, list) and msgs:
        print(msgs[-1])
    else:
        print("chunk.data keys:", list(data.keys()))
        print(json.dumps(data, indent=2, default=str))
    print("---")


async def run_local() -> None:
    """Run the agent locally via the LangGraph API."""

    # This is the URL of the local development server
    # Make sure it's running via `langgraph dev` in your terminal

    URL = "http://127.0.0.1:2024"
    await _assert_local_server_is_up(URL)
    client = get_client(url=URL)

    # Search all hosted graphs (assistants)
    assistants = await client.assistants.search()
    print(f"Found {len(assistants)} assistants on local server.")

    if not assistants:
        raise RuntimeError(
            "No assistants found on the local server.\n"
            "This usually means your local server is running, but it isn't serving any graphs yet.\n"
            "Double-check you started `langgraph dev` in the correct /studio folder for this module."
        )

    # Pretty-print each assistant
    for idx, a in enumerate(assistants):
        # assistants are typically dict-like objects
        assistant_id = a.get("assistant_id") or a.get("id")
        name = a.get("name")
        graph_id = a.get("graph_id")
        updated_at = a.get("updated_at") or a.get("modified_at")
        created_at = a.get("created_at")

        print(f"--- Assistant #{idx} ---")
        print(f"id:        {assistant_id}")
        print(f"name:      {name}")
        print(f"graph_id:   {graph_id}")
        print(f"created_at: {created_at}")
        print(f"updated_at: {updated_at}")

        # Full raw payload (useful when you’re not sure what fields exist)
        print("raw:")
        print(json.dumps(a, indent=2, default=str))
        print()

    # Pick the assistant you want (you only have 1 right now)
    assistant = assistants[0]
    assistant_id = assistant.get("assistant_id") or assistant.get("id")
    graph_id = assistant.get("graph_id")

    # We create a thread for tracking the state of our run
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    print(f"Created thread_id: {thread_id}")

    """
    Now, we can run our agent  [with `client.runs.stream`](https://docs.langchain.com/oss/python/langgraph/graph-api/#stream-and-astream) with:
    * The `thread_id`
    * The `graph_id`
    * The `input`
    * The `stream_mode`

    We'll discuss streaming in depth in a future module.
    For now, just recognize that we are [streaming](https://docs.langchain.com/langsmith/streaming) the full value of the state after each step of the graph with `stream_mode="values"`.
    The state is captured in the `chunk.data`.
    """

    from langchain_core.messages import HumanMessage

    # Input (avoid naming this variable `input`)
    input_payload = {"graph_state": "Hi, I am Matthew."}
    # Stream the run (graph_id must match what your local server exposes)
    async for chunk in client.runs.stream(
        thread_id,
        graph_id,
        input=input_payload,
        stream_mode="values",
    ):
        if not chunk.data or chunk.event == "metadata":
            continue

        if isinstance(chunk.data, dict) and "error" in chunk.data:
            raise RuntimeError(
                f"{chunk.data.get('error')}: {chunk.data.get('message')}"
            )

        _print_stream_chunk(chunk)  # <-- add this

        print("Run complete.")


"""
## Testing with Cloud

We can deploy to Cloud via LangSmith, as outlined [here](https://docs.langchain.com/langsmith/deployment-quickstart#deploy-from-github-with-langgraph-cloud).

### Create a New Repository on GitHub

* Go to your GitHub account
* Click on the "+" icon in the upper-right corner and select `"New repository"`
* Name your repository (e.g., `langchain-academy`)

### Add Your GitHub Repository as a Remote

* Go back to your terminal where you cloned `langchain-academy` at the start of this course
* Add your newly created GitHub repository as a remote

```
git remote add origin https://github.com/your-username/your-repo-name.git
```
* Push to it
```
git push -u origin main
```

### Connect LangSmith to your GitHub Repository

* Go to [LangSmith](hhttps://smith.langchain.com/)
* Click on `deployments` tab on the left LangSmith panel
* Add `+ New Deployment`
* Then, select the Github repository (e.g., `langchain-academy`) that you just created for the course
* Point the `LangGraph API config file` at one of the `studio` directories
* For example, for module-1 use: `module-1/studio/langgraph.json`
* Set your API keys (e.g., you can just copy from your `module-1/studio/.env` file)

![Screenshot 2024-09-03 at 11.35.12 AM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbad4fd61c93d48e5d0f47_deployment2.png)

### Work with your deployment

We can then interact with our deployment a few different ways:

* With the SDK, as before.
* With [LangGraph Studio](https://docs.langchain.com/langsmith/deployment-quickstart#3-test-your-application-in-studio).

![Screenshot 2024-08-23 at 10.59.36 AM.png](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66dbad4fa159a09a51d601de_deployment3.png)

To use the SDK here in the notebook, simply ensure that `LANGSMITH_API_KEY` is set!
"""


async def run_cloud() -> None:
    """Run against a deployed LangSmith Deployment (formerly LangGraph Cloud)."""
    _set_env("LANGSMITH_API_KEY")

    # Replace this with the URL of your deployed graph
    URL = "https://langchain-academy-8011c561878d50b1883f7ed11b32d720.default.us.langgraph.app"
    client = get_client(url=URL)

    assistants = await client.assistants.search()
    print(f"Found {len(assistants)} assistants on cloud deployment.")

    if not assistants:
        raise RuntimeError("No assistants found on the cloud deployment.")

    for idx, a in enumerate(assistants):
        assistant_id = a.get("assistant_id") or a.get("id")
        name = a.get("name")
        graph_id = a.get("graph_id")
        updated_at = a.get("updated_at") or a.get("modified_at")
        created_at = a.get("created_at")

        print(f"--- Assistant #{idx} ---")
        print(f"id:        {assistant_id}")
        print(f"name:      {name}")
        print(f"graph_id:   {graph_id}")
        print(f"created_at: {created_at}")
        print(f"updated_at: {updated_at}")

        print("raw:")
        print(json.dumps(a, indent=2, default=str))
        print()

    assistant = assistants[0]
    graph_id = assistant.get("graph_id")

    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    print(f"Created thread_id: {thread_id}")

    input_payload = {"graph_state": "Hi, I am Matthew."}

    async for chunk in client.runs.stream(
        thread_id,
        graph_id,
        input=input_payload,
        stream_mode="values",
    ):
        if chunk.data and chunk.event != "metadata":
            _print_stream_chunk(chunk)


async def main() -> None:
    # Toggle these as needed
    run_local_server = True
    run_cloud_deployment = False

    if run_local_server:
        await run_local()

    if run_cloud_deployment:
        await run_cloud()


if __name__ == "__main__":
    asyncio.run(main())
