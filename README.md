# Running LangGraph Studio Locally with Poetry

### 1) Install the LangGraph CLI (this is the part that adds the `langgraph` command)

The CLI is a separate package: **`langgraph-cli`** (not just `langgraph`). The official install is: `pip install langgraph-cli`. ([LangChain Docs][1])

Since you’re using Poetry, do **one** of these:

**Option A (recommended with Poetry):**

```powershell
poetry add --group dev "langgraph-cli[inmem]"
poetry install
```

### 2) Verify the command exists

```powershell
poetry run langgraph --help
poetry run langgraph --version
```

The docs explicitly list `langgraph --help` as the verification step. ([LangChain Docs][1])

### 3) Run the local Studio server

Go to the folder that contains your `langgraph.json` (you already are in `...\src\langgraph_projects\level-1`), then run:

```powershell
poetry run langgraph dev
```

`langgraph dev` starts a lightweight local dev server (no Docker required).

### 4) Open Studio in your browser

When it starts, you’ll see URLs like:

- API: `http://127.0.0.1:2024`
- Studio UI: a `smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024` link

The “Studio UI” link is what you open to use LangSmith Studio with your local server.

---

# Running LangSmith

---

    ## LangSmith

    We can look at traces in LangSmith.
    https://smith.langchain.com/

---

[1]: https://python.langchain.com/en/latest/ecosystem/langgraph/cli.html
