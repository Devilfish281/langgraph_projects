Here’s the “reset-proof” checklist to get your **Browser → FastAPI → Postgres** setup running again on a fresh Windows install.

---

## 1) Install the basics

### A) Install Docker Desktop (Windows 11)

1. Download + install Docker Desktop for Windows.
2. Open Docker Desktop → Settings → enable **“Use the WSL 2 based engine”** (if prompted).
3. Restart Docker Desktop.

Docker’s official install + WSL2 setup steps are here. ([Docker Documentation][1])

### B) Install Python 3.12+

- Install Python and make sure you check **“Add Python to PATH”**.

### C) Install Git

- Install Git for Windows (so you can clone your repo).

---

## 2) Get your project back on the computer

1. Clone your repo (or copy your project folder back).
2. Open PowerShell and `cd` into your level folder:

```powershell
cd C:\Users\ME\Documents\Python\2026\Projects\langgraph_projects\src\langgraph_projects\level_7_1
```

---

## 3) Start Postgres (the same way you did before)

### A) Create the shared network (once)

```powershell
docker network create lg-shared
```

### B) Run Postgres container

```powershell
docker run --name lg-postgres `
  -e POSTGRES_PASSWORD=postgres `
  -e POSTGRES_USER=postgres `
  -e POSTGRES_DB=langgraph `
  -p 5432:5432 `
  -d postgres:16
```

### C) Attach Postgres to the shared network

```powershell
docker network connect lg-shared lg-postgres
```

This “same network = containers can resolve each other by name” idea is core to Docker networking. ([Docker Documentation][2])

---

## 4) Run your FastAPI app with Docker Compose

### A) Make sure your **docker-compose.yml** is the “old one”

(Uses `lg-postgres` hostname and `lg-shared` external network.)

### B) Start the app

From the `level_7_1` folder:

```powershell
docker compose up --build -d
docker compose logs -f app
```

You want to see:

- “Application startup complete”
- “Uvicorn running on [http://0.0.0.0:8000”](http://0.0.0.0:8000”)

---

## 5) Verify Postgres tables got created

Your app calls `checkpointer.setup()` on startup, so tables should exist now.

```powershell
docker exec -it lg-postgres psql -U postgres -d langgraph
```

Inside `psql`:

```sql
\dt
```

That’s the standard `psql` command to list tables. ([Python documentation][3])

---

## 6) Test the API with PowerShell (Invoke-RestMethod)

Run:

```powershell
$uri = "http://localhost:8000/chat"

$body = @{
  thread_id = "demo-1"
  message   = "Hi this is Matthew"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri $uri -ContentType "application/json" -Body $body
```

This is the documented pattern for POSTing JSON with `Invoke-RestMethod`. ([Microsoft Learn][4])

---

## 7) Run your tiny browser page (optional)

If you have your `index.html` in the `tiny-web` folder:

```powershell
cd .\tiny-web
python -m http.server 5500
```

Then open:

- `http://localhost:5500/`

Python’s `http.server` serves the current directory. ([Python documentation][3])

---

## Quick “if something breaks” checks

- **Port 5432 already allocated**: you already have a Postgres running on 5432 → stop it:

  ```powershell
  docker stop lg-postgres
  ```

- **Can’t resolve `lg-postgres`**: re-run:

  ```powershell
  docker network connect lg-shared lg-postgres
  docker compose up -d
  ```

If you paste:

1. `docker compose ps`
2. the first 20 lines of `docker compose logs app`
   I’ll tell you exactly what step failed and the exact fix.

[1]: https://docs.docker.com/desktop/setup/install/windows-install/?utm_source=chatgpt.com "Install Docker Desktop on Windows"
[2]: https://docs.docker.com/desktop/features/wsl/?utm_source=chatgpt.com "Docker Desktop WSL 2 backend on Windows"
[3]: https://docs.python.org/3/library/http.html?utm_source=chatgpt.com "HTTP modules — Python 3.14.3 documentation"
[4]: https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.utility/invoke-restmethod?view=powershell-7.5&utm_source=chatgpt.com "Invoke-RestMethod (Microsoft.PowerShell.Utility)"
