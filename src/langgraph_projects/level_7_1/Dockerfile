FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir poetry

# Copy dependency metadata first (best caching)
COPY pyproject.toml poetry.lock* /app/
COPY README.md /app/README.md

# Install deps only (no project code needed yet)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --without dev --no-root

# Now copy the actual source code
COPY src /app/src

# Install ONLY your project package now that src exists
RUN poetry install --no-interaction --no-ansi --only-root

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "langgraph_projects.level_7_1.api:app", "--host", "0.0.0.0", "--port", "8000"]
