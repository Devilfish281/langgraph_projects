# my_utils/load_env.py


import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger(__name__)  # Reuse the global logger


def load_dotenv_only() -> None:
    """Load .env into os.environ (no validation, no logging dependency)."""  # Added Code:
    load_dotenv()  # Added Code:


def validate_environment(log: logging.Logger | None = None) -> None:
    """Validate required env vars (logs only if a logger is provided)."""  # Added Code:
    _log = log or logger  # Added Code:

    apikey = os.getenv("OPENAI_API_KEY", "").strip()
    if not apikey:
        _log.error("OPENAI_API_KEY environment variable not set")
        raise ValueError("OPENAI_API_KEY environment variable not set")

    if len(apikey) < 20:
        _log.error("OPENAI_API_KEY looks too short to be valid")
        raise ValueError("OPENAI_API_KEY looks too short to be valid")

    _log.debug("API Key loaded successfully.")
    _log.debug("All required environment variables are set.")

    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("TAVILY_API_KEY is not set.")
    if not os.getenv("LANGCHAIN_API_KEY"):
        raise RuntimeError("LANGCHAIN_API_KEY is not set.")

    if os.getenv("USE_ANTHROPIC", "0") == "1" and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
