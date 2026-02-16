# src/langgraph_projects/my_utils/configuration.py
# This file defines a Configuration dataclass that can be constructed from a RunnableConfig.
# It is used to extract user_id and thread_id from the RunnableConfig passed to the graph nodes.
# The Configuration class has a class method from_runnable_config that takes a RunnableConfig or a dict and returns a Configuration instance.
# how to use:
# from langgraph_projects.my_utils.configuration import Configuration
# config = Configuration.from_runnable_config(runnable_config)
# The Configuration class is frozen, meaning its instances are immutable after creation.

# src/langgraph_projects/my_utils/configuration.py

from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.runnables.config import RunnableConfig


@dataclass(frozen=True)
class Configuration:
    """Runtime configuration passed via RunnableConfig['configurable']."""

    user_id: str
    thread_id: Optional[str] = None

    @classmethod
    def from_runnable_config(
        cls, config: RunnableConfig | dict[str, Any]
    ) -> "Configuration":
        # RunnableConfig is a TypedDict; Studio may pass a plain dict too.
        cfg = config.get("configurable", {})  # works for both RunnableConfig and dict
        return cls(
            user_id=str(cfg.get("user_id", "")),
            thread_id=(str(cfg["thread_id"]) if "thread_id" in cfg else None),
        )


# ---- Compatibility export ----
# Allows: from langgraph_projects.my_utils.configuration import configuration
# so callers can do: configuration.Configuration.from_runnable_config(...)
import sys as _sys

configuration = _sys.modules[__name__]

__all__ = ["Configuration", "configuration"]
