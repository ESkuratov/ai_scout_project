"""Define the configurable parameters for the agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated

from prompts import SYSTEM_PROMPT


@dataclass(kw_only=True)
class Context:
    """The context for the agent."""

    system_prompt: str = field(
        default=SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-oss-120b",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    base_url: Annotated[str | None, {"__template_metadata__": {"kind": "llm_base_url"}}] = field(
        default=None,
        metadata={
            "description": "Base URL used when issuing requests to the language model provider.",
            "env": "BASE_URL",
        },
    )

    api_key: Annotated[str | None, {"__template_metadata__": {"kind": "secret"}}] = field(
        default=None,
        metadata={
            "description": "API key for authenticating requests to the configured language model provider.",
            "env": "OPENROUTER_KEY",
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )

    def __post_init__(self) -> None:
        if self.api_key:
            os.environ.setdefault("OPENAI_API_KEY", self.api_key)
            print(f"DEBUG: Set OPENAI_API_KEY: {self.api_key[:10]}...")  # Only print first 10 chars for security
        if self.base_url:
            os.environ.setdefault("OPENAI_BASE_URL", self.base_url)
            print(f"DEBUG: Set OPENAI_BASE_URL: {self.base_url}")

