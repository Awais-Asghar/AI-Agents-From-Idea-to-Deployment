"""Global configuration for the Agentic AI workshop project."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Any, TYPE_CHECKING

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from crewai.llm import LLM

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from openai import OpenAI

# Ensure environment variables from a local .env file are available during development.
load_dotenv()

MODEL_NAME = "meta-llama/llama-3.3-70b-instruct:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_HEADERS = {
    "HTTP-Referer": "https://github.com/your-org/agentic-workshop",
    "X-Title": "Agentic AI Workshop",
}

LLM_CONFIG: Dict[str, object] = {
    "model": MODEL_NAME,
    "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
    "temperature": 0.2,
    "max_tokens": 800,
}


def _split_env_list(env_var: str) -> list[str]:
    """Return a sanitized list from a comma-separated environment variable."""

    raw_value = os.getenv(env_var, "")
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


@dataclass
class OpenRouterLLMConfig:
    """Helper container to build consistently configured OpenRouter clients."""

    model: str = str(LLM_CONFIG["model"])
    api_key: str = str(LLM_CONFIG["openrouter_api_key"])
    temperature: float = float(LLM_CONFIG["temperature"])
    max_tokens: int = int(LLM_CONFIG["max_tokens"])
    base_url: str = OPENROUTER_BASE_URL
    headers: Dict[str, str] = field(default_factory=lambda: dict(OPENROUTER_DEFAULT_HEADERS))
    fallback_base_urls: list[str] = field(
        default_factory=lambda: _split_env_list("OPENROUTER_FALLBACK_BASE_URLS")
    )
    fallback_models: list[str] = field(
        default_factory=lambda: _split_env_list("OPENROUTER_FALLBACK_MODELS")
    )


def get_openrouter_client() -> "OpenAI":
    """Instantiate an OpenAI-compatible client configured for OpenRouter."""
    from openai import OpenAI

    config = OpenRouterLLMConfig()
    if not config.api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is missing. Set it in your environment or .env file."
        )
    return OpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        default_headers=config.headers,
    )


def build_openrouter_chat_llm(**overrides: Any) -> ChatOpenAI:
    """Return a LangChain ChatOpenAI client configured for OpenRouter usage."""

    config = OpenRouterLLMConfig()
    if not config.api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is missing. Set it in your environment or .env file."
        )

    return ChatOpenAI(
        model=overrides.get("model", config.model),
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=overrides.get("temperature", config.temperature),
        max_tokens=overrides.get("max_tokens", config.max_tokens),
        default_headers=config.headers,
    )


def build_crewai_llm(**overrides: Any) -> LLM:
    """Return a CrewAI LLM instance configured for OpenRouter via LiteLLM."""

    config = OpenRouterLLMConfig()
    if not config.api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is missing. Set it in your environment or .env file."
        )

    raw_model = overrides.get("model", config.model)
    provider_override = overrides.get("provider")

    if provider_override == "openai":
        model_name = raw_model
    else:
        model_name = (
            raw_model
            if str(raw_model).startswith("openrouter/")
            else f"openrouter/{raw_model}"
        )
    temperature = overrides.get("temperature", config.temperature)
    max_tokens = overrides.get("max_tokens", config.max_tokens)
    extra_headers = overrides.get("extra_headers", config.headers)
    base_url = overrides.get("base_url", config.base_url)

    llm_kwargs: Dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_key": config.api_key,
        "base_url": base_url,
        "api_base": base_url,
        "custom_llm_provider": "openrouter",
        "extra_headers": extra_headers,
    }

    if provider_override:
        llm_kwargs["provider"] = provider_override
        if provider_override == "openai":
            llm_kwargs["default_headers"] = overrides.get(
                "default_headers", extra_headers
            )
            llm_kwargs.pop("custom_llm_provider", None)

    # Allow callers to extend with LiteLLM-specific parameters.
    llm_kwargs.update(overrides.get("litellm_params", {}))

    return LLM(**llm_kwargs)
