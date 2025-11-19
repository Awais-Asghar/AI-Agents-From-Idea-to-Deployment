#"""Research agent that combines RAG retrieval with live web search."""

# Agent 2
from __future__ import annotations

from typing import Any, Iterable, Optional

from crewai import Agent

from config.settings import build_crewai_llm

SYSTEM_PROMPT = (
    """Add system prompt content here."""
    # "You are the Research Specialist for the workshop. "
    # "Synthesize information from the local RAG knowledge base and the live web. "
    # "Validate claims, cite sources, and prepare concise bullet summaries for downstream teams."
)


def create_researcher_agent(
    tools: Optional[Iterable[object]] = None,
    llm_overrides: dict[str, Any] | None = None,
) -> Agent:
    """Create the researcher agent that fuses structured and unstructured sources."""
    return Agent(
        name="---",  # e.g., "Insight Researcher"
        role="",  # "Curate authoritative context for workshop deliverables"
        goal="",  # "Blend RAG insights with verified web findings to back every recommendation"
        backstory=(
            "If any add a backstory here."
            # "Placeholder: Replace with scenario-specific research focus during the workshop. "
            # "You are rigorous about citations, fact-checking, and keeping insights actionable."
        ),
        llm=build_crewai_llm(**(llm_overrides or {})),
        allow_delegation=False,
        verbose=True,
        system_prompt=SYSTEM_PROMPT,
        tools=list(tools or []),  # Call tools here
    )
