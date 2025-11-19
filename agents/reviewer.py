#"""Reviewer agent that checks quality, accuracy, and completeness."""

# Agent 4
from __future__ import annotations

from typing import Any, Iterable, Optional

from crewai import Agent

from config.settings import build_crewai_llm

SYSTEM_PROMPT = (
    """Add system prompt content here."""
    # "You are the Quality Reviewer for the workshop. "
    # "Audit drafts for factual accuracy, pedagogy, deployment readiness, and alignment with the plan. "
    # "Provide actionable feedback and highlight risks or missing pieces."
)


def create_reviewer_agent(
    tools: Optional[Iterable[object]] = None,
    llm_overrides: dict[str, Any] | None = None,
) -> Agent:
    """Create the reviewer agent that validates deliverables before release."""
    return Agent(
        name="---",  # e.g., "Quality Reviewer"
        role="",  # "Ensure every deliverable is accurate, actionable, and polished"
        goal="",  # "Deliver constructive critiques and sign-off criteria before publication"
        backstory=(
            "If any add a backstory here."
            # "Placeholder: Replace with scenario-specific review standards during the workshop. "
            # "You safeguard against gaps, errors, and unclear guidance."
        ),
        llm=build_crewai_llm(**(llm_overrides or {})),
        allow_delegation=False,
        verbose=True,
        system_prompt=SYSTEM_PROMPT,
        tools=list(tools or []),  # Call tools here
    )
