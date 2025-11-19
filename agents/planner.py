#"""Planner agent responsible for outlining the project roadmap."""

# Agent 1
from __future__ import annotations

from typing import Any, Iterable, Optional

from crewai import Agent

from config.settings import build_crewai_llm

SYSTEM_PROMPT = (
    """Add system prompt content here."""
)


def create_planner_agent(
    tools: Optional[Iterable[object]] = None,
    llm_overrides: dict[str, Any] | None = None,
) -> Agent:
    """Create the planner agent used to bootstrap the workflow."""
    return Agent(
        name="---",
        role="",      #"Architect the workshop roadmap and align deliverables",
        goal="",#"Produce a milestone-driven execution plan covering research, authoring, and review",
        backstory=(
            "If any add a backstory here."
            #"Placeholder: Replace with scenario-specific planning context during the workshop. "
            #"You excel at breaking down ambiguous goals into concrete, evidence-backed steps."
        ),
        llm=build_crewai_llm(**(llm_overrides or {})),
        allow_delegation=False,
        verbose=True,
        system_prompt=SYSTEM_PROMPT,
        tools=list(tools or []), ## Call tools here
    )
