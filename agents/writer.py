#"""Writer agent that composes workshop materials from vetted research."""

# Agent 3
from __future__ import annotations

from typing import Any, Iterable, Optional

from crewai import Agent

from config.settings import build_crewai_llm

SYSTEM_PROMPT = (
    """Add system prompt content here."""
    # "You are the Lead Content Writer for the workshop. "
    # "Transform research findings and plans into compelling narratives, lesson outlines, and code walkthroughs. "
    # "Maintain clarity, instructor-friendly tone, and actionable takeaways."
)


def create_writer_agent(
    tools: Optional[Iterable[object]] = None,
    llm_overrides: dict[str, Any] | None = None,
) -> Agent:
    """Create the writer agent responsible for draft generation."""
    return Agent(
        name="---",  # e.g., "Lead Content Writer"
        role="",  # "Author workshop scripts, lab guides, and deployment notes"
        goal="",  # "Produce polished, instructor-ready materials grounded in researched evidence"
        backstory=(
            "If any add a backstory here."
            # "Placeholder: Replace with scenario-specific writing guidance during the workshop. "
            # "You specialize in translating complex AI workflows into accessible, hands-on content."
        ),
        llm=build_crewai_llm(**(llm_overrides or {})),
        allow_delegation=False,
        verbose=True,
        system_prompt=SYSTEM_PROMPT,
        tools=list(tools or []),  # Call tools here
    )
