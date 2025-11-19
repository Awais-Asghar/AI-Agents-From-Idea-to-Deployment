#"""Task definitions for the Agentic AI Workshop crew."""
from __future__ import annotations

from typing import List

from crewai import Task

from tools import create_calculator_tool, create_rag_tool, create_web_search_tool


def create_planning_task(agent) -> Task:
    """Task 1 placeholder: draft an execution plan."""
    return Task(
        description=(
            "Add description for Task 1."
            # "Analyze the workshop topic '{topic}' and craft a milestone-based execution plan. "
            # "List required assets, responsible roles, tooling, and a realistic timeline."
        ),
        expected_output=(
            "Add expected output for Task 1."
            # "A structured plan including objectives, three to five milestones, resource requirements, "
            # "risk mitigation ideas, and success metrics."
        ),
        agent=agent,
        name="Task 1",  # "Planning"
    )


def create_research_task(agent, tools=None) -> Task:
    """Task 2 placeholder: gather supporting research."""
    tools = list(tools) if tools is not None else [
        create_rag_tool(),
        create_web_search_tool(),
        create_calculator_tool(),
    ]
    return Task(
        description=(
            "Add description for Task 2."
            # "Use the local knowledge base and live web results to validate the plan for '{topic}'. "
            # "Cite at least three trustworthy sources and capture data points that justify each milestone."
        ),
        expected_output=(
            "Add expected output for Task 2."
            # "A bullet list of insights with inline citations, key statistics, and references to the RAG documents."
        ),
        agent=agent,
        tools=tools,
        name="Task 2",  # "Research"
    )


def create_writing_task(agent) -> Task:
    """Task 3 placeholder: author deliverables."""
    return Task(
        description=(
            "Add description for Task 3."
            # "Draft the workshop narrative for '{topic}', including an overview, prerequisites, step-by-step labs, and deployment notes. "
            # "Incorporate the research insights and calculator results where helpful."
        ),
        expected_output=(
            "Add expected output for Task 3."
            # "A Markdown-formatted workshop guide with sections for Goals, Agenda, Hands-on Labs, Deployment, and Resources."
        ),
        agent=agent,
        name="Task 3",  # "Writing"
    )


def create_review_task(agent) -> Task:
    """Task 4 placeholder: review compiled deliverables."""
    return Task(
        description=(
            "Add description for Task 4."
            # "Review the draft content for '{topic}' for accuracy, completeness, and pedagogy. Provide an executive summary of strengths, "
            # "list gaps or issues, and suggest concrete improvements."
        ),
        expected_output=(
            "Add expected output for Task 4."
            # "A review report with sections for Summary, Major Findings, Minor Suggestions, and Final Recommendation."
        ),
        agent=agent,
        name="Task 4",  # "Reviewing"
    )


def build_workshop_tasks(planner, researcher, writer, reviewer, research_tools=None) -> List[Task]:
    """Convenience helper to create the full task list order."""
    return [
        create_planning_task(planner),
        create_research_task(researcher, tools=research_tools),
        create_writing_task(writer),
        create_review_task(reviewer),
    ]
