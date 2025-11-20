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
    """Task: Author workshop deliverables using research insights."""
    return Task(
        name="Author Workshop Materials",
        agent=agent,
        description=(
            "Draft the complete workshop narrative for the given topic, including: "
            "1) Overview and learning objectives, "
            "2) Prerequisites and setup instructions, "
            "3) Step-by-step lab exercises with examples, "
            "4) Deployment notes and practical recommendations. "
            "Incorporate research insights from the researcher agent "
            "and any calculator/tool outputs where relevant."
        ),
        expected_output=(
            "A well-structured Markdown document including the following sections:\n"
            "- Goals & Agenda\n"
            "- Hands-on Labs with code walkthroughs\n"
            "- Deployment Guidelines\n"
            "- Resources & References\n"
            "Content should be clear, actionable, and instructor-friendly."
        ),
    )

<<<<<<< HEAD

# def create_review_task(agent) -> Task:
#     """Task 4 placeholder: review compiled deliverables."""
#     return Task(
#         description=(
#             "Add description for Task 4."
#             # "Review the draft content for '{topic}' for accuracy, completeness, and pedagogy. Provide an executive summary of strengths, "
#             # "list gaps or issues, and suggest concrete improvements."
#         ),
#         expected_output=(
#             "Add expected output for Task 4."
#             # "A review report with sections for Summary, Major Findings, Minor Suggestions, and Final Recommendation."
#         ),
#         agent=agent,
#         name="Task 4",  # "Reviewing"
#     )

# My part of the code(Muhammad Ahmad)

=======
>>>>>>> 0370c27ac95ad3d718b5ba934afd0c0de0da2044
def create_review_task(agent) -> Task:
    """Task 4: Review the research output for accuracy and completeness."""
    return Task(
        description=(
            "Review the research findings produced in the previous task. "
            "Check for factual accuracy, clarity, completeness, and proper reasoning. "
            "Identify missing details, unsupported claims, logical gaps, or unclear explanations. "
            "Provide constructive, actionable feedback to improve the final research output."
        ),
        expected_output=(
            "A structured review report including: "
            "1) Summary of strengths, "
            "2) Identified issues or corrections, "
            "3) Missing or unclear information, "
            "4) Recommendations for improvement, "
            "5) Final judgment on whether the research is ready."
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
