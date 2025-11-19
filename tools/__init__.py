"""Tool factories used across the workshop agents and tasks."""
from __future__ import annotations

from pathlib import Path
from typing import List

from crewai.tools import BaseTool

from .calculator import CalculatorTool
from .rag_tool import LocalRAGTool
from .web_search import create_web_search_tool

__all__ = [
    "create_rag_tool",
    "create_web_search_tool",
    "create_calculator_tool",
    "get_default_toolkit",
]


DEFAULT_VECTORSTORE_DIR = Path(__file__).resolve().parents[1] / "rag" / "vectorstore"


def create_rag_tool(vectorstore_path: Path | None = None, *, top_k: int = 4) -> LocalRAGTool:
    """Instantiate the local RAG retrieval tool."""
    target_path = vectorstore_path or DEFAULT_VECTORSTORE_DIR
    return LocalRAGTool(vectorstore_path=target_path, top_k=top_k)


def create_calculator_tool() -> CalculatorTool:
    """Instantiate the deterministic calculator tool."""
    return CalculatorTool()


def get_default_toolkit() -> List[BaseTool]:
    """Provide the standard set of tools shared by research-heavy agents."""
    return [
        create_rag_tool(),
        create_web_search_tool(),
        create_calculator_tool(),
    ]
