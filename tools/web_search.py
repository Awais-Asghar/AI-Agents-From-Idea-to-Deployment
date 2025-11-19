"""DuckDuckGo-powered open web search tool."""
from __future__ import annotations

import logging
from typing import Any

from crewai.tools import BaseTool
from duckduckgo_search import DDGS
from pydantic import Field


class DuckDuckGoSearchTool(BaseTool):
    """DuckDuckGo search tool that logs queries before returning results."""

    name: str = "duckduckgo_search"
    description: str = (
        "Query DuckDuckGo for current information. Provide a short search phrase."
    )
    max_results: int = Field(default=5, ge=1, description="Number of hits to return")
    backend: str = Field(
        default="text",
        description="DuckDuckGo backend to use (text, news, images).",
    )

    _logger = logging.getLogger(__name__)

    def _run(self, query: str) -> str:
        self._logger.info("DuckDuckGo search for query: %s", query)
        results = self._search(query)
        if not results:
            return "No DuckDuckGo results found for that query."

        formatted = []
        for index, item in enumerate(results, start=1):
            title = item.get("title") or item.get("heading") or "Untitled result"
            url = item.get("href") or item.get("url") or ""
            summary = (
                item.get("body")
                or item.get("snippet")
                or item.get("description")
                or "No summary provided."
            )
            formatted.append(
                f"Result {index}: {title}\nURL: {url}\nSummary: {summary.strip()}"
            )

        serialized = "\n\n".join(formatted)
        self._logger.debug("DuckDuckGo raw results: %s", results)
        return serialized

    def _search(self, query: str) -> list[dict[str, Any]]:
        try:
            with DDGS() as ddgs:
                if self.backend == "news":
                    iterator = ddgs.news(query, max_results=self.max_results)
                elif self.backend == "images":
                    iterator = ddgs.images(query, max_results=self.max_results)
                else:
                    iterator = ddgs.text(query, max_results=self.max_results)
                return list(iterator)
        except Exception as exc:  # pragma: no cover - network variability
            self._logger.exception("DuckDuckGo search failed for '%s'", query)
            raise ValueError(f"DuckDuckGo search failed: {exc}") from exc


def create_web_search_tool() -> DuckDuckGoSearchTool:
    """Create a tool that performs top-k DuckDuckGo searches."""
    return DuckDuckGoSearchTool(max_results=5)
