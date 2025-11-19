"""Deterministic calculator tool for quick quantitative reasoning."""
from __future__ import annotations

import ast
import logging
import operator
from typing import Any, Dict

from crewai.tools import BaseTool

_ALLOWED_OPERATORS: Dict[type[ast.AST], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


class CalculatorTool(BaseTool):
    name: str = "deterministic_calculator"
    description: str = (
        "Perform precise arithmetic on simple expressions. "
        "Supports addition, subtraction, multiplication, division, modulus, and powers."
    )

    _logger = logging.getLogger(__name__)

    def _run(self, query: str) -> str:
        try:
            expression = ast.parse(query, mode="eval").body
            result = self._eval(expression)
            self._logger.info("Calculator evaluated '%s' -> %s", query, result)
            return str(result)
        except Exception as exc:  # pragma: no cover - defensive layer
            self._logger.exception("Calculator failed for expression '%s'", query)
            raise ValueError(f"Failed to evaluate expression '{query}': {exc}") from exc

    def _eval(self, node: ast.AST) -> float:
        if isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return float(node.n)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](self._eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
            left = self._eval(node.left)
            right = self._eval(node.right)
            return _ALLOWED_OPERATORS[type(node.op)](left, right)
        raise ValueError(f"Unsupported expression: {ast.dump(node, include_attributes=False)}")
