from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    tool_name: str
    status: str
    output: Any = None
    error: str | None = None


@dataclass(frozen=True)
class ModelTurn:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class FunctionCallBudget:
    max_tool_calls: int = 8
    max_rounds: int = 4

    def __post_init__(self) -> None:
        if self.max_tool_calls < 1 or self.max_rounds < 1:
            raise ValueError("Function-call budgets must be positive")


@dataclass(frozen=True)
class FunctionCallTrace:
    response: str
    tool_results: tuple[ToolResult, ...]
    rounds: int


class ToolCallingModel(Protocol):
    def __call__(
        self,
        prompt: str,
        tools: tuple[ToolSpec, ...],
        prior_results: tuple[ToolResult, ...],
    ) -> ModelTurn: ...


class ToolBudgetExceeded(RuntimeError):
    pass


class FunctionCallingEngine:
    """Provider-independent model/tool loop with explicit call and round budgets."""

    def __init__(
        self,
        model: ToolCallingModel,
        executor: Callable[[ToolCall], ToolResult],
        tools: tuple[ToolSpec, ...],
        budget: FunctionCallBudget | None = None,
    ) -> None:
        self.model = model
        self.executor = executor
        self.tools = tools
        self.budget = budget or FunctionCallBudget()

    def run(self, prompt: str) -> FunctionCallTrace:
        results: list[ToolResult] = []
        seen_call_ids: set[str] = set()
        for round_number in range(1, self.budget.max_rounds + 1):
            turn = self.model(prompt, self.tools, tuple(results))
            if not turn.tool_calls:
                return FunctionCallTrace(
                    response=turn.content,
                    tool_results=tuple(results),
                    rounds=round_number,
                )
            if len(results) + len(turn.tool_calls) > self.budget.max_tool_calls:
                raise ToolBudgetExceeded("Model exceeded the maximum tool-call budget")
            for call in turn.tool_calls:
                if call.id in seen_call_ids:
                    raise ValueError("Model emitted a duplicate tool-call id")
                seen_call_ids.add(call.id)
                results.append(self.executor(call))
        raise ToolBudgetExceeded("Model exceeded the maximum function-call rounds")
