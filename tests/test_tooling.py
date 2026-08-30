from __future__ import annotations

import pytest

from apex.tooling import (
    FunctionCallBudget,
    FunctionCallingEngine,
    ModelTurn,
    ToolBudgetExceeded,
    ToolCall,
    ToolResult,
    ToolSpec,
)


def test_provider_independent_function_call_loop() -> None:
    def model(_prompt, _tools, results):
        if not results:
            return ModelTurn(
                content="",
                tool_calls=(ToolCall("call-1", "echo", {"value": "hello"}),),
            )
        return ModelTurn(content=f"Tool said: {results[0].output['value']}")

    def execute(call: ToolCall) -> ToolResult:
        return ToolResult(call.id, call.name, "completed", output=call.arguments)

    trace = FunctionCallingEngine(
        model,
        execute,
        (
            ToolSpec(
                "echo",
                "Echo a value",
                {"type": "object", "properties": {"value": {"type": "string"}}},
            ),
        ),
    ).run("Use a tool")
    assert trace.response == "Tool said: hello"
    assert trace.rounds == 2
    assert len(trace.tool_results) == 1


def test_function_call_budget_is_enforced() -> None:
    def model(_prompt, _tools, _results):
        return ModelTurn(
            content="",
            tool_calls=(
                ToolCall("one", "echo", {}),
                ToolCall("two", "echo", {}),
            ),
        )

    engine = FunctionCallingEngine(
        model,
        lambda call: ToolResult(call.id, call.name, "completed"),
        (),
        FunctionCallBudget(max_tool_calls=1, max_rounds=1),
    )
    with pytest.raises(ToolBudgetExceeded):
        engine.run("exceed")
