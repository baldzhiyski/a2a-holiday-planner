import json
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Part,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils.errors import ServerError
from budget_agent.app.agent import BudgetAgent


class BudgetExecutor(AgentExecutor):
    def __init__(self):
        # Uses your Option-1 BudgetAgent (no LangGraph, structured output only)
        self.agent = BudgetAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # 1) Guards
        if not context.task_id or not context.context_id:
            raise ValueError("RequestContext must have task_id and context_id")
        if not context.message:
            raise ValueError("RequestContext must have a message")

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            await updater.submit()
        await updater.start_work()

        # 2) (Optional) mirror the “always False” validate gate in the sample
        if self._validate_request(context):
            raise ServerError(error=InvalidParamsError())

        # 3) Get user input and normalize to a single prompt string
        raw_input: Any = context.get_user_input()
        if isinstance(raw_input, dict):
            # Accept JSON payloads like {"total_budget_eur": 2500, "passengers": 3}
            tb = raw_input.get("total_budget_eur")
            pax = raw_input.get("passengers")
            if tb is not None and pax is not None:
                user_text = f"total_budget_eur={tb}, passengers={pax}"
            else:
                # Fallback: pass the dict as text for the LLM to extract
                user_text = json.dumps(raw_input)
        else:
            # Assume stringy/natural language like "We have 2500 EUR for 3 passengers"
            user_text = str(raw_input)

        try:
            # 4) Run the structured LLM call (no recursion risk)
            result = self.agent.invoke(user_text)  # returns BudgetOutput (Pydantic)

            # 5) Emit as artifact (stringified JSON)
            out_json_str = result.model_dump_json()
            parts = [Part(root=TextPart(text=out_json_str))]
            await updater.add_artifact(parts)
            await updater.complete()
            return

        except Exception as e:
            print(f"[budget] Error invoking BudgetAgent: {e}")
            raise ServerError(error=InternalError()) from e

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Not supported for this simple executor
        raise ServerError(error=UnsupportedOperationError())

    def _validate_request(self, context: RequestContext) -> bool:
        # Mirror the working example: never reject here
        return False
