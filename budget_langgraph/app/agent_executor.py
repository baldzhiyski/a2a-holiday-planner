import json

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
from budget_langgraph.app.agent import BudgetAgent


class BudgetExecutor(AgentExecutor):
    def __init__(self):
        self.agent = BudgetAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # 1) Same guards as the working sample
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

        user_text = context.get_user_input()  # host sends a JSON string here
        try:
            # 3) Run your langgraph
            res = self.agent.graph.invoke(
                {"messages": [("user", user_text)]},
                config={"configurable": {"thread_id": str(context.context_id)}},
            )

            # 4) Extract structured JSON safely
            #    Expect either a pydantic model at structured_response or a dict
            sr = res.get("structured_response")
            if sr is None:
                # fallback: maybe the graph returned the dict directly
                out_json = res if isinstance(res, dict) else {"caps": {}}
            else:
                # pydantic model: dump to json
                if hasattr(sr, "model_dump_json"):
                    out_json_str = sr.model_dump_json()
                elif hasattr(sr, "model_dump"):
                    out_json_str = json.dumps(sr.model_dump())
                else:
                    out_json_str = json.dumps(sr)

                parts = [Part(root=TextPart(text=out_json_str))]
                await updater.add_artifact(parts)
                await updater.complete()
                return

            # fallback path (dict/string)
            out_json_str = out_json if isinstance(out_json, str) else json.dumps(out_json)
            parts = [Part(root=TextPart(text=out_json_str))]
            await updater.add_artifact(parts)
            await updater.complete()
            return

        except Exception as e:
            # 5) Log and propagate as A2A error so the host sees a proper failure
            print(f"[budget] Error invoking agent: {e}")
            raise ServerError(error=InternalError()) from e

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())

    def _validate_request(self, context: RequestContext) -> bool:
        # Mirror the working example: never reject here
        return False
