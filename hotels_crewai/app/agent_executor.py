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
from hotels_crewai.app.agent import HotelsScraperAgent


class HotelsExecutor(AgentExecutor):
    def __init__(self):
        self.agent = HotelsScraperAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Guards (same as your working sample)
        if not context.task_id or not context.context_id:
            raise ValueError("RequestContext must have task_id and context_id")
        if not context.message:
            raise ValueError("RequestContext must have a message")

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            await updater.submit()
        await updater.start_work()

        # Optional gate mirroring the sample (never rejects)
        if self._validate_request(context):
            raise ServerError(error=InvalidParamsError())

        try:
            user_query = context.get_user_input()  # JSON string from host
            result = self.agent.invoke_structured(user_query)

            # Ensure we add a JSON string artifact
            if isinstance(result, (dict, list)):
                result_str = json.dumps(result)
            elif hasattr(result, "model_dump_json"):
                result_str = result.model_dump_json()
            else:
                result_str = str(result)

            parts = [Part(root=TextPart(text=result_str))]
            await updater.add_artifact(parts)  # avoid name= for widest compatibility
            await updater.complete()

        except Exception as e:
            print(f"[hotels] Error invoking agent: {e}")
            raise ServerError(error=InternalError()) from e

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())

    def _validate_request(self, context: RequestContext) -> bool:
        # Mirror working example: never reject
        return False
