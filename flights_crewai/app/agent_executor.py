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
from flights_crewai.app.agent import FlightsScraperAgent


class FlightsExecutor(AgentExecutor):
    def __init__(self):
        self.agent = FlightsScraperAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # ---- Same guards as the working example
        if not context.task_id or not context.context_id:
            raise ValueError("RequestContext must have task_id and context_id")
        if not context.message:
            raise ValueError("RequestContext must have a message")

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            await updater.submit()
        await updater.start_work()

        # Optional validation gate (keep identical behavior to working example)
        if self._validate_request(context):
            # NOTE: working example *raises when True* (effectively never raises because it returns False)
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()  # raw text (your JSON string)
        try:
            result_json = self.agent.invoke_structured(query)  # returns a JSON string
            print(f"[flights] Final Result ===> {result_json}")
        except Exception as e:
            print(f"[flights] Error invoking agent: {e}")
            # surface an A2A error so the host gets a proper failure
            raise ServerError(error=InternalError()) from e

        parts = [Part(root=TextPart(text=result_json))]

        # IMPORTANT: no extra kwargs; some versions don't accept name=
        await updater.add_artifact(parts)
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())

    def _validate_request(self, context: RequestContext) -> bool:
        # Mirror the working example: always returns False (so it never rejects)
        return False
