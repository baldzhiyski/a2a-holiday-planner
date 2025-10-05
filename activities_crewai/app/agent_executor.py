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
from activities_crewai.app.agent import ActivitiesScraperAgent


class ActivitiesExecutor(AgentExecutor):
    """AgentExecutor for the ActivitiesScraperAgent."""

    def __init__(self):
        self.agent = ActivitiesScraperAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Executes the activities scraping agent."""
        if not context.task_id or not context.context_id:
            raise ValueError("RequestContext must have task_id and context_id")
        if not context.message:
            raise ValueError("RequestContext must have a message")

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task:
            await updater.submit()
        await updater.start_work()

        try:
            # The input from the HostAgent comes as a plain JSON string
            user_query = context.get_user_input()

            # Run the structured agent
            result = self.agent.invoke_structured(user_query)

            # Ensure the output is JSON-serializable (stringify if necessary)
            if isinstance(result, (dict, list)):
                result_str = json.dumps(result)
            elif hasattr(result, "model_dump_json"):
                result_str = result.model_dump_json()
            else:
                result_str = str(result)

            # Add artifact and complete
            await updater.add_artifact([Part(root=TextPart(text=result_str))], name="activities_json")
            await updater.complete()

        except Exception as e:
            print(f"[activities] Error invoking agent: {e}")
            raise ServerError(error=InternalError()) from e

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
