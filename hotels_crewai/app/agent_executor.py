from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TextPart, UnsupportedOperationError
from hotels_crewai.app.agent import HotelsScraperAgent

class HotelsExecutor(AgentExecutor):
    def __init__(self): self.agent=HotelsScraperAgent()
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.task_id or not context.context_id:
            raise ValueError("RequestContext must have task_id and context_id")
        if not context.message:
            raise ValueError("RequestContext must have a message")
        updater=TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task: await updater.submit()
        await updater.start_work()
        res=self.agent.invoke_structured(context.get_user_input())
        await updater.add_artifact([Part(root=TextPart(text=res))], name="hotels_json")
        await updater.complete()
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError()
