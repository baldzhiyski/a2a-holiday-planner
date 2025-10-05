from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TextPart, UnsupportedOperationError
from budget_langgraph.app.agent import BudgetAgent

class BudgetExecutor(AgentExecutor):
    def __init__(self): self.agent=BudgetAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater=TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task: await updater.submit()
        await updater.start_work()
        res = self.agent.graph.invoke(
            {"messages": [("user", context.get_user_input())]},
            config={"configurable": {"thread_id": str(context.context_id)}}
        )

        # Return the structured JSON from response
        out = res["structured_response"].model_dump_json()
        await updater.add_artifact([Part(root=TextPart(text=out))], name="budget_caps_json")
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError()
