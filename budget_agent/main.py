import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from budget_agent.app.agent import BudgetAgent
from budget_agent.app.agent_executor import BudgetExecutor

def main():
    card=AgentCard(
        name="BudgetPolicy",
        description="Allocates per-category budget caps as structured JSON.",
        url="http://localhost:12024/",
        version="1.0.0",
        default_input_modes=BudgetAgent.SUPPORTED_CONTENT_TYPES,
        default_output_modes=BudgetAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=AgentCapabilities(streaming=False),
        skills=[AgentSkill(id="budget_caps", name="Budget Caps", description="Returns JSON caps", tags=["budget","policy"])],
        preferred_transport="HTTP+JSON"

    )
    app=A2AStarletteApplication(agent_card=card, http_handler=DefaultRequestHandler(agent_executor=BudgetExecutor(), task_store=InMemoryTaskStore())).build()
    uvicorn.run(app, host="localhost", port=12024)

if __name__=="__main__":
    main()
