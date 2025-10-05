import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from activities_crewai.app.agent import ActivitiesScraperAgent
from activities_crewai.app.agent_executor import ActivitiesExecutor

def main():
    card=AgentCard(
        name="ActivitiesScraper",
        description="Scrapes activities/tours and returns strict JSON.",
        url="http://localhost:12023/",
        version="1.0.0",
        default_input_modes=ActivitiesScraperAgent.SUPPORTED_CONTENT_TYPES,
        default_output_modes=ActivitiesScraperAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=AgentCapabilities(streaming=False),
        skills=[AgentSkill(id="activities", name="Activities Scraper", description="Returns JSON activities", tags=["scrape","activities"])]
    )
    app=A2AStarletteApplication(agent_card=card, http_handler=DefaultRequestHandler(ActivitiesExecutor(), InMemoryTaskStore())).build()
    uvicorn.run(app, host="localhost", port=12023)

if __name__=="__main__":
    main()
