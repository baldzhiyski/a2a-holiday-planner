import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from flights_crewai.app.agent import FlightsScraperAgent, FlightQuery, FlightList
from flights_crewai.app.agent_executor import FlightsExecutor

def main():

    card=AgentCard(
        name="FlightsScraper",
        description="Searches & scrapes flight options; returns strict JSON.",
        url="http://localhost:12021/",
        version="1.0.0",
        default_input_modes=FlightsScraperAgent.SUPPORTED_CONTENT_TYPES,
        default_output_modes=FlightsScraperAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=AgentCapabilities(streaming=False),
        skills=[AgentSkill(id="flights", name="Flight Search Scraper", description="Returns JSON flights", tags=["scrape","flights"])]
    )
    app=A2AStarletteApplication(agent_card=card, http_handler=DefaultRequestHandler(FlightsExecutor(), InMemoryTaskStore())).build()
    uvicorn.run(app, host="localhost", port=12021)

if __name__=="__main__":
    main()
