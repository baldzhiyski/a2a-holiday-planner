import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from hotels_crewai.app.agent import HotelsScraperAgent
from hotels_crewai.app.agent_executor import HotelsExecutor

def main():
    card=AgentCard(
        name="HotelsScraper",
        description="Scrapes hotels and returns strict JSON.",
        url="http://localhost:12022/",
        version="1.0.0",
        default_input_modes=HotelsScraperAgent.SUPPORTED_CONTENT_TYPES,
        default_output_modes=HotelsScraperAgent.SUPPORTED_CONTENT_TYPES,
        capabilities=AgentCapabilities(streaming=False),
        skills=[AgentSkill(id="hotels", name="Hotels Scraper", description="Returns JSON hotels", tags=["scrape","hotels"])],
        preferred_transport="HTTP+JSON"
    )
    app=A2AStarletteApplication(agent_card=card, http_handler=DefaultRequestHandler(agent_executor=HotelsExecutor(), task_store=InMemoryTaskStore())).build()
    try:
        for r in app.routes:
            print("[routes]", getattr(r, "methods", "?"), getattr(r, "path", "?"))
    except Exception:
        pass

    uvicorn.run(app, host="localhost", port=12022)

if __name__=="__main__":
    main()
