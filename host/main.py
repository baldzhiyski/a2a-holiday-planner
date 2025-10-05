import asyncio, logging, uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from host.app.host_graph import TripPlannerHost
from host.app.agent_executor import HostAgentExecutor

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

async def build_app():
    host=TripPlannerHost()
    await host.init_remotes([
        "http://localhost:12021", # FlightsScraper
        "http://localhost:12022", # HotelsScraper
        "http://localhost:12023", # ActivitiesScraper
        "http://localhost:12024", # BudgetPolicy
    ])
    card=AgentCard(
        name="TripPlannerHost",
        description="Composes end-to-end trip itineraries by coordinating scraping agents and budget policy.",
        url="http://localhost:11020/",
        version="1.0.0",
        default_input_modes=["text","text/plain"],
        default_output_modes=["text","text/plain"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[AgentSkill(
            id="trip_planning",
            name="Trip Planning",
            description="Plan & book a multi-day trip given constraints, using scraped flights/hotels/activities.",
            tags=["travel","scraping","itinerary"]
        )]
    )
    handler=DefaultRequestHandler(agent_executor=HostAgentExecutor(host), task_store=InMemoryTaskStore())
    return A2AStarletteApplication(agent_card=card, http_handler=handler).build()

def main():
    uvicorn.run(asyncio.run(build_app()), host="localhost", port=11020)

if __name__=="__main__":
    main()
