import json
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from dotenv import load_dotenv

load_dotenv()

memory=MemorySaver()

class BudgetOutput(BaseModel):
    flight_cap_eur: float
    hotel_cap_eur: float
    activities_cap_eur: float
    notes: str | None = None

class BudgetAgent:
    SUPPORTED_CONTENT_TYPES=["text/plain"]
    SYSTEM=(
        "You are BudgetPolicy. Input includes total_budget_eur and passengers. "
        "Allocate caps: flights ~45%, hotel ~40%, activities ~15%, with small adjustments for 2+ pax. "
        "Return a STRICT JSON object matching BudgetOutput."
    )

    def __init__(self):
        self.llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.graph=create_react_agent(
            self.llm,
            tools=[self.compute_caps],
            checkpointer=memory,
            prompt=self.SYSTEM,
            response_format=BudgetOutput
        )

    @tool
    def compute_caps(self, total_budget_eur: int, passengers: int) -> str:
        """
        Compute budget caps for flights, hotel, and activities.

                Args:
                    total_budget_eur: Total trip budget in EUR.
                    passengers: Number of travelers.

                Returns:
                    JSON string matching BudgetOutput with keys:
                    flight_cap_eur, hotel_cap_eur, activities_cap_eur, notes.
        """
        tb=float(total_budget_eur)
        # baselines
        flights=tb*0.45
        hotel=tb*0.40
        activities=tb*0.15
        if passengers>=2:
            activities*=1.1
            hotel*=0.95
        return json.dumps({
            "flight_cap_eur": round(flights,2),
            "hotel_cap_eur": round(hotel,2),
            "activities_cap_eur": round(activities,2),
            "notes":"Baseline caps; tweak per market."
        })
