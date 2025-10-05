import json, os
from typing import List
from pydantic import BaseModel, ValidationError
from crewai import LLM, Agent, Crew, Process, Task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from dotenv import load_dotenv

load_dotenv()

# -------- Input schema from Host --------
class FlightQuery(BaseModel):
    origin: str
    dest: str
    start_date: str
    end_date: str
    passengers: int = 1
    no_redeye: bool = True
    depart_after: str = "00:00"
    return_after: str = "00:00"

# -------- Output schema items (mirror Host’s Flight) --------
class FlightItem(BaseModel):
    source: str
    dest: str
    depart_iso: str
    arrive_iso: str
    airline: str
    flight_no: str | None = None
    duration_min: int
    price_eur: float
    cabin: str | None = "Economy"
    link: str | None = None
    source_site: str | None = None

# -------- Wrapper object for CrewAI output --------
class FlightList(BaseModel):
    flights: List[FlightItem]


class FlightsScraperAgent:
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        self.llm = LLM(model="gpt-4o-mini")
        self.search = SerperDevTool()
        self.scrape = ScrapeWebsiteTool()

        self.agent = Agent(
            role="FlightsScraper",
            goal=("Search reliable aggregators (e.g., Google Flights, Skyscanner, etc.), "
                  "scrape result pages, and return a structured JSON object with a 'flights' array."),
            backstory="Expert at flight meta-search; outputs only valid JSON under key 'flights'.",
            tools=[self.search, self.scrape],
            llm=self.llm,
            verbose=False,
            allow_delegation=False,
        )

    def _plan_task(self, query: FlightQuery) -> Task:
        prompt = (
            "You must return a STRICT JSON OBJECT (not text) with this structure:\n\n"
            "{\n"
            '  "flights": [\n'
            "    {\n"
            '      "source": "JFK",\n'
            '      "dest": "LHR",\n'
            '      "depart_iso": "2025-07-01T09:30:00",\n'
            '      "arrive_iso": "2025-07-01T20:30:00",\n'
            '      "airline": "British Airways",\n'
            '      "flight_no": "BA178",\n'
            '      "duration_min": 480,\n'
            '      "price_eur": 450.50,\n'
            '      "cabin": "Economy",\n'
            '      "link": "https://...",\n'
            '      "source_site": "Skyscanner"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Constraints: from {origin} to {dest} on {out}, return on {ret}; "
            "depart after {dap} local, return after {rap} local; no red-eyes if requested; {pax} passengers.\n\n"
            "Steps:\n"
            "1) Search 1–2 aggregator result pages (Google Flights, Skyscanner, etc.)\n"
            "2) Use scrape tool to extract data.\n"
            "3) Normalize to the schema above.\n"
            "4) Return ONLY the JSON object, no commentary."
        ).format(
            origin=query.origin, dest=query.dest,
            out=query.start_date, ret=query.end_date,
            dap=query.depart_after, rap=query.return_after, pax=query.passengers
        )

        return Task(
            description=prompt,
            expected_output="Strict JSON object with key 'flights' containing an array of flight objects.",
            agent=self.agent,
            output_json=FlightList,
        )

    def invoke_structured(self, raw: str) -> str:
        # raw is a JSON string from Host with FlightQuery fields
        q = FlightQuery.model_validate_json(raw)
        task = self._plan_task(q)
        crew = Crew(agents=[self.agent], tasks=[task], process=Process.sequential, verbose=False)
        out = crew.kickoff()

        # Convert output to JSON safely
        s = out if isinstance(out, str) else str(out)
        try:
            data = json.loads(s)
            # Validate against FlightList (ensures object with 'flights' key)
            validated = FlightList.model_validate(data)
            return validated.model_dump_json(indent=2)
        except ValidationError as e:
            print("Validation failed:", e)
            return '{"flights": []}'
        except Exception as e:
            print("JSON parsing error:", e)
            return '{"flights": []}'
