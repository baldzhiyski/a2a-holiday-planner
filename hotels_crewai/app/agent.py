import json
from typing import List
from pydantic import BaseModel, Field, ValidationError
from crewai import LLM, Agent, Crew, Process, Task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

from dotenv import load_dotenv

load_dotenv()

class HotelQuery(BaseModel):
    city: str
    checkin: str
    checkout: str
    min_rating: float = 0.0
    style: str = "any"
    walkable: bool = False
    max_total_eur: float = 999999.0

class HotelItem(BaseModel):
    name: str
    address: str
    checkin_iso: str
    checkout_iso: str
    rating: float
    price_total_eur: float
    district: str | None = None
    link: str | None = None
    source_site: str | None = None


class Hotels(BaseModel):
    hotels: List[HotelItem]

class HotelsScraperAgent:
    SUPPORTED_CONTENT_TYPES=["text","text/plain"]
    def __init__(self):
        self.llm=LLM(model="openai/gpt-4o-mini")
        self.search=SerperDevTool()
        self.scrape=ScrapeWebsiteTool()
        self.agent=Agent(
            role="HotelsScraper",
            goal=("Search hotel listings for the target dates/city; prefer boutique, ≥min_rating, walkable areas if asked; "
                  "return STRICT JSON array of hotel objects only."),
            backstory="Hotel meta-search & extraction specialist",
            tools=[self.search, self.scrape],
            llm=self.llm,
            verbose=False,
            allow_delegation=False
        )

    def _task(self, q: HotelQuery)->Task:
        desc = (
            "You must return a STRICT JSON OBJECT (not text) with this structure:\n\n"
            "{\n"
            '  "hotels": [\n'
            "    {\n"
            '      "name": "Hotel Central",\n'
            '      "address": "123 Main St, Berlin",\n'
            '      "checkin_iso": "2025-07-01T15:00:00",\n'
            '      "checkout_iso": "2025-07-05T11:00:00",\n'
            '      "rating": 4.5,\n'
            '      "price_total_eur": 620.0,\n'
            '      "district": "Mitte",\n'
            '      "link": "https://...",\n'
            '      "source_site": "Booking.com"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"Search parameters:\n"
            f"- City: {q.city}\n"
            f"- Check-in: {q.checkin}\n"
            f"- Check-out: {q.checkout}\n"
            f"- Minimum rating: {q.min_rating}\n"
            f"- Style preference: {q.style}\n"
            f"- Walkable area required: {q.walkable}\n"
            f"- Maximum total budget: €{q.max_total_eur:.0f}\n\n"
            "Steps:\n"
            "1) Use the search tool to find 1–2 reliable hotel aggregators (e.g., Booking.com, Hotels.com, Expedia).\n"
            "2) Use the scrape tool to extract hotel data from those pages.\n"
            "3) Collect 5–12 competitive hotel options.\n"
            "4) Normalize dates to ISO 8601 format (YYYY-MM-DDTHH:MM:SS).\n"
            "5) Return ONLY the JSON object, no commentary or markdown."
        )

        return Task(description=desc, expected_output="Strict JSON array only.", agent=self.agent, output_json=Hotels)

    def invoke_structured(self, raw: str) -> str:
        q = HotelQuery.model_validate_json(raw)
        out = Crew(agents=[self.agent], tasks=[self._task(q)], process=Process.sequential).kickoff()
        # Convert output to JSON safely
        s = out if isinstance(out, str) else str(out)
        try:
            data = json.loads(s)
            validated = Hotels.model_validate(data)
            return validated.model_dump_json(indent=2)
        except ValidationError as e:
            print("Validation failed:", e)
            return '{"hotels": []}'
        except Exception as e:
            print("JSON parsing error:", e)
            return '{"hotels": []}'
