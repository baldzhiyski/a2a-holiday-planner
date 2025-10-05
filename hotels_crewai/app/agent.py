import json, os, re, ast, logging
from typing import List
from pydantic import BaseModel, ValidationError
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
    SUPPORTED_CONTENT_TYPES = ["text/plain"]

    def __init__(self):
        # NOTE: Set GOOGLE_API_KEY and SERPER_API_KEY in your .env
        self.llm = LLM(
            model="openai/gpt-4o"
        )
        self.search = SerperDevTool()
        self.scrape  = ScrapeWebsiteTool(verbose=True)

        self.agent = Agent(
            role="HotelsScraper",
            goal=("Search hotel listings for the target dates/city; prefer boutique, ≥min_rating, walkable areas if asked; "
                  "return STRICT JSON object with key 'hotels' only."),
            backstory="Hotel meta-search & extraction specialist",
            tools=[self.search, self.scrape],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    def _task(self, q: HotelQuery) -> Task:
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

        return Task(
            description=desc,
            expected_output="Strict JSON object with key 'hotels'.",
            agent=self.agent,
            output_json=Hotels,
        )

    def _extract_first_json(self, s: str):
        """Tolerant extractor: handles prose, code fences, JSON, and Python-literal dict/list."""
        candidates = [s]

        # strip ```json fences if present
        fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(),
                        flags=re.IGNORECASE | re.MULTILINE)
        if fenced != s:
            candidates.append(fenced)

        # find first {...} or [...] block
        m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
        if m:
            candidates.append(m.group(1))

        for c in candidates:
            # try strict JSON
            try:
                return json.loads(c)
            except Exception:
                pass
            # try Python literal (single quotes)
            try:
                data = ast.literal_eval(c)
                if isinstance(data, (dict, list)):
                    return data
            except Exception:
                pass
        return None

    def invoke_structured(self, raw: str) -> str:
        # 0) Validate input JSON (from Host)
        try:
            q = HotelQuery.model_validate_json(raw)
        except Exception as e:
            print("[hotels] bad input JSON:", e)
            print("[hotels] raw was:", raw[:400])
            return '{"hotels": []}'

        # 0.5) Stub mode if keys missing (lets you test wiring fast)
        if not os.getenv("OPENAI_API_KEY") or not os.getenv("SERPER_API_KEY"):
            print("[hotels] missing API keys; returning stub data")
            return json.dumps({
                "hotels": [
                    {
                        "name": "Casa Boutique Eixample",
                        "address": "Carrer d'Aragó 255, Barcelona",
                        "checkin_iso": f"{q.checkin}T15:00:00",
                        "checkout_iso": f"{q.checkout}T11:00:00",
                        "rating": 4.6,
                        "price_total_eur": 780.0,
                        "district": "Eixample",
                        "link": "https://example/h1",
                        "source_site": "stub"
                    },
                    {
                        "name": "Hotel Born Ramblas",
                        "address": "La Rambla 120, Barcelona",
                        "checkin_iso": f"{q.checkin}T15:00:00",
                        "checkout_iso": f"{q.checkout}T11:00:00",
                        "rating": 4.4,
                        "price_total_eur": 650.0,
                        "district": "Ciutat Vella",
                        "link": "https://example/h2",
                        "source_site": "stub"
                    }
                ]
            }, indent=2)

        # 1) Run Crew
        task = self._task(q)
        crew = Crew(agents=[self.agent], tasks=[task], process=Process.sequential, verbose=True)
        out = crew.kickoff()

        # 2) Log raw output
        s = out if isinstance(out, str) else str(out)
        print("[hotels] raw crew output (first 1000):\n", s[:1000], flush=True)

        # 3) Extract JSON robustly
        data = self._extract_first_json(s)
        if data is None:
            print("[hotels] JSON extraction failed; returning empty.")
            return '{"hotels": []}'

        # Accept either {"hotels":[...]} or a bare array
        if isinstance(data, list):
            data = {"hotels": data}

        # 4) Validate against schema and return
        try:
            validated = Hotels.model_validate(data)
            print("[hotels] validated items:", len(validated.hotels))
            return validated.model_dump_json(indent=2)
        except ValidationError as e:
            print("[hotels] schema validation failed:", e)
            return '{"hotels": []}'
        except Exception as e:
            print("[hotels] unexpected error:", e)
            return '{"hotels": []}'

