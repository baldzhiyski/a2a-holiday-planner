import ast
import json, os, re
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
    SUPPORTED_CONTENT_TYPES = ["text/plain"]

    def __init__(self):
        self.llm = LLM(model="openai/gpt-4o-mini")
        self.search = SerperDevTool()
        self.scrape = ScrapeWebsiteTool()

        self.agent = Agent(
            role="FlightsScraper",
            goal=("Search reliable aggregators (e.g., Google Flights, Skyscanner, etc.), "
                  "scrape result pages, and return a structured JSON object with a 'flights' array."),
            backstory="Expert at flight meta-search; outputs only valid JSON under key 'flights'.",
            tools=[self.search, self.scrape],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    def _plan_task(self, query: FlightQuery) -> Task:
        prompt = (
            "You must return a STRICT JSON OBJECT (not text) with this structure:\n\n"
            "{{\n"
            '  "flights": [\n'
            "    {{\n"
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
            "    }}\n"
            "  ]\n"
            "}}\n\n"
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
            dap=query.depart_after, rap=query.return_after,
            pax=query.passengers
        )

        return Task(
            description=prompt,
            expected_output="Strict JSON object with key 'flights' containing an array of flight objects.",
            agent=self.agent,
            output_json=FlightList,
        )

    def _extract_first_json(self, s: str):
        """Tolerant extractor: handles prose, code fences, JSON, and Python-literal dicts/lists."""
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
            # 1) Try strict JSON first
            try:
                return json.loads(c)
            except Exception:
                pass
            # 2) Try Python literal (Crew often returns single-quoted dicts/lists)
            try:
                data = ast.literal_eval(c)
                if isinstance(data, (dict, list)):
                    return data
            except Exception:
                pass

        return None

    def invoke_structured(self, raw: str) -> str:
        # 0) Validate input JSON
        try:
            q = FlightQuery.model_validate_json(raw)
        except Exception as e:
            print("[flights] bad input JSON:", e)
            print("[flights] raw was:", raw[:400])
            return '{"flights": []}'

        # 0.5) Stub when keys are missing (lets you test wiring)
        if not os.getenv("OPENAI_API_KEY") or not os.getenv("SERPER_API_KEY"):
            print("[flights] missing API keys; returning stub data")
            return json.dumps({
                "flights": [
                    {
                        "source": q.origin,
                        "dest": q.dest,
                        "depart_iso": f"{q.start_date}T10:45:00",
                        "arrive_iso": f"{q.start_date}T13:15:00",
                        "airline": "Vueling",
                        "flight_no": "VY1885",
                        "duration_min": 150,
                        "price_eur": 145.0,
                        "cabin": "Economy",
                        "link": "https://example/fl1",
                        "source_site": "stub"
                    },
                    {
                        "source": q.dest,
                        "dest": q.origin,
                        "depart_iso": f"{q.end_date}T16:10:00",
                        "arrive_iso": f"{q.end_date}T18:45:00",
                        "airline": "easyJet",
                        "flight_no": "U21834",
                        "duration_min": 155,
                        "price_eur": 160.0,
                        "cabin": "Economy",
                        "link": "https://example/fl2",
                        "source_site": "stub"
                    }
                ]
            }, indent=2)

        # 1) Run Crew
        task = self._plan_task(q)
        crew = Crew(agents=[self.agent], tasks=[task], process=Process.sequential, verbose=True)
        out = crew.kickoff()

        # 2) Log raw output
        s = out if isinstance(out, str) else str(out)
        print("[flights] raw crew output (first 600):\n", s[:600])

        # 3) Extract JSON robustly
        data = self._extract_first_json(s)
        if data is None:
            print("[flights] JSON extraction failed; returning empty.")
            return '{"flights": []}'

        # Accept either {"flights":[...]} or [...]
        if isinstance(data, list):
            data = {"flights": data}

        # 4) Validate and serialize
        try:
            validated = FlightList.model_validate(data)
            print("[flights] validated items:", len(validated.flights))
            return validated.model_dump_json(indent=2)
        except ValidationError as e:
            print("[flights] schema validation failed:", e)
            return '{"flights": []}'
        except Exception as e:
            print("[flights] unexpected error:", e)
            return '{"flights": []}'

