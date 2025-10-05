import json, re, ast
from typing import List
from pydantic import BaseModel, Field, ValidationError
from crewai import LLM, Agent, Crew, Process, Task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from dotenv import load_dotenv

load_dotenv()


class ActivitiesQuery(BaseModel):
    city: str
    date_from: str
    date_to: str
    categories: List[str] = Field(default_factory=list)
    min_rating: float = 0.0
    max_price_eur: float = 999999.0

class ActivityItem(BaseModel):
    title: str
    date_iso: str
    start_local: str
    end_local: str
    price_eur: float
    category: str
    rating: float
    link: str | None = None
    source_site: str | None = None
    location_hint: str | None = None

class ActivitiesResponse(BaseModel):
    items: List[ActivityItem]

class ActivitiesScraperAgent:
    SUPPORTED_CONTENT_TYPES=["text","text/plain"]

    def __init__(self):
        self.llm = LLM(model="openai/gpt-4o-mini")
        self.search = SerperDevTool()
        self.scrape = ScrapeWebsiteTool(verbose=True)
        self.agent = Agent(
            role="ActivitiesScraper",
            goal=("Find top-rated activities/tours within dates & budget; "
                  "return a STRICT JSON object with key 'items'."),
            backstory="Tour & activity aggregator specialist",
            tools=[self.search, self.scrape],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def _task(self, q: ActivitiesQuery)->Task:
        desc = (
            "You must return a STRICT JSON OBJECT (not text) with this structure:\n\n"
            "{\n"
            '  "items": [\n'
            "    {\n"
            '      "title": "Sunset Sailing Tour",\n'
            '      "date_iso": "2025-08-15T18:00:00",\n'
            '      "start_local": "18:00",\n'
            '      "end_local": "21:00",\n'
            '      "price_eur": 85.0,\n'
            '      "category": "Boat Tour",\n'
            '      "rating": 4.7,\n'
            '      "link": "https://...",\n'
            '      "source_site": "GetYourGuide",\n'
            '      "location_hint": "Port of Barcelona"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"Search parameters:\n"
            f"- City: {q.city}\n"
            f"- Date range: {q.date_from} → {q.date_to}\n"
            f"- Categories: {', '.join(q.categories) if q.categories else 'Any'}\n"
            f"- Minimum rating: {q.min_rating}\n"
            f"- Maximum price: €{q.max_price_eur:.0f}\n\n"
            "Steps:\n"
            "1) Use the search tool to find 1–3 reputable activity/tour aggregators "
            "(e.g., GetYourGuide, Viator, TripAdvisor Experiences).\n"
            "2) Use the scrape tool to extract relevant activities from those result pages.\n"
            "3) Collect 6–12 high-quality, well-rated activities matching filters.\n"
            "4) Normalize date/time to ISO 8601 and 24h local time.\n"
            "5) Return ONLY the JSON object — no text, no markdown, no code fences."
        )
        return Task(description=desc, expected_output="Strict JSON object with key 'items'.", agent=self.agent, output_json=ActivitiesResponse)

    def _extract_first_json(self, s: str):
        """Handle JSON, ```json fences, or Python dict/list with single quotes."""
        candidates = [s]

        # Strip code fences
        fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(),
                        flags=re.IGNORECASE | re.MULTILINE)
        if fenced != s:
            candidates.append(fenced)

        # First {...} or [...] block
        m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
        if m:
            candidates.append(m.group(1))

        for c in candidates:
            # Try strict JSON
            try:
                return json.loads(c)
            except Exception:
                pass
            # Try Python literal (single quotes)
            try:
                val = ast.literal_eval(c)
                if isinstance(val, (dict, list)):
                    return val
            except Exception:
                pass
        return None

    def invoke_structured(self, raw: str) -> str:
        # Validate input JSON
        try:
            q = ActivitiesQuery.model_validate_json(raw)
        except Exception as e:
            print("[activities] bad input JSON:", e)
            print("[activities] raw was:", raw[:400])
            return '{"items": []}'

        crew = Crew(agents=[self.agent], tasks=[self._task(q)], process=Process.sequential, verbose=True)
        out = crew.kickoff()

        s = out if isinstance(out, str) else str(out)
        print("[activities] Raw LLM output (first 1000):\n", s[:1000], "\n", flush=True)

        data = self._extract_first_json(s)
        if data is None:
            print("[activities] JSON extraction failed; returning empty.")
            return '{"items": []}'

        if isinstance(data, list):
            data = {"items": data}

        try:
            validated = ActivitiesResponse.model_validate(data)
            print("[activities] validated items:", len(validated.items))
            return validated.model_dump_json(indent=2)
        except ValidationError as e:
            print("[activities] schema validation failed:", e)
            return '{"items": []}'
        except Exception as e:
            print("[activities] unexpected error:", e)
            return '{"items": []}'
