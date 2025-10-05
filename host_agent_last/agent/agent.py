# host_trip_agent.py
import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, AsyncIterable, Dict, List, Optional

import httpx
import nest_asyncio
from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
)
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# Use the SAME connection wrapper as your working host (expects SendMessageRequest)
from host_agent_last.agent.remote_connection import RemoteAgentConnections

load_dotenv()
nest_asyncio.apply()


# -------------------- small utils --------------------
def _extract_text_artifacts(send_response: SendMessageResponse) -> List[str]:
    """Extract all text parts from an A2A SendMessageResponse."""
    if not isinstance(send_response.root, SendMessageSuccessResponse):
        return []
    raw = json.loads(send_response.root.model_dump_json(exclude_none=True))
    parts: List[str] = []
    for art in (raw.get("result", {}) or {}).get("artifacts", []) or []:
        for p in art.get("parts", []) or []:
            if p.get("type") == "text" and "text" in p:
                parts.append(p["text"])
    return parts


def _first_json(payloads: List[str]) -> Optional[dict]:
    """Return the first valid JSON object from a list of strings."""
    for txt in payloads:
        try:
            obj = json.loads(txt)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            continue
    return None


def _safe_get(d: dict, key: str, default):
    try:
        return d.get(key, default)
    except Exception:
        return default


# -------------------- Host Agent --------------------
class TripHostAgent:
    """ADK Host for multi-agent trip planning."""

    def __init__(self) -> None:
        self.remote_agent_connections: Dict[str, RemoteAgentConnections] = {}
        self.cards: Dict[str, AgentCard] = {}
        self.agents_shortlist: str = ""

        self._agent = self._create_agent()
        self._user_id = "trip_host"
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )
        self._last_candidates: Dict[str, List[dict]] = {}

    # ---------- bootstrapping remotes ----------
    async def _async_init_components(self, remote_agent_addresses: List[str]) -> None:
        async with httpx.AsyncClient(timeout=30) as client:
            for address in remote_agent_addresses:
                try:
                    card = await A2ACardResolver(client, address).get_agent_card()
                    self.remote_agent_connections[card.name] = RemoteAgentConnections(
                        agent_card=card, agent_url=address
                    )
                    self.cards[card.name] = card
                except Exception as e:
                    print(f"[host] Failed to init {address}: {e}")

        info_lines = [
            json.dumps(
                {
                    "name": c.name,
                    "description": c.description,
                    "skills": [s.name for s in (c.skills or [])],
                    "url": c.url,
                }
            )
            for c in self.cards.values()
        ]
        self.agents_shortlist = "\n".join(info_lines) if info_lines else "No agents discovered"

    @classmethod
    async def create(cls, remote_agent_addresses: List[str]) -> "TripHostAgent":
        inst = cls()
        await inst._async_init_components(remote_agent_addresses)
        return inst

    # ---------- ADK Agent ----------
    def _create_agent(self) -> Agent:
        return Agent(
            model="gemini-2.5-flash",
            name="Trip_Host",
            instruction=self._root_instruction,
            description="Orchestrates flights, hotels, activities and budget policy via A2A agents.",
            tools=[
                self.send_message,           # generic A2A bridge
                self.list_connected_agents,  # discovery
                self.plan_trip,              # high-level orchestration
                self.book_itinerary,         # confirm a candidate
            ],
        )

    def _root_instruction(self, _: ReadonlyContext) -> str:
        return f"""
**Role:** You are the Trip Host. You coordinate with A2A “friend agents” to plan a multi-day trip end-to-end.

**How to work:**
- Use `list_connected_agents()` if asked which agents are available.
- When planning, call `send_message` to contact:
  • **BudgetPolicy** — to split the total budget into per-category caps.  
  • **FlightsScraper** — to get flight options.  
  • **HotelsScraper** — to get hotel options.  
  • **ActivitiesScraper** — to get tours & activities.
- Then call `plan_trip(...)` to compose top itineraries from the JSON returned by those agents.
- When a user picks an option, call `book_itinerary(option_index)` to confirm (mock booking).

**Input hints:**
- Ask clarifying questions only when critical fields are missing (origin, destination, dates, pax, budget).
- Keep responses concise; use bullet points for options.

**Today's Date:** {datetime.now().strftime("%Y-%m-%d")}

<Available Agents JSON (name/description/skills/url)>
{self.agents_shortlist}
</Available Agents JSON>
        """.strip()

    # ---------- Streaming to ADK ----------
    async def stream(self, query: str, session_id: str) -> AsyncIterable[dict[str, Any]]:
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name, user_id=self._user_id, session_id=session_id
        )
        content = types.Content(role="user", parts=[types.Part.from_text(text=query)])
        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name, user_id=self._user_id, state={}, session_id=session_id
            )

        async for event in self._runner.run_async(
            user_id=self._user_id, session_id=session.id, new_message=content
        ):
            if event.is_final_response():
                text = ""
                if event.content and event.content.parts:
                    text = "\n".join([p.text for p in event.content.parts if getattr(p, "text", "")])
                yield {"is_task_complete": True, "content": text}
            else:
                yield {"is_task_complete": False, "updates": "Trip Host is coordinating…"}

    # ---------- Tools ----------
    async def send_message(self, agent_name: str, task: str, tool_context: ToolContext):
        """Bridge to a remote agent. Payload is a JSON string in `task`."""
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f"Agent {agent_name} not found")
        client = self.remote_agent_connections[agent_name]

        # ID management mirrors your working host
        state = tool_context.state
        task_id = state.get("task_id", str(uuid.uuid4()))
        context_id = state.get("context_id", str(uuid.uuid4()))
        message_id = str(uuid.uuid4())

        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task}],
                "messageId": message_id,
                "taskId": task_id,
                "contextId": context_id,
            },
        }

        # Build the SAME envelope as the working host
        message_request = SendMessageRequest(
            id=message_id, params=MessageSendParams.model_validate(payload)
        )

        # If your RemoteAgentConnections was switched to ClientFactory that expects a "Message",
        # replace the next line with:
        #   send_response = await client.send_message(message_request.params.message)
        send_response: SendMessageResponse = await client.send_message(message_request)
        print("[trip host] send_response", send_response)

        if not isinstance(send_response.root, SendMessageSuccessResponse) or not isinstance(
            send_response.root.result, Task
        ):
            print("[trip host] Received a non-success or non-task response. Cannot proceed.")
            return

        response_content = send_response.root.model_dump_json(exclude_none=True)
        json_content = json.loads(response_content)

        resp: List[str] = []
        if json_content.get("result", {}).get("artifacts"):
            for artifact in json_content["result"]["artifacts"]:
                if artifact.get("parts"):
                    # collect raw text parts (identical to your working host)
                    for p in artifact["parts"]:
                        if isinstance(p, dict) and p.get("type") == "text" and "text" in p:
                            resp.append(p["text"])
        return resp

    def list_connected_agents(self) -> str:
        items = []
        for c in self.cards.values():
            items.append(
                {
                    "name": c.name,
                    "description": c.description,
                    "skills": [getattr(s, "name", None) for s in (c.skills or [])],
                    "url": c.url,
                }
            )
        return json.dumps(items, ensure_ascii=False)

    async def plan_trip(
        self,
        origin: str,
        dest: str,
        start_date: str,
        end_date: str,
        passengers: int,
        total_budget_eur: float,
        tool_context: ToolContext,
        walkable: bool = True,
        boutique: bool = True,
        no_redeye: bool = True,
        depart_after: str = "09:00",
        return_after: str = "14:00",
    ) -> str:
        ctx_id = tool_context.state.get("context_id", "ctx-" + uuid.uuid4().hex[:8])

        # 1) BudgetPolicy
        bp_task = json.dumps({"total_budget_eur": int(total_budget_eur), "passengers": passengers})
        bp_texts = await self.send_message("BudgetPolicy", bp_task, tool_context)
        bp_json = _first_json(bp_texts or []) or {}

        # 2) Flights
        fl_task = json.dumps(
            {
                "origin": origin,
                "dest": dest,
                "start_date": start_date,
                "end_date": end_date,
                "passengers": passengers,
                "no_redeye": no_redeye,
                "depart_after": depart_after,
                "return_after": return_after,
            }
        )
        fl_texts = await self.send_message("FlightsScraper", fl_task, tool_context)
        fl_json = _first_json(fl_texts or []) or {}
        flights: List[dict] = _safe_get(fl_json, "flights", [])

        # 3) Hotels
        ht_task = json.dumps(
            {
                "city": dest,
                "checkin": start_date,
                "checkout": end_date,
                "min_rating": 4.0 if boutique else 0.0,
                "style": "boutique" if boutique else "any",
                "walkable": walkable,
                "max_total_eur": total_budget_eur * 0.6,
            }
        )
        ht_texts = await self.send_message("HotelsScraper", ht_task, tool_context)
        ht_json = _first_json(ht_texts or []) or {}
        hotels: List[dict] = _safe_get(ht_json, "hotels", [])

        # 4) Activities
        ac_task = json.dumps(
            {
                "city": dest,
                "date_from": start_date,
                "date_to": end_date,
                "categories": ["food tour", "walking tour", "museum", "viewpoint"],
                "min_rating": 4.5,
                "max_price_eur": 140,
            }
        )
        ac_texts = await self.send_message("ActivitiesScraper", ac_task, tool_context)
        ac_json = _first_json(ac_texts or []) or {}
        activities: List[dict] = _safe_get(ac_json, "items", [])

        # 5) Compose candidates (simple heuristic)
        candidates: List[dict] = []
        outs = [f for f in flights if f.get("source", "").lower() == origin.lower()]
        ins = [f for f in flights if f.get("dest", "").lower() == origin.lower()]

        for o in outs[:5]:
            for i in ins[:5]:
                for h in hotels[:5]:
                    if not (o.get("depart_iso") and i.get("depart_iso")):
                        continue
                    picked = activities[: max(0, min(6, len(activities)))]
                    acts_total = sum([a.get("price_eur", 0) for a in picked])

                    total_cost = (
                        float(o.get("price_eur", 0))
                        + float(i.get("price_eur", 0))
                        + float(h.get("price_total_eur", 0))
                        + acts_total
                    )
                    if total_cost > total_budget_eur:
                        continue

                    cand = {
                        "summary": f"{origin} → {dest} {start_date}–{end_date}, {h.get('name','?')}",
                        "outbound": o,
                        "inbound": i,
                        "hotel": h,
                        "days": [],
                        "price_breakdown_eur": {
                            "outbound": o.get("price_eur", 0),
                            "inbound": i.get("price_eur", 0),
                            "hotel": h.get("price_total_eur", 0),
                            "activities": acts_total,
                        },
                        "total_eur": total_cost,
                        "hold_links": {
                            "flights": o.get("link", ""),
                            "hotel": h.get("link", ""),
                        },
                    }
                    candidates.append(cand)

        candidates = sorted(candidates, key=lambda c: c["total_eur"])[:3]
        self._last_candidates[ctx_id] = candidates

        if not candidates:
            return (
                "No viable options under the given constraints. "
                "Try adjusting dates, budget, or preferences (walkable/boutique/no_redeye)."
            )

        lines = ["Top itinerary options:"]
        for i, c in enumerate(candidates, 1):
            pb = c["price_breakdown_eur"]
            flights_sum = (pb.get("outbound", 0) or 0) + (pb.get("inbound", 0) or 0)
            lines.append(
                f"{i}) {c['summary']} — Total €{c['total_eur']:.0f} "
                f"(Flights €{flights_sum:.0f}, Hotel €{pb.get('hotel',0):.0f}, "
                f"Activities €{pb.get('activities',0):.0f})"
            )
        lines.append("\nSay: `Book option N` to confirm.")
        return "\n".join(lines) + "\n\n" + json.dumps({"candidates": candidates}, ensure_ascii=False)

    async def book_itinerary(self, option_index: int, tool_context: ToolContext) -> str:
        ctx_id = tool_context.state.get("context_id", "unknown")
        cands = self._last_candidates.get(ctx_id) or []
        if not cands or option_index < 1 or option_index > len(cands):
            return json.dumps({"status": "error", "message": "Invalid option index"})
        chosen = cands[option_index - 1]
        conf = {
            "status": "success",
            "booking": {
                "flights_confirmation": f"FL-{uuid.uuid4().hex[:8]}",
                "hotel_confirmation": f"HT-{uuid.uuid4().hex[:8]}",
                "activities_count": 0,
            },
            "itinerary": chosen,
        }
        return json.dumps(conf, ensure_ascii=False)


# -------------------- factory (sync helper) --------------------
def build_trip_host_agent_sync() -> Agent:
    """Create & init the TripHostAgent synchronously (for notebooks / simple scripts)."""
    async def _async_main():
        remote_urls = [
            "http://localhost:12021",  # FlightsScraper
            "http://localhost:12022",  # HotelsScraper
            "http://localhost:12023",  # ActivitiesScraper
            "http://localhost:12024",  # BudgetPolicy
        ]
        host = await TripHostAgent.create(remote_urls)
        return host._create_agent()

    try:
        return asyncio.run(_async_main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called" in str(e):
            print(f"[host] Warning: {e}")
        else:
            raise


root_agent = build_trip_host_agent_sync()
