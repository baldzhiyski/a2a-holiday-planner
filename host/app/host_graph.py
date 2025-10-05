import json, uuid, re, asyncio
from datetime import datetime
from typing import Any, Dict, List

import httpx
from a2a.client import A2ACardResolver
from a2a.types import AgentCard, MessageSendParams, SendMessageRequest, SendMessageResponse, SendMessageSuccessResponse, Task
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from pydantic import BaseModel

from host.app.schemas import (
    TripRequest,       # <-- NEW
    Flight, Hotel, Activity,
    BudgetPolicyDecision, CandidateItinerary
)
from host.app.itinerary_tools import align_windows, choose_activities, score_itinerary
from host.app.remote_connection import RemoteAgentConnections

from dotenv import load_dotenv

load_dotenv()

memory = MemorySaver()

# --- Structured response model for Host's final message ---
class HostResponse(BaseModel):
    status: str
    message: str

class TripPlannerHost:
    def __init__(self):
        self.remote: Dict[str, RemoteAgentConnections] = {}
        self.cards: Dict[str, AgentCard] = {}
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        self.extract_llm = self.llm.with_structured_output(TripRequest)

        self.graph = create_react_agent(
            self.llm,
            tools=[
                self.parse_user_query,   # <-- NEW tool
                self.send_message,
                self.list_agents,
                self.compose_itineraries,
                self.book_itinerary
            ],
            checkpointer=memory,
            prompt=self._system_prompt(),
            response_format=HostResponse,  # structured output
        )
        self._last_candidate_lists: Dict[str, List[dict]] = {}

    async def init_remotes(self, addresses: List[str]):
        async with httpx.AsyncClient(timeout=30) as client:
            for addr in addresses:
                card = await A2ACardResolver(client, addr).get_agent_card()
                self.remote[card.name] = RemoteAgentConnections(card, addr)
                self.cards[card.name] = card

    def _system_prompt(self) -> str:
        return (
            "You are TripPlannerHost. Extract: origin, dest, start/end, pax, budget EUR, prefs "
            "(walkable,boutique,no_redeye,depart_after,return_after).\n"
            "Tools:\n"
            "  - list_agents(): return the connected A2A agents (call this when the user asks who is available).\n"
            "  - send_message(agent_name, task, thread_id): ask a remote agent to do something.\n"
            "  - compose_itineraries(...): build and rank options.\n"
            "  - book_itinerary(option_index, context_id): confirm an option.\n"
            "Remote agents respond with JSON wrappers as documented.\n"
            "Reply concisely."
        )

    # NEW: LLM parser tool that returns a strict TripRequest as JSON
    @tool
    def parse_user_query(self, text: str) -> str:
        """
        Parse a free-text user message into a strict TripRequest JSON.
        Normalize dates to YYYY-MM-DD. If currency is mentioned, normalize to EUR numeric.
        Only infer defaults if the user is clearly ambiguous.
        """
        req = self.extract_llm.invoke(
            "Extract trip planning parameters from this user message.\n"
            "Return only the structured object; do not add commentary.\n\n"
            f"Message:\n{text}"
        )
        return req.model_dump_json()

    @tool
    def list_agents(self) -> str:
        """
        Return a strict JSON array of connected agents with name, description, skills, and url.
        """
        data = []
        for card in self.cards.values():
            data.append({
                "name": card.name,
                "description": card.description,
                "skills": [getattr(s, "name", None) for s in (card.skills or [])],
                "url": card.url,
            })
        return json.dumps(data, ensure_ascii=False)

    @tool
    async def send_message(self, agent_name: str, task: str, thread_id: str) -> str:
        """Send a JSON task string to a remote A2A agent and return concatenated text artifacts."""
        if agent_name not in self.remote:
            return f"ERROR: agent {agent_name} not found"
        msg_id = str(uuid.uuid4())
        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task}],
                "messageId": msg_id,
                "taskId": msg_id,
                "contextId": thread_id
            }
        }
        req = SendMessageRequest(id=msg_id, params=MessageSendParams.model_validate(payload))
        resp: SendMessageResponse = await self.remote[agent_name].send_message(req)
        if not (isinstance(resp.root, SendMessageSuccessResponse) and isinstance(resp.root.result, Task)):
            return "ERROR: remote call failed."
        data = json.loads(resp.root.model_dump_json(exclude_none=True))
        texts: List[str] = []
        for art in data.get("result", {}).get("artifacts", []):
            for p in art.get("parts", []):
                if p.get("type") == "text":
                    texts.append(p["text"])
        return "\n".join(texts) if texts else "(no response)"

    @tool
    async def compose_itineraries(
        self,
        flights_json: str,
        hotels_json: str,
        activities_json: str,
        budget_json: str,
        start_date: str,
        end_date: str,
        prefs_json: str,
        context_id: str
    ) -> str:
        """Combine flights, hotels, and activities into up to 3 ranked candidate itineraries."""
        try:
            # Helper to unwrap wrapper objects (dict) or accept list directly
            def _unwrap(data, key):
                return data.get(key, []) if isinstance(data, dict) else data

            flights_raw = _unwrap(json.loads(flights_json), "flights")
            hotels_raw = _unwrap(json.loads(hotels_json), "hotels")
            activities_raw = _unwrap(json.loads(activities_json), "items")

            flights = [Flight.model_validate(x) for x in flights_raw]
            hotels = [Hotel.model_validate(x) for x in hotels_raw]
            activities = [Activity.model_validate(x) for x in activities_raw]

            budget = BudgetPolicyDecision.model_validate(json.loads(budget_json))
            prefs = json.loads(prefs_json or "{}")
        except Exception as e:
            return json.dumps({"status": "error", "message": f"bad json: {e}"})

        # Split outbound/inbound by origin
        origin = prefs.get("origin", "")
        outs = [f for f in flights if f.source.lower() == origin.lower()]
        ins = [f for f in flights if f.dest.lower() == origin.lower()]

        cands: List[dict] = []
        for o in outs:
            for i in ins:
                for h in hotels:
                    if not align_windows(o, i, h):
                        continue
                    # naive per-day budget heuristic
                    per_day_budget = budget.activities_cap_eur / max(1, ((len(activities) // 3) or 1))
                    days = choose_activities(activities, start_date, end_date, per_day_budget=per_day_budget)
                    acts_sum = sum(a.price_eur for d in days for a in d.booked_activities)
                    total = o.price_eur + i.price_eur + h.price_total_eur + acts_sum
                    if total > prefs.get("budget_eur", 9e9):
                        continue
                    sc = score_itinerary(total, prefs, h, [a for d in days for a in d.booked_activities])
                    cands.append(CandidateItinerary(
                        summary=f"{prefs.get('origin','?')} → {prefs.get('dest','?')} {start_date}–{end_date}, {h.name}",
                        outbound=o, inbound=i, hotel=h, days=days,
                        price_breakdown_eur={
                            "outbound": o.price_eur,
                            "inbound": i.price_eur,
                            "hotel": h.price_total_eur,
                            "activities": acts_sum
                        },
                        total_eur=total, score=sc,
                        hold_links={"flights": o.link or "", "hotel": h.link or ""}
                    ).model_dump())

        cands = sorted(cands, key=lambda x: (-x["score"], x["total_eur"]))[:3]
        self._last_candidate_lists[context_id] = cands
        return json.dumps({"status": "ok", "candidates": cands})

    @tool
    async def book_itinerary(self, option_index: int, context_id: str) -> str:
        """Confirm a selected itinerary and return booking confirmation references."""
        cands = self._last_candidate_lists.get(context_id) or []
        if not cands or option_index < 1 or option_index > len(cands):
            return json.dumps({"status": "error", "message": "Invalid option index"})
        chosen = cands[option_index - 1]
        conf = {
            "status": "success",
            "booking": {
                "flights_confirmation": f"FL-{uuid.uuid4().hex[:8]}",
                "hotel_confirmation": f"HT-{uuid.uuid4().hex[:8]}",
                "activities_count": sum(len(d['booked_activities']) for d in chosen["days"])
            },
            "itinerary": chosen
        }
        return json.dumps(conf)

    def invoke(self, text: str, thread_id: str):
        out = self.graph.invoke(
            {"messages": [{"role": "user", "content": text}]},
            config={"configurable": {"thread_id": str(thread_id)}}
        )
        return out["structured_response"].model_dump_json()
