# host_trip_agent.py
import asyncio
import json
import uuid
from datetime import datetime, date
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
from google.adk.memory import InMemoryMemoryService as _IMM  # keep alias stable if imports vary
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

# Your existing connection wrapper
from host_agent_last.agent.remote_connection import RemoteAgentConnections


load_dotenv()
nest_asyncio.apply()


# ---------- tolerant helpers ----------
def _collect_text_parts(send_response: SendMessageResponse) -> List[str]:
    """Collect *any* textual payloads from artifacts (no strict 'type' checks)."""
    if not isinstance(send_response.root, SendMessageSuccessResponse):
        return []
    try:
        raw = json.loads(send_response.root.model_dump_json(exclude_none=True))
    except Exception:
        return []
    parts_out: List[str] = []
    artifacts = (raw.get("result", {}) or {}).get("artifacts", []) or []
    for art in reversed(artifacts):  # newest first
        for p in (art.get("parts") or []):
            if isinstance(p, dict) and "text" in p and isinstance(p["text"], str):
                parts_out.append(p["text"])
            elif isinstance(p, str):
                parts_out.append(p)
            else:
                try:
                    parts_out.append(json.dumps(p, ensure_ascii=False))
                except Exception:
                    parts_out.append(str(p))
    return parts_out


def _first_json(parts: List[str]) -> Optional[dict]:
    """Return the first valid JSON object/list found across strings; else None."""
    for txt in parts:
        try:
            obj = json.loads(txt)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            continue
    return None


def _eu(n: float) -> str:
    try:
        return f"€{float(n):.0f}"
    except Exception:
        return "€–"


def _loose(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _match_leg(f: dict, src: str, dst: str) -> bool:
    """Loose match for BER/Berlin and BCN/Barcelona, etc."""
    fs, fd = _loose(f.get("source")), _loose(f.get("dest"))
    src, dst = _loose(src), _loose(dst)
    return (src in fs or fs in src) and (dst in fd or fd in dst)


def _mk_quick_picks(origin, dest, start_date, end_date, passengers, total_budget_eur, bp, fljs, htjs, acjs) -> List[dict]:
    """Best-effort: pair 1 out + 1 in + 1 hotel + a couple of activities."""
    flight_cap = float(bp.get("flight_cap_eur", total_budget_eur))
    hotel_cap = float(bp.get("hotel_cap_eur", total_budget_eur))

    flights  = fljs.get("flights", []) if isinstance(fljs, dict) else []
    hotels   = htjs.get("hotels",  []) if isinstance(htjs, dict) else []
    acts_all = acjs.get("items",   []) if isinstance(acjs, dict) else []

    outbounds = [f for f in flights if _match_leg(f, origin, dest)] or flights[:1]
    inbounds  = [f for f in flights if _match_leg(f, dest, origin)] or flights[-1:]

    hotels_sorted = sorted(
        hotels,
        key=lambda h: (
            float(h.get("price_total_eur", 1e9)) <= hotel_cap,
            float(h.get("rating", 0.0)),
            -float(h.get("price_total_eur", 0.0))
        ),
        reverse=True
    )
    acts_sorted = sorted(
        acts_all,
        key=lambda a: (float(a.get("rating", 0.0)), -float(a.get("price_eur", 0.0))),
        reverse=True
    )

    picks = []
    built = 0
    for o in outbounds[:3]:
        for i in inbounds[:3]:
            for h in hotels_sorted[:4]:
                price_out_pp = float(o.get("price_eur", 0.0) or 0.0)
                price_in_pp  = float(i.get("price_eur", 0.0) or 0.0)
                flight_total = (price_out_pp + price_in_pp) * max(1, passengers)
                hotel_total  = float(h.get("price_total_eur", 0.0) or 0.0)
                chosen_acts  = acts_sorted[:2]
                acts_total   = sum(float(a.get("price_eur", 0.0) or 0.0) for a in chosen_acts)
                grand        = flight_total + hotel_total + acts_total

                picks.append({
                    "hotel_name": h.get("name", "Hotel"),
                    "hotel_rating": h.get("rating", None),
                    "hotel_district": h.get("district", ""),
                    "hotel_price": hotel_total,
                    "flight_total": flight_total,
                    "out_pp": price_out_pp,
                    "in_pp": price_in_pp,
                    "acts_titles": [a.get("title", "Activity") for a in chosen_acts],
                    "acts_total": acts_total,
                    "grand_total": grand,
                    "links": {
                        "out": o.get("link", ""),
                        "back": i.get("link", ""),
                        "hotel": h.get("link", "")
                    }
                })
                built += 1
                if built >= 3:
                    return picks
    return picks


def _format_picks_table(picks: List[dict]) -> str:
    if not picks:
        return "_No quick picks available._"
    lines = ["| Option | Hotel (rating • area) | Costs | Links |",
             "|---:|---|---|---|"]
    for idx, p in enumerate(picks, 1):
        hotel = f"**{p['hotel_name']}**" + (f" (⭐ {p['hotel_rating']:.1f})" if p.get("hotel_rating") else "") \
                + (f" • {p['hotel_district']}" if p.get("hotel_district") else "")
        costs = (
            f"Total: **{_eu(p['grand_total'])}**  \n"
            f"Flights: {_eu(p['flight_total'])} (out {_eu(p['out_pp']*2)} / in {_eu(p['in_pp']*2)})  \n"
            f"Hotel: {_eu(p['hotel_price'])}  \n"
            + (f"Activities: {', '.join(p['acts_titles'])} ({_eu(p['acts_total'])})" if p.get("acts_titles") else "")
        )
        links = f"[out]({p['links']['out']}) • [back]({p['links']['back']}) • [hotel]({p['links']['hotel']})"
        lines.append(f"| **{idx}** | {hotel} | {costs} | {links} |")
    return "\n".join(lines)


# -------------------- Host Agent --------------------
class TripHostAgent:
    """Simple orchestrator: fetch crew outputs, show it cleanly, and propose quick picks."""

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
            memory_service=_IMM(),
        )

    # ---------- bootstrap remotes ----------
    async def _async_init_components(self, remote_agent_addresses: List[str]) -> None:
        async with httpx.AsyncClient() as client:
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
                {"name": c.name, "description": c.description,
                 "skills": [s.name for s in (c.skills or [])], "url": c.url}
            ) for c in self.cards.values()
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
            description="Fetch crew text, show it clearly, and propose quick picks (tolerant parsing).",
            tools=[
                self.list_connected_agents,
                self.plan_trip,  # pretty summary + quick picks + optional raw
            ],
        )

    def _root_instruction(self, _: ReadonlyContext) -> str:
        return f"""
You are the Trip Host. Keep things friendly and readable:
- Call remote agents (BudgetPolicy, FlightsScraper, HotelsScraper, ActivitiesScraper).
- Be tolerant: if JSON is present, lightly parse it to build a few "Quick Picks".
- Never fail hard on schema mismatches; fall back to showing raw crew text.
- Keep the response well-formatted for a human.

Today's Date: {datetime.now().strftime("%Y-%m-%d")}
<Agents>
{self.agents_shortlist}
</Agents>
        """.strip()

    # ---------- streaming ----------
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

    # ---------- private bridge ----------
    async def _send(self, agent_name: str, task_text: str) -> List[str]:
        if agent_name not in self.remote_agent_connections:
            return [f"[{agent_name}] not connected."]
        client = self.remote_agent_connections[agent_name]

        message_id = str(uuid.uuid4())
        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task_text}],
                "messageId": message_id,
            },
        }

        req = SendMessageRequest(id=message_id, params=MessageSendParams.model_validate(payload))
        resp: SendMessageResponse = await client.send_message(req)
        print("[trip host] send_response", resp)

        if not isinstance(resp.root, SendMessageSuccessResponse) or not isinstance(resp.root.result, Task):
            return [f"[{agent_name}] unexpected response."]
        return _collect_text_parts(resp)

    def list_connected_agents(self) -> str:
        items = []
        for c in self.cards.values():
            items.append(
                {"name": c.name, "description": c.description,
                 "skills": [getattr(s, "name", None) for s in (c.skills or [])], "url": c.url}
            )
        return json.dumps(items, ensure_ascii=False)

    # ---------- pretty planner ----------
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
        show_raw: bool = False,      # << new: hide raw by default
    ) -> str:

        # Simple JSON tasks for crews
        budget_task = json.dumps({"total_budget_eur": total_budget_eur, "passengers": passengers})
        flights_task = json.dumps({
            "origin": origin, "dest": dest,
            "start_date": start_date, "end_date": end_date,
            "passengers": passengers, "no_redeye": no_redeye,
            "depart_after": depart_after, "return_after": return_after
        })
        hotels_task = json.dumps({
            "city": dest, "checkin": start_date, "checkout": end_date,
            "min_rating": 4.0 if boutique else 0.0,
            "style": "boutique" if boutique else "any",
            "walkable": walkable, "max_total_eur": total_budget_eur * 0.6
        })
        activities_task = json.dumps({
            "city": dest, "date_from": start_date, "date_to": end_date,
            "categories": ["food tour", "walking tour", "museum", "viewpoint"],
            "min_rating": 4.5, "max_price_eur": 140
        })

        # Call all crews
        budget_texts     = await self._send("BudgetPolicy",      budget_task)
        flights_texts    = await self._send("FlightsScraper",    flights_task)
        hotels_texts     = await self._send("HotelsScraper",     hotels_task)
        activities_texts = await self._send("ActivitiesScraper", activities_task)

        # Light JSON extraction (tolerant)
        bp   = _first_json(budget_texts)     or {}
        fljs = _first_json(flights_texts)    or {}
        htjs = _first_json(hotels_texts)     or {}
        acjs = _first_json(activities_texts) or {}

        # Quick picks
        picks = _mk_quick_picks(origin, dest, start_date, end_date, passengers, total_budget_eur, bp, fljs, htjs, acjs)

        header = (
            f"## Trip plan: **{origin} → {dest}**\n"
            f"**Dates:** {start_date} → {end_date} • **Pax:** {passengers} • **Budget:** {_eu(total_budget_eur)}\n\n"
            f"### Quick picks\n{_format_picks_table(picks)}\n\n"
            "You can reply with: **Shortlist option 1**, **Swap hotel Casa Bonay**, **Refine flights**, "
            "**Replace activity Tapas tour**, etc.\n"
        )

        if not show_raw:
            return header + "\n_(Say: **show raw** to include raw crew payloads.)_"

        # Raw section (optional)
        def _block(title: str, parts: List[str]) -> str:
            if not parts:
                return f"#### {title}\n(no results)\n"
            body = "\n\n".join(parts)
            return f"#### {title}\n{body}\n"

        raw_section = (
            "\n---\n### Crew outputs (raw)\n"
            + _block("Budget", budget_texts)
            + _block("Flights", flights_texts)
            + _block("Hotels",  hotels_texts)
            + _block("Activities", activities_texts)
        )
        return header + raw_section


# -------------------- factory (sync helper) --------------------
def build_trip_host_agent_sync() -> Agent:
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
