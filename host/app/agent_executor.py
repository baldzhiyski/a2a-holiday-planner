import json, re
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TextPart, TaskState, UnsupportedOperationError
from a2a.utils.errors import ServerError
from host.app.host_graph import TripPlannerHost

class HostAgentExecutor(AgentExecutor):
    def __init__(self, host: TripPlannerHost):
        self.host = host

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.task_id or not context.context_id or not context.message:
            raise ValueError("Missing")
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if not context.current_task: await updater.submit()
        await updater.start_work()

        text = context.get_user_input().strip()

        # Booking branch (keep regex here—it's explicit and robust)
        m = re.match(r"(?i)book option (\d+)", text)
        if m:
            idx = int(m.group(1))
            result = await self.host.book_itinerary.invoke(option_index=idx, context_id=context.context_id)
            await updater.add_artifact([Part(root=TextPart(text=result))], name="booking_confirmation")
            await updater.complete()
            return

        # NEW: LLM-based structured extraction
        await updater.update_status(
            TaskState.working,
            message=updater.new_agent_message([Part(root=TextPart(text="Parsing your request..."))])
        )
        parsed_json = await self.host.parse_user_query.invoke(text=text)
        try:
            req = json.loads(parsed_json) if parsed_json else {}
        except Exception:
            req = {}

        # Defaults / normalization
        origin_city = (req.get("origin") or "Berlin").strip()
        dest_city = (req.get("dest") or "Lisbon").strip()
        start_date = req.get("start_date") or "2025-11-10"
        end_date   = req.get("end_date") or "2025-11-14"
        passengers = int(req.get("passengers") or 2)
        budget_eur = float(req.get("budget_eur") or 2500.0)

        raw_prefs = req.get("prefs") or {}
        prefs = {
            "origin": origin_city,
            "dest": dest_city,
            "budget_eur": budget_eur,
            "passengers": passengers,
            "walkable": bool(raw_prefs.get("walkable", "walkable" in text.lower())),
            "boutique": bool(raw_prefs.get("boutique", "boutique" in text.lower())),
            "no_redeye": bool(raw_prefs.get("no_redeye", ("avoid redeye" in text.lower() or "avoid redeyes" in text.lower()))),
            "depart_after": raw_prefs.get("depart_after", "09:00"),
            "return_after": raw_prefs.get("return_after", "14:00"),
        }

        # ---- Fan out to micro-agents ----
        await updater.update_status(TaskState.working, message=updater.new_agent_message([Part(root=TextPart(text="Collecting budget policy..."))]))
        budget_json = await self.host.send_message.invoke(
            agent_name="BudgetPolicy",
            task=json.dumps({"total_budget_eur": int(budget_eur), "passengers": passengers}),
            thread_id=context.context_id
        )

        await updater.update_status(TaskState.working, message=updater.new_agent_message([Part(root=TextPart(text="Fetching flights..."))]))
        flights_json = await self.host.send_message.invoke(
            agent_name="FlightsScraper",
            task=json.dumps({
                "origin": origin_city, "dest": dest_city,
                "start_date": start_date, "end_date": end_date,
                "passengers": passengers, "no_redeye": prefs["no_redeye"],
                "depart_after": prefs["depart_after"], "return_after": prefs["return_after"]
            }),
            thread_id=context.context_id
        )

        await updater.update_status(TaskState.working, message=updater.new_agent_message([Part(root=TextPart(text="Fetching hotels..."))]))
        hotels_json = await self.host.send_message.invoke(
            agent_name="HotelsScraper",
            task=json.dumps({
                "city": dest_city, "checkin": start_date, "checkout": end_date,
                "min_rating": 4.0 if prefs["boutique"] else 0.0, "style": "boutique" if prefs["boutique"] else "any",
                "walkable": prefs["walkable"], "max_total_eur": budget_eur * 0.6
            }),
            thread_id=context.context_id
        )

        await updater.update_status(TaskState.working, message=updater.new_agent_message([Part(root=TextPart(text="Fetching activities..."))]))
        activities_json = await self.host.send_message.invoke(
            agent_name="ActivitiesScraper",
            task=json.dumps({
                "city": dest_city, "date_from": start_date, "date_to": end_date,
                "categories": ["food tour","walking tour","museum","viewpoint"],
                "min_rating": 4.5, "max_price_eur": 140
            }),
            thread_id=context.context_id
        )

        # ---- Compose itineraries ----
        await updater.update_status(TaskState.working, message=updater.new_agent_message([Part(root=TextPart(text="Composing itineraries..."))]))
        candidates = await self.host.compose_itineraries.invoke(
            flights_json=flights_json, hotels_json=hotels_json, activities_json=activities_json,
            budget_json=budget_json, start_date=start_date, end_date=end_date, prefs_json=json.dumps(prefs),
            context_id=context.context_id
        )
        parsed = json.loads(candidates) if candidates else {"status":"error"}
        if parsed.get("status") != "ok" or not parsed.get("candidates"):
            await updater.update_status(
                TaskState.input_required,
                message=updater.new_agent_message([Part(root=TextPart(text="No viable options. Adjust dates/budget/prefs and try again."))])
            )
            return

        lines = ["Here are your top itineraries:"]
        for i, c in enumerate(parsed["candidates"], start=1):
            bd = c["price_breakdown_eur"]
            lines.append(
                f"{i}) {c['summary']} — Total €{c['total_eur']:.0f} "
                f"(Flights €{bd['outbound']+bd['inbound']:.0f}, Hotel €{bd['hotel']:.0f}, Activities €{bd['activities']:.0f})"
            )
        lines.append("\nSay: `Book option N` to proceed.")

        await updater.add_artifact([Part(root=TextPart(text=json.dumps(parsed['candidates'], ensure_ascii=False)))], name="candidate_itineraries_json")
        await updater.update_status(TaskState.working, message=updater.new_agent_message([Part(root=TextPart(text="\n".join(lines)))]))
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
