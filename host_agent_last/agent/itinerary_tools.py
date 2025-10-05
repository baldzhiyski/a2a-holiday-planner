from datetime import datetime
from typing import List

from host_agent_last.agent.schemas import Flight, Hotel, DayPlan, Activity


def align_windows(out: Flight, inn: Flight, hotel: Hotel) -> bool:
    try:
        o_arr = datetime.fromisoformat(out.arrive_iso.replace("Z","+00:00"))
        i_dep = datetime.fromisoformat(inn.depart_iso.replace("Z","+00:00"))
        h_in  = datetime.fromisoformat(hotel.checkin_iso.replace("Z","+00:00"))
        h_out = datetime.fromisoformat(hotel.checkout_iso.replace("Z","+00:00"))
        return o_arr <= h_in <= h_out <= i_dep
    except Exception:
        return True  # be lenient if scrapers don’t provide perfect ISO

def choose_activities(activities: List[Activity], start_date: str, end_date: str, per_day_budget: float) -> List[DayPlan]:
    # naive slice/budget fill
    by_day: dict[str, list[Activity]] = {}
    for a in activities:
        by_day.setdefault(a.date_iso[:10], []).append(a)
    days = []
    for day_iso, items in sorted(by_day.items()):
        total = 0.0
        picked = []
        for it in items:
            if total + it.price_eur <= per_day_budget:
                picked.append(it); total += it.price_eur
        if picked:
            days.append(DayPlan(date_iso=day_iso, booked_activities=picked))
    return days

def score_itinerary(total: float, prefs: dict, hotel: Hotel, acts: List[Activity]) -> float:
    base = 1000.0 / max(total, 1.0)
    bonus = 0.0
    if prefs.get("boutique") and hotel.rating >= 4.0:
        bonus += 50
    if prefs.get("walkable"):
        bonus += 20
    bonus += min(len(acts)*5, 40)
    return base + bonus
