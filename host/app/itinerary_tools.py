from __future__ import annotations
from datetime import datetime, timedelta
from typing import List
from host.app.schemas import Flight, Hotel, Activity, ItineraryDay, CandidateItinerary

def align_windows(outbound: Flight, inbound: Flight, hotel: Hotel) -> bool:
    arr = _dt(outbound.arrive_iso)
    chk_in = _dt(hotel.checkin_iso)
    chk_out = _dt(hotel.checkout_iso)
    dep = _dt(inbound.depart_iso)
    return arr.date() <= chk_in.date() and dep.date() >= chk_out.date()

def choose_activities(acts: List[Activity], start_date: str, end_date: str, per_day_budget: float) -> List[ItineraryDay]:
    s=_d(start_date); e=_d(end_date)
    by_date={}
    for a in acts:
        by_date.setdefault(a.date_iso, []).append(a)
    days=[]
    cur=s
    while cur<=e:
        key=cur.strftime("%Y-%m-%d")
        cand=sorted(by_date.get(key, []), key=lambda x:(-x.rating, x.price_eur))
        chosen=[]; spent=0.0; slots={"morning":None,"afternoon":None,"evening":None}
        for act in cand:
            if spent+act.price_eur>per_day_budget: continue
            h=int(act.start_local.split(":")[0])
            bucket="morning" if h<12 else "afternoon" if h<18 else "evening"
            if slots[bucket] is not None: continue
            slots[bucket]=act.title
            chosen.append(act); spent+=act.price_eur
            if len(chosen)>=2: break
        days.append(ItineraryDay(date_iso=key, morning=slots["morning"], afternoon=slots["afternoon"], evening=slots["evening"], booked_activities=chosen))
        cur+=timedelta(days=1)
    return days

def score_itinerary(total: float, prefs: dict, hotel: Hotel, activities: List[Activity]) -> float:
    score=0.0
    score += max(0, 10000/(total+1))
    score += hotel.rating * 50
    if activities:
        score += sum(a.rating for a in activities)/len(activities) * 25
    if prefs.get("walkable"): score += 100
    if prefs.get("boutique") and hotel.rating>=4.0: score += 50
    return score

def _dt(s:str)->datetime: return datetime.fromisoformat(s)
def _d(s:str):
    return datetime.fromisoformat(s).date() if "T" in s else datetime.strptime(s,"%Y-%m-%d").date()
