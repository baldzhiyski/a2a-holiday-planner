from pydantic import BaseModel
from typing import List, Optional, Dict

class TripRequest(BaseModel):
    origin: Optional[str] = None
    dest: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    passengers: Optional[int] = 2
    budget_eur: Optional[float] = None
    prefs: Dict[str, object] = {}

class Flight(BaseModel):
    source: str
    dest: str
    depart_iso: str
    arrive_iso: str
    price_eur: float
    airline: str
    link: Optional[str] = None

class Hotel(BaseModel):
    name: str
    rating: float
    checkin_iso: str
    checkout_iso: str
    price_total_eur: float
    link: Optional[str] = None

class Activity(BaseModel):
    title: str
    date_iso: str
    price_eur: float

class BudgetPolicyDecision(BaseModel):
    flights_cap_eur: float = 0
    hotel_cap_eur: float = 0
    activities_cap_eur: float = 0

class DayPlan(BaseModel):
    date_iso: str
    booked_activities: List[Activity] = []

class CandidateItinerary(BaseModel):
    summary: str
    outbound: Flight
    inbound: Flight
    hotel: Hotel
    days: List[DayPlan] = []
    price_breakdown_eur: Dict[str, float] = {}
    total_eur: float
    score: float
    hold_links: Dict[str, str] = {}
