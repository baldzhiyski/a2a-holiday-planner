from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class Flight(BaseModel):
    source: str
    dest: str
    depart_iso: str
    arrive_iso: str
    airline: str
    flight_no: Optional[str] = None
    duration_min: int
    price_eur: float
    cabin: Optional[str] = "Economy"
    link: Optional[str] = None
    source_site: Optional[str] = None

class Hotel(BaseModel):
    name: str
    address: str
    checkin_iso: str
    checkout_iso: str
    rating: float
    price_total_eur: float
    district: Optional[str] = None
    link: Optional[str] = None
    source_site: Optional[str] = None

class Activity(BaseModel):
    title: str
    date_iso: str
    start_local: str
    end_local: str
    price_eur: float
    category: str
    rating: float
    link: Optional[str] = None
    source_site: Optional[str] = None
    location_hint: Optional[str] = None

class BudgetPolicyDecision(BaseModel):
    flight_cap_eur: float
    hotel_cap_eur: float
    activities_cap_eur: float
    notes: Optional[str] = None

class ItineraryDay(BaseModel):
    date_iso: str
    morning: Optional[str] = None
    afternoon: Optional[str] = None
    evening: Optional[str] = None
    booked_activities: List[Activity] = Field(default_factory=list)

class CandidateItinerary(BaseModel):
    summary: str
    outbound: Flight
    inbound: Flight
    hotel: Hotel
    days: List[ItineraryDay]
    price_breakdown_eur: dict
    total_eur: float
    score: float
    hold_links: dict

class TripRequest(BaseModel):
    """LLM-structured parse of a user trip request."""
    origin: Optional[str] = Field(None, description="Departure city, e.g., Berlin")
    dest: Optional[str] = Field(None, description="Destination city, e.g., Lisbon")
    start_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="YYYY-MM-DD")
    passengers: int = 1
    budget_eur: Optional[float] = None
    # Free-form preferences; we commonly expect:
    #   walkable (bool), boutique (bool), no_redeye (bool),
    #   depart_after (HH:MM), return_after (HH:MM)
    prefs: Dict[str, Any] = Field(default_factory=dict)