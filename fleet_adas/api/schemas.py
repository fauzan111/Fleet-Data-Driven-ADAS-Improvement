"""Pydantic request/response schemas for the ADAS risk API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    n_segments: int
    n_hotspots: int


class AdasParams(BaseModel):
    risk_score: float
    risk_category: str
    fcw_lead_time_s: float
    headway_time_s: float
    aeb_sensitivity: str
    lka_gain: float
    speed_advisory_kph: int | None
    rationale: list[str]


class RiskScoreResponse(BaseModel):
    lat: float
    lon: float
    matched_segment_id: int
    match_distance_m: float
    road_type: str
    speed_limit_kph: int
    curvature: float
    risk_score: float
    risk_category: str
    adas: AdasParams


class Hotspot(BaseModel):
    hotspot_id: int
    lat: float
    lon: float
    n_events: int
    dominant_type: str
    mean_severity: float
    hotspot_score: float


class HotspotsResponse(BaseModel):
    count: int
    hotspots: list[Hotspot]


class Waypoint(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class RouteRequest(BaseModel):
    waypoints: list[Waypoint] = Field(..., min_length=1)


class SegmentRisk(BaseModel):
    segment_id: int
    risk_score: float
    risk_category: str
    curvature: float
    speed_limit_kph: int


class RouteRecommendationResponse(BaseModel):
    n_segments: int
    mean_risk_score: float
    max_risk_score: float
    critical_segment_id: int
    adas_recommendation: AdasParams
    per_segment: list[SegmentRisk]
