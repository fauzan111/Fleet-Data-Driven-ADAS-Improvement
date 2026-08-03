"""
FastAPI service exposing fleet-derived ADAS risk intelligence.

Endpoints
---------
GET  /health                      liveness + artifact status
GET  /risk-score?lat=&lon=        risk score + ADAS tuning for a location
GET  /hotspots?limit=             ranked hazard hotspots
GET  /segments/{id}               risk for a specific road segment
POST /recommendations             route-level ADAS recommendation
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

from .. import __version__
from .schemas import (HealthResponse, HotspotsResponse, RiskScoreResponse,
                      RouteRecommendationResponse, RouteRequest)
from .service import get_service

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(
    title="Fleet-Data-Driven ADAS Risk API",
    version=__version__,
    description="Turns anonymised fleet driving events into per-road-segment "
                "risk scores and concrete ADAS parameter recommendations.",
)


@app.get("/", include_in_schema=False)
def root():
    """Send visitors of the bare URL straight to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health():
    svc = get_service()
    return HealthResponse(
        status="ok",
        model_loaded=(svc.cfg.model_root / "risk_model.txt").exists(),
        n_segments=len(svc._seg_by_id),
        n_hotspots=svc.hotspots.height,
    )


@app.get("/risk-score", response_model=RiskScoreResponse)
def risk_score(lat: float = Query(..., ge=-90, le=90),
               lon: float = Query(..., ge=-180, le=180)):
    return get_service().risk_at(lat, lon)


@app.get("/hotspots", response_model=HotspotsResponse)
def hotspots(limit: int = Query(20, ge=1, le=500)):
    items = get_service().top_hotspots(limit)
    return HotspotsResponse(count=len(items), hotspots=items)


@app.get("/segments/{segment_id}")
def segment(segment_id: int):
    row = get_service().segment_risk(segment_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"segment {segment_id} not found")
    return row


@app.post("/recommendations", response_model=RouteRecommendationResponse)
def recommendations(req: RouteRequest):
    waypoints = [(w.lat, w.lon) for w in req.waypoints]
    return get_service().route_recommendation(waypoints)
