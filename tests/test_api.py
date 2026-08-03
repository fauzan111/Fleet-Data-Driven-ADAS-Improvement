"""API tests — drive every endpoint through the ASGI app with test artifacts."""
from __future__ import annotations

import polars as pl
import pytest
from starlette.testclient import TestClient

from fleet_adas.api import service as service_mod
from fleet_adas.api.service import FleetService


@pytest.fixture(scope="module")
def client(cfg):
    # point the singleton at the test lake, then drive the real app
    service_mod._service = FleetService(cfg)
    from fleet_adas.api.main import app
    return TestClient(app)


@pytest.fixture(scope="module")
def sample_point(cfg):
    seg = pl.read_parquet(cfg.features_dir / "segment_risk.parquet").sort(
        "risk_score", descending=True).row(0, named=True)
    return (seg["a_lat"] + seg["b_lat"]) / 2, (seg["a_lon"] + seg["b_lon"]) / 2


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_segments"] > 0


def test_risk_score(client, sample_point):
    lat, lon = sample_point
    r = client.get("/risk-score", params={"lat": lat, "lon": lon})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["adas"]["fcw_lead_time_s"] >= 2.0
    assert "rationale" in body["adas"]


def test_risk_score_validates_bounds(client):
    assert client.get("/risk-score", params={"lat": 200, "lon": 0}).status_code == 422


def test_hotspots(client):
    r = client.get("/hotspots", params={"limit": 5})
    assert r.status_code == 200
    assert r.json()["count"] >= 1


def test_segment_404(client):
    assert client.get("/segments/10000000").status_code == 404


def test_recommendations(client, sample_point):
    lat, lon = sample_point
    r = client.post("/recommendations",
                    json={"waypoints": [{"lat": lat, "lon": lon},
                                        {"lat": lat + 0.01, "lon": lon + 0.01}]})
    assert r.status_code == 200
    body = r.json()
    assert body["n_segments"] >= 1
    assert 0.0 <= body["max_risk_score"] <= 1.0
    assert body["adas_recommendation"]["aeb_sensitivity"] in {
        "standard", "elevated", "high"}
