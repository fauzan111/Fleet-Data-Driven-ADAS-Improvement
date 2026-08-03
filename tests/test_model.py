"""Risk-model and ADAS-recommender tests."""
from __future__ import annotations

import polars as pl

from fleet_adas.modeling.recommender import recommend_adas, recommend_for_route
from fleet_adas.modeling.risk_model import RiskModel, risk_category


def test_model_ranks_hazard(cfg):
    _ = cfg  # ensures artifacts built
    seg = pl.read_parquet(cfg.features_dir / "segment_risk.parquet").filter(
        pl.col("n_samples") >= 20)
    # risk_score should rank the hidden hazard well above chance
    sp = seg.select(pl.corr("risk_score", "hazard_level", method="spearman")).item()
    assert sp > 0.5


def test_risk_score_bounds(cfg):
    model = RiskModel.load(cfg)
    out = model.predict_one({"road_type": "urban", "speed_limit_kph": 50,
                             "curvature": 0.5, "length_m": 400,
                             "avg_speed_mps": 12, "speed_std_mps": 2,
                             "night_fraction": 0.3, "rain_fraction": 0.2,
                             "fog_fraction": 0.0, "traversals": 40,
                             "critical_rate": 40.0})
    assert 0.0 <= out["risk_score"] <= 1.0
    assert out["risk_category"] in {"low", "medium", "high"}


def test_recommender_monotonic():
    """Higher risk => earlier warning, bigger gap, never weaker AEB."""
    low = recommend_adas(0.1, curvature=0.2, speed_limit_kph=50)
    high = recommend_adas(0.9, curvature=0.2, speed_limit_kph=50)
    assert high.fcw_lead_time_s > low.fcw_lead_time_s
    assert high.headway_time_s > low.headway_time_s
    assert high.lka_gain >= low.lka_gain
    order = {"standard": 0, "elevated": 1, "high": 2}
    assert order[high.aeb_sensitivity] >= order[low.aeb_sensitivity]
    # a speed advisory only appears on genuinely risky segments
    assert low.speed_advisory_kph is None
    assert high.speed_advisory_kph is not None


def test_risk_category_thresholds():
    assert risk_category(0.1) == "low"
    assert risk_category(0.5) == "medium"
    assert risk_category(0.9) == "high"


def test_route_recommendation_uses_worst_segment():
    segs = [
        {"segment_id": 1, "risk_score": 0.2, "curvature": 0.1, "speed_limit_kph": 90},
        {"segment_id": 2, "risk_score": 0.85, "curvature": 0.5, "speed_limit_kph": 50},
    ]
    out = recommend_for_route(segs)
    assert out["critical_segment_id"] == 2
    assert out["max_risk_score"] == 0.85
    assert out["adas_recommendation"]["aeb_sensitivity"] == "high"
