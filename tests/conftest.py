"""Shared fixtures: build a small, self-contained data lake + trained model
once per test session in a temp dir, so tests never touch real artifacts."""
from __future__ import annotations

import pytest

from fleet_adas.config import Settings
from fleet_adas.generation import generate_raw_events
from fleet_adas.modeling.risk_model import score_all_segments, train_risk_model
from fleet_adas.pipeline import build_segment_features, detect_hotspots, run_etl
from fleet_adas.road_network import RoadNetwork


@pytest.fixture(scope="session")
def cfg(tmp_path_factory) -> Settings:
    d = tmp_path_factory.mktemp("lake")
    cfg = Settings(
        data_root=d / "data", model_root=d / "models",
        n_trips=1500, n_segments=80,
        # concentrate events so hotspots form in a small test run
        lat_min=45.20, lat_max=45.40, lon_min=8.95, lon_max=9.15,
    )
    cfg.ensure_dirs()
    net = RoadNetwork.generate(cfg)
    net.save(cfg)
    generate_raw_events(cfg, net)
    run_etl(cfg)
    build_segment_features(cfg)
    detect_hotspots(cfg)
    train_risk_model(cfg, verbose=False)
    score_all_segments(cfg)
    return cfg
