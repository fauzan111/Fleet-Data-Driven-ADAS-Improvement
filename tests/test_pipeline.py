"""Pipeline tests: generation, ETL flag physics, aggregation, hotspots."""
from __future__ import annotations

import numpy as np
import polars as pl

from fleet_adas.pipeline.etl import G


def test_raw_events_generated(cfg):
    raw = pl.read_parquet(str(cfg.raw_dir / "date=*" / "events.parquet"))
    assert raw.height > 1000
    # raw feed must NOT leak labels — only telematics + context
    assert "is_critical" not in raw.columns
    assert {"speed_mps", "long_accel_mps2", "yaw_rate_radps"} <= set(raw.columns)


def test_etl_flags_match_thresholds(cfg):
    ev = pl.read_parquet(cfg.curated_dir / "events.parquet")
    # every flagged hard-brake really exceeds the decel threshold
    hb = ev.filter(pl.col("is_hard_brake"))
    assert (hb["long_accel_mps2"] <= -cfg.hard_brake_g * G + 1e-6).all()
    sw = ev.filter(pl.col("is_swerve"))
    assert (sw["yaw_rate_radps"].abs() >= cfg.swerve_yawrate - 1e-6).all()
    assert ev["is_critical"].sum() > 0


def test_segment_rate_correlates_with_hidden_hazard(cfg):
    seg = pl.read_parquet(cfg.features_dir / "segment_features.parquet").filter(
        pl.col("n_samples") >= 20)
    corr = np.corrcoef(seg["critical_rate"].to_numpy(),
                       seg["hazard_level"].to_numpy())[0, 1]
    # the observed rate should carry real signal about the latent hazard
    assert corr > 0.5


def test_hotspots_detected(cfg):
    hot = pl.read_parquet(cfg.features_dir / "hotspots.parquet")
    assert hot.height >= 1
    assert (hot["n_events"] >= cfg.hotspot_min_events).all()
