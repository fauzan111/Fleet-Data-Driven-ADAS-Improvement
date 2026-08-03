"""
Map-match curated events to road segments and aggregate per-segment features.

Each event is snapped to the nearest road segment (the pipeline never sees the
generator's segment assignment — it re-derives it, like a real map-matcher).
Segment-level aggregates become the feature table the risk model learns from.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ..config import Settings, settings as default_settings
from ..road_network import RoadNetwork

# Events snapped further than this from any segment are treated as off-network.
MAX_MATCH_DIST_M = 200.0


def build_segment_features(cfg: Settings | None = None,
                           net: RoadNetwork | None = None) -> pl.DataFrame:
    cfg = cfg or default_settings
    net = net or RoadNetwork.load(cfg)

    events = pl.read_parquet(cfg.curated_dir / "events.parquet")
    seg_id, dist, _ = net.match(events["lat"].to_numpy(), events["lon"].to_numpy())
    events = events.with_columns(
        pl.Series("segment_id", seg_id),
        pl.Series("match_dist_m", dist),
    ).filter(pl.col("match_dist_m") <= MAX_MATCH_DIST_M)

    # persist map-matched events so the model can build a temporal train/test
    # split (predict future risk from historical events) without re-matching
    events.write_parquet(cfg.curated_dir / "events_matched.parquet")

    agg = (
        events.group_by("segment_id")
        .agg(
            pl.len().alias("n_samples"),
            pl.col("trip_id").n_unique().alias("traversals"),
            pl.col("is_critical").sum().alias("n_critical"),
            pl.col("is_hard_brake").sum().alias("n_hard_brake"),
            pl.col("is_swerve").sum().alias("n_swerve"),
            pl.col("is_hard_stop").sum().alias("n_hard_stop"),
            pl.col("speed_mps").mean().alias("avg_speed_mps"),
            pl.col("speed_mps").std().fill_null(0).alias("speed_std_mps"),
            pl.col("severity").mean().alias("mean_severity"),
            pl.col("is_night").mean().alias("night_fraction"),
            (pl.col("weather") == "rain").mean().alias("rain_fraction"),
            (pl.col("weather") == "fog").mean().alias("fog_fraction"),
        )
        .with_columns(
            # Primary target: safety-critical events per 1,000 telematics samples
            # (≈ per unit distance, since samples are ~evenly spaced). This is a
            # stable exposure-normalised rate — unlike per-trip counts, which are
            # dominated by noise on lightly-driven segments.
            (1000.0 * pl.col("n_critical") / pl.col("n_samples"))
            .alias("critical_rate"),
            (100.0 * pl.col("n_critical") / pl.col("traversals"))
            .alias("critical_per_100_traversals"),
        )
    )

    # join the static road attributes (and the hidden ground-truth hazard, kept
    # only for offline validation of the model — never used as a feature)
    static = net.frame.select([
        "segment_id", "road_type", "speed_limit_kph", "curvature", "length_m",
        "a_lat", "a_lon", "b_lat", "b_lon", "hazard_level", "is_planted_hazard",
    ])
    features = static.join(agg, on="segment_id", how="left").with_columns(
        pl.col(["n_samples", "traversals", "n_critical", "n_hard_brake",
                "n_swerve", "n_hard_stop"]).fill_null(0),
        pl.col(["avg_speed_mps", "speed_std_mps", "mean_severity",
                "night_fraction", "rain_fraction", "fog_fraction",
                "critical_rate", "critical_per_100_traversals"]).fill_null(0.0),
    )

    features.write_parquet(cfg.features_dir / "segment_features.parquet")
    return features
