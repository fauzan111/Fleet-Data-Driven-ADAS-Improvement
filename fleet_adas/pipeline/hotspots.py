"""
Hazard hotspot detection.

Spatially clusters safety-critical events with DBSCAN (density-based, so it needs
no preset cluster count and ignores isolated one-off events as noise). Each
cluster is a hazard hotspot — a place where many drivers brake hard or swerve —
independent of the road-segment binning.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.cluster import DBSCAN

from ..config import Settings, settings as default_settings
from ..geo import to_local_xy


def detect_hotspots(cfg: Settings | None = None) -> pl.DataFrame:
    cfg = cfg or default_settings

    events = pl.read_parquet(cfg.curated_dir / "events.parquet").filter(
        pl.col("is_critical"))
    if events.height == 0:
        empty = pl.DataFrame(schema={"hotspot_id": pl.Int64, "lat": pl.Float64,
                                     "lon": pl.Float64, "n_events": pl.Int64})
        empty.write_parquet(cfg.features_dir / "hotspots.parquet")
        return empty

    lat0 = (cfg.lat_min + cfg.lat_max) / 2
    lon0 = (cfg.lon_min + cfg.lon_max) / 2
    x, y = to_local_xy(events["lat"].to_numpy(), events["lon"].to_numpy(),
                       lat0, lon0)
    xy = np.column_stack((x, y))

    labels = DBSCAN(eps=cfg.hotspot_eps_m,
                    min_samples=cfg.hotspot_min_events).fit_predict(xy)
    events = events.with_columns(pl.Series("cluster", labels))

    clusters = (
        events.filter(pl.col("cluster") >= 0)
        .group_by("cluster")
        .agg(
            pl.len().alias("n_events"),
            pl.col("lat").mean().alias("lat"),
            pl.col("lon").mean().alias("lon"),
            pl.col("is_hard_brake").sum().alias("n_hard_brake"),
            pl.col("is_swerve").sum().alias("n_swerve"),
            pl.col("is_hard_stop").sum().alias("n_hard_stop"),
            pl.col("severity").mean().alias("mean_severity"),
        )
        .with_columns(
            # dominant manoeuvre + a simple severity-weighted hotspot score
            pl.when((pl.col("n_swerve") >= pl.col("n_hard_brake")))
            .then(pl.lit("swerve")).otherwise(pl.lit("hard_brake"))
            .alias("dominant_type"),
            (pl.col("n_events") * (1 + pl.col("mean_severity")))
            .alias("hotspot_score"),
        )
        # DBSCAN can attach small satellite clusters; a reportable hotspot must
        # clear the minimum-events bar.
        .filter(pl.col("n_events") >= cfg.hotspot_min_events)
        .sort("hotspot_score", descending=True)
        .with_row_index("hotspot_id")
    )

    clusters.write_parquet(cfg.features_dir / "hotspots.parquet")
    return clusters
