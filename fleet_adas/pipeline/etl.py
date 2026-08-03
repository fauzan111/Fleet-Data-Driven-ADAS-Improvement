"""
ETL: raw telematics -> curated events.

Reads the date-partitioned raw zone, derives the safety-critical flags from the
sensor physics (the labels the raw feed intentionally omits), classifies each
event, and writes a cleaned, typed curated table. This is the boundary where
"sensor signals" become "analysable safety events".
"""
from __future__ import annotations

import polars as pl

from ..config import Settings, settings as default_settings

G = 9.81


def run_etl(cfg: Settings | None = None) -> pl.DataFrame:
    cfg = cfg or default_settings
    cfg.ensure_dirs()

    raw = pl.read_parquet(str(cfg.raw_dir / "date=*" / "events.parquet"))

    decel_g = (-pl.col("long_accel_mps2") / G)
    is_hard_brake = decel_g >= cfg.hard_brake_g
    is_swerve = pl.col("yaw_rate_radps").abs() >= cfg.swerve_yawrate
    is_hard_stop = is_hard_brake & (pl.col("speed_mps") < cfg.hard_stop_speed)

    curated = (
        raw
        # basic cleaning: drop physically impossible samples
        .filter((pl.col("speed_mps") >= 0) & (pl.col("speed_mps") < 80))
        .with_columns(
            decel_g.alias("decel_g"),
            is_hard_brake.alias("is_hard_brake"),
            is_swerve.alias("is_swerve"),
            is_hard_stop.alias("is_hard_stop"),
            (is_hard_brake | is_swerve).alias("is_critical"),
        )
        .with_columns(
            # one label per event; abrupt stop > swerve > hard brake > normal
            pl.when(pl.col("is_hard_stop")).then(pl.lit("hard_stop"))
            .when(pl.col("is_swerve")).then(pl.lit("swerve"))
            .when(pl.col("is_hard_brake")).then(pl.lit("hard_brake"))
            .otherwise(pl.lit("normal")).alias("event_type"),
            # severity in [0,1]: how far past the trigger threshold
            pl.max_horizontal(
                (pl.col("decel_g") / cfg.hard_brake_g - 1).clip(0, None),
                (pl.col("yaw_rate_radps").abs() / cfg.swerve_yawrate - 1).clip(0, None),
            ).clip(0, 1).alias("severity"),
        )
    )

    curated.write_parquet(cfg.curated_dir / "events.parquet")
    return curated
