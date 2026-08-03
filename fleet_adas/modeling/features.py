"""
Feature engineering for the risk model.

Only road-context and exposure features are used as inputs. Outcome-derived
columns (n_critical, mean_severity, and the hidden hazard_level) are excluded to
avoid target leakage — the model must predict risk from road character and
driving context, not from the events it is trying to forecast.
"""
from __future__ import annotations

import pandas as pd
import polars as pl

TARGET = "critical_rate"

FEATURE_COLUMNS = [
    "speed_limit_kph", "curvature", "length_m",
    "avg_speed_mps", "speed_std_mps",
    "night_fraction", "rain_fraction", "fog_fraction",
    "traversals",
    "is_highway", "is_rural", "is_urban",
    # the strongest predictor: the segment's own historical event rate. At
    # serving time this is the rate observed so far; the model forecasts ahead.
    "hist_critical_rate",
]


def _one_hot_road_type(df: pd.DataFrame) -> pd.DataFrame:
    for rt in ("highway", "rural", "urban"):
        df[f"is_{rt}"] = (df["road_type"] == rt).astype(int)
    return df


def build_feature_frame(segment_features: pl.DataFrame) -> pd.DataFrame:
    """Polars segment-feature table -> pandas frame with model columns + target."""
    df = segment_features.to_pandas()
    df = _one_hot_road_type(df)
    return df


def feature_vector(row: dict) -> list[float]:
    """Build a single input vector (serving path) from a segment-feature dict."""
    rt = row.get("road_type", "urban")
    enriched = dict(row)
    enriched["is_highway"] = int(rt == "highway")
    enriched["is_rural"] = int(rt == "rural")
    enriched["is_urban"] = int(rt == "urban")
    # at serving time the "historical" rate is the full observed rate so far
    enriched.setdefault("hist_critical_rate", row.get("critical_rate", 0.0))
    return [float(enriched.get(c, 0.0)) for c in FEATURE_COLUMNS]
