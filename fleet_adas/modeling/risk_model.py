"""
Per-segment risk model.

A LightGBM gradient-boosting regressor predicts the safety-critical event rate
of a road segment from its road character and driving-context features. The raw
predicted rate is mapped to a 0–1 ``risk_score`` via train-set percentiles.

The model is validated not just on held-out rate prediction (R²/MAE) but on how
well the predicted risk ranks segments by the *hidden* ground-truth hazard — the
metric that actually matters for ADAS tuning.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from ..config import Settings, settings as default_settings
from .features import FEATURE_COLUMNS, TARGET, build_feature_frame, feature_vector

MODEL_FILE = "risk_model.txt"
META_FILE = "risk_model_meta.json"


def risk_category(score: float) -> str:
    return "high" if score >= 0.66 else "medium" if score >= 0.33 else "low"


@dataclass
class RiskModel:
    booster: lgb.Booster
    lo: float          # 5th-pct predicted rate  -> risk 0
    hi: float          # 95th-pct predicted rate -> risk 1

    def predict_rate(self, X) -> np.ndarray:
        return self.booster.predict(np.asarray(X, dtype=float))

    def score_from_rate(self, rate) -> np.ndarray:
        rate = np.asarray(rate, dtype=float)
        span = max(self.hi - self.lo, 1e-6)
        return np.clip((rate - self.lo) / span, 0.0, 1.0)

    def predict_score(self, X) -> np.ndarray:
        return self.score_from_rate(self.predict_rate(X))

    def predict_one(self, feature_row: dict) -> dict:
        x = np.array([feature_vector(feature_row)])
        rate = float(self.predict_rate(x)[0])
        score = float(self.score_from_rate([rate])[0])
        return {"predicted_critical_rate": rate, "risk_score": score,
                "risk_category": risk_category(score)}

    # ------------------------------------------------------------------- io
    def save(self, cfg: Settings | None = None) -> None:
        cfg = cfg or default_settings
        cfg.ensure_dirs()
        self.booster.save_model(str(cfg.model_root / MODEL_FILE))
        (cfg.model_root / META_FILE).write_text(json.dumps(
            {"lo": self.lo, "hi": self.hi, "features": FEATURE_COLUMNS}, indent=2))

    @classmethod
    def load(cls, cfg: Settings | None = None) -> "RiskModel":
        cfg = cfg or default_settings
        booster = lgb.Booster(model_file=str(cfg.model_root / MODEL_FILE))
        meta = json.loads((cfg.model_root / META_FILE).read_text())
        return cls(booster, meta["lo"], meta["hi"])


MIN_EXPOSURE = 20  # min samples per window to trust a segment's observed rate


def _temporal_rates(cfg: Settings):
    """Split matched events at their median time into a history window (features)
    and a future window (target), and compute per-segment critical rates for
    each. This is the realistic setup: forecast future risk from past events."""
    ev = pl.read_parquet(cfg.curated_dir / "events_matched.parquet")
    cutoff = ev["timestamp"].median()

    def seg_rate(frame, name):
        return (frame.group_by("segment_id").agg(
            (1000.0 * pl.col("is_critical").sum() / pl.len()).alias(name),
            pl.len().alias(name + "_n")))

    hist = seg_rate(ev.filter(pl.col("timestamp") < cutoff), "hist_critical_rate")
    fut = seg_rate(ev.filter(pl.col("timestamp") >= cutoff), "future_critical_rate")
    return hist.join(fut, on="segment_id", how="inner")


def train_risk_model(cfg: Settings | None = None, verbose: bool = True):
    cfg = cfg or default_settings
    seg = pl.read_parquet(cfg.features_dir / "segment_features.parquet")
    rates = _temporal_rates(cfg)

    joined = (seg.join(rates, on="segment_id", how="inner").filter(
        (pl.col("hist_critical_rate_n") >= MIN_EXPOSURE)
        & (pl.col("future_critical_rate_n") >= MIN_EXPOSURE))
        # stable row order so the train/test split is reproducible (polars
        # joins do not preserve order)
        .sort("segment_id"))
    df = build_feature_frame(joined)

    X = df[FEATURE_COLUMNS]
    y = df["future_critical_rate"]          # predict the *future* rate
    hazard = df["hazard_level"]             # hidden ground truth, validation only

    X_tr, X_te, y_tr, y_te, hz_tr, hz_te = train_test_split(
        X, y, hazard, test_size=0.25, random_state=cfg.seed)

    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=15,
        random_state=cfg.seed, verbosity=-1,
        # fully reproducible fits (LightGBM is otherwise thread-nondeterministic)
        deterministic=True, force_row_wise=True, num_threads=1)
    model.fit(X_tr, y_tr)
    booster = model.booster_

    pred_te = booster.predict(X_te)
    lo, hi = np.percentile(booster.predict(X_tr), [5, 95])
    risk = RiskModel(booster, float(lo), float(hi))

    metrics = {
        "r2": float(r2_score(y_te, pred_te)),
        "mae": float(mean_absolute_error(y_te, pred_te)),
        # the metric that matters: does predicted risk rank the hidden hazard?
        "spearman_risk_vs_hazard": float(
            pd.Series(risk.score_from_rate(pred_te)).corr(
                pd.Series(hz_te.values), method="spearman")),
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
    }
    risk.save(cfg)
    if verbose:
        print(f"risk model  R²={metrics['r2']:.3f}  MAE={metrics['mae']:.3f}  "
              f"Spearman(risk,hazard)={metrics['spearman_risk_vs_hazard']:.3f}  "
              f"(train={metrics['n_train']}, test={metrics['n_test']})")
    return risk, metrics


def score_all_segments(cfg: Settings | None = None) -> pl.DataFrame:
    """Attach risk_score to every segment and persist for fast serving."""
    cfg = cfg or default_settings
    risk = RiskModel.load(cfg)
    seg = pl.read_parquet(cfg.features_dir / "segment_features.parquet")
    df = build_feature_frame(seg)
    # at serving time, the historical rate = the full observed rate so far
    df["hist_critical_rate"] = df["critical_rate"]
    scores = risk.predict_score(df[FEATURE_COLUMNS])
    out = seg.with_columns(pl.Series("risk_score", scores)).with_columns(
        pl.col("risk_score").map_elements(risk_category, return_dtype=pl.String)
        .alias("risk_category"))
    out.write_parquet(cfg.features_dir / "segment_risk.parquet")
    return out
