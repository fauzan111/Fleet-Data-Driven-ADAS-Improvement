"""
Pipeline orchestration CLI.

Runs the full offline flow end-to-end:

    generate -> etl -> segment features -> hotspots -> train -> score

Each stage reads/writes the data lake, so stages can also be run individually.

Usage:
    python -m fleet_adas.cli all          # full pipeline (default)
    python -m fleet_adas.cli generate
    python -m fleet_adas.cli train
"""
from __future__ import annotations

import argparse
import time

from .config import settings
from .generation import generate_raw_events
from .modeling.risk_model import train_risk_model, score_all_segments
from .pipeline import build_segment_features, detect_hotspots, run_etl
from .road_network import RoadNetwork


def _timed(label, fn):
    t = time.time()
    out = fn()
    print(f"  [{label}] {time.time() - t:.1f}s")
    return out


def cmd_generate():
    net = RoadNetwork.generate(settings)
    net.save(settings)
    df = _timed("generate", lambda: generate_raw_events(settings, net))
    print(f"  raw events: {df.height:,}")


def cmd_etl():
    df = _timed("etl", lambda: run_etl(settings))
    print(f"  curated events: {df.height:,}  (critical: {int(df['is_critical'].sum()):,})")


def cmd_features():
    df = _timed("segment features", lambda: build_segment_features(settings))
    print(f"  segments: {df.height}")


def cmd_hotspots():
    df = _timed("hotspots", lambda: detect_hotspots(settings))
    print(f"  hotspots: {df.height}")


def cmd_train():
    _, metrics = _timed("train", lambda: train_risk_model(settings))
    _timed("score", lambda: score_all_segments(settings))
    return metrics


def cmd_all():
    settings.ensure_dirs()
    print("Running full ADAS risk pipeline...")
    cmd_generate()
    cmd_etl()
    cmd_features()
    cmd_hotspots()
    cmd_train()
    print("Done. Artifacts in", settings.data_root, "and", settings.model_root)


COMMANDS = {
    "all": cmd_all, "generate": cmd_generate, "etl": cmd_etl,
    "features": cmd_features, "hotspots": cmd_hotspots, "train": cmd_train,
}


def main():
    ap = argparse.ArgumentParser(description="Fleet ADAS pipeline")
    ap.add_argument("command", nargs="?", default="all", choices=list(COMMANDS))
    args = ap.parse_args()
    COMMANDS[args.command]()


if __name__ == "__main__":
    main()
