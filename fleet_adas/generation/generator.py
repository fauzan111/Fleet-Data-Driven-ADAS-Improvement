"""
Synthetic fleet telematics generator.

Simulates anonymised trips over the reference road network and emits raw
telematics samples (timestamp, GPS, speed, longitudinal/lateral accel, yaw rate,
steering, environment tags). Hazardous segments, rain, and night driving raise
the rate of safety-critical *physics* (hard braking, evasive swerves, abrupt
stops) — but the raw feed carries **no labels**; the ETL derives them, exactly
as a real pipeline would from onboard-sensor signals.

Output is written to the raw zone partitioned by date (`raw/date=YYYY-MM-DD/`),
mirroring an S3 data lake.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from ..config import Settings, settings as default_settings
from ..geo import local_meters_per_degree
from ..road_network import RoadNetwork

G = 9.81
WEATHER = np.array(["clear", "rain", "fog"])
WEATHER_P = np.array([0.72, 0.22, 0.06])


def _neighbor_table(net: RoadNetwork, k: int = 6):
    """k nearest segments to each segment (for building coherent routes)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(net._mid)
    _, idx = tree.query(net._mid, k=k + 1)
    return idx[:, 1:]  # drop self


def generate_raw_events(cfg: Settings | None = None,
                        net: RoadNetwork | None = None) -> pl.DataFrame:
    cfg = cfg or default_settings
    cfg.ensure_dirs()
    rng = np.random.default_rng(cfg.seed + 1)
    if net is None:
        net = RoadNetwork.generate(cfg)
        net.save(cfg)

    seg = net.frame
    seg_hazard = seg["hazard_level"].to_numpy()
    seg_curv = seg["curvature"].to_numpy()
    seg_speed = seg["speed_limit_kph"].to_numpy() / 3.6  # -> m/s
    seg_len = seg["length_m"].to_numpy()
    a_lat, a_lon = seg["a_lat"].to_numpy(), seg["a_lon"].to_numpy()
    b_lat, b_lon = seg["b_lat"].to_numpy(), seg["b_lon"].to_numpy()
    neighbors = _neighbor_table(net)
    n_seg = len(seg)

    start_day = dt.datetime(2025, 6, 1)
    cols: dict[str, list] = {c: [] for c in (
        "event_id", "trip_id", "vehicle_id", "timestamp", "lat", "lon",
        "speed_mps", "long_accel_mps2", "lat_accel_mps2", "yaw_rate_radps",
        "steering_angle_rad", "heading_rad", "weather", "is_night")}
    eid = 0

    for trip in range(cfg.n_trips):
        vehicle = int(rng.integers(0, cfg.n_trips // 4 + 1))
        weather = rng.choice(WEATHER, p=WEATHER_P)
        is_night = bool(rng.random() < 0.28)
        t = (start_day + dt.timedelta(
            days=float(rng.uniform(0, 30)),
            seconds=float(rng.uniform(0, 86400))))

        w_mult = {"clear": 1.0, "rain": 1.6, "fog": 1.9}[str(weather)]
        n_mult = 1.4 if is_night else 1.0

        # build a route as a short random walk over spatial neighbours
        s = int(rng.integers(0, n_seg))
        route_len = int(rng.integers(2, 7))
        route = [s]
        for _ in range(route_len - 1):
            s = int(rng.choice(neighbors[s]))
            route.append(s)

        for sid in route:
            n_samples = max(2, int(seg_len[sid] / 80))  # ~a sample every 80 m
            ts = np.linspace(0.05, 0.95, n_samples)
            lat = a_lat[sid] + ts * (b_lat[sid] - a_lat[sid])
            lon = a_lon[sid] + ts * (b_lon[sid] - a_lon[sid])
            heading = np.arctan2(b_lat[sid] - a_lat[sid], b_lon[sid] - a_lon[sid])

            base_speed = seg_speed[sid] * (1 - 0.35 * seg_curv[sid])
            speed = np.clip(base_speed + rng.normal(0, 1.2, n_samples), 1.0, None)
            long_acc = rng.normal(0, 0.5, n_samples)
            yaw = rng.normal(0, 0.04 * (1 + 3 * seg_curv[sid]), n_samples)
            steer = rng.normal(0, 0.03, n_samples) + 0.25 * seg_curv[sid]
            lat_acc = speed * yaw

            # per-sample probability of a safety-critical manoeuvre
            p_evt = np.clip(0.02 * (0.3 + 2.6 * seg_hazard[sid]) * w_mult * n_mult,
                            0.0, 0.85)
            fires = rng.random(n_samples) < p_evt
            for j in np.where(fires)[0]:
                kind = rng.choice(["brake", "swerve", "stop"], p=[0.55, 0.3, 0.15])
                if kind in ("brake", "stop"):
                    long_acc[j] = -rng.uniform(0.42, 0.85) * G
                    if kind == "stop":
                        speed[j] = rng.uniform(0.0, cfg.hard_stop_speed)
                else:  # swerve
                    yaw[j] = rng.choice([-1, 1]) * rng.uniform(0.36, 0.75)
                    steer[j] = np.sign(yaw[j]) * rng.uniform(0.2, 0.5)
                    lat_acc[j] = speed[j] * yaw[j]

            dt_s = np.maximum(seg_len[sid] / n_samples / np.maximum(speed, 1.0), 0.2)
            for j in range(n_samples):
                cols["event_id"].append(eid); eid += 1
                cols["trip_id"].append(trip)
                cols["vehicle_id"].append(vehicle)
                cols["timestamp"].append(t)
                cols["lat"].append(float(lat[j])); cols["lon"].append(float(lon[j]))
                cols["speed_mps"].append(float(speed[j]))
                cols["long_accel_mps2"].append(float(long_acc[j]))
                cols["lat_accel_mps2"].append(float(lat_acc[j]))
                cols["yaw_rate_radps"].append(float(yaw[j]))
                cols["steering_angle_rad"].append(float(steer[j]))
                cols["heading_rad"].append(float(heading))
                cols["weather"].append(str(weather))
                cols["is_night"].append(is_night)
                t = t + dt.timedelta(seconds=float(dt_s[j]))

    frame = pl.DataFrame(cols).with_columns(
        pl.col("timestamp").dt.date().alias("date"))
    _write_partitioned(frame, cfg)
    return frame


def _write_partitioned(frame: pl.DataFrame, cfg: Settings) -> None:
    """Write raw/date=YYYY-MM-DD/events.parquet (S3-style partitioning)."""
    for (date,), part in frame.group_by(["date"], maintain_order=True):
        out = cfg.raw_dir / f"date={date}"
        out.mkdir(parents=True, exist_ok=True)
        part.drop("date").write_parquet(out / "events.parquet")
