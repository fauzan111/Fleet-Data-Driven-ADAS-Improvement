"""
Central configuration.

Paths describe an S3-style data lake laid out in zones (raw / curated /
features). Locally these are folders; in the cloud the same prefixes map onto an
S3 bucket (see README for the AWS mapping). All tunables are overridable via
environment variables (prefix ``FLEET_``) so the same code runs locally, in
Docker, and in the cloud without edits.
"""
from __future__ import annotations

import pathlib

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FLEET_", env_file=".env",
                                      extra="ignore")

    # --- Data lake (local folders <-> S3 prefixes) ---------------------------
    data_root: pathlib.Path = REPO_ROOT / "data"
    model_root: pathlib.Path = REPO_ROOT / "models"

    # --- Geography: bounding box over Northern Italy (Milan–Turin corridor) --
    lat_min: float = 45.00
    lat_max: float = 45.60
    lon_min: float = 7.60
    lon_max: float = 9.30

    # --- Synthetic fleet size ------------------------------------------------
    n_segments: int = 260          # road segments in the reference network
    n_trips: int = 6000            # simulated trips
    hazard_fraction: float = 0.18  # fraction of segments that are hazardous
    seed: int = 42

    # --- Safety-critical event thresholds (physics of the flags) -------------
    hard_brake_g: float = 0.40     # |decel| >= 0.40 g  -> hard braking
    swerve_yawrate: float = 0.35   # |yaw rate| rad/s   -> evasive swerve
    hard_stop_speed: float = 1.0   # speed drops below this after braking -> stop

    # --- Hotspot clustering (DBSCAN) -----------------------------------------
    hotspot_eps_m: float = 120.0   # cluster radius [m]
    hotspot_min_events: int = 8    # min safety-critical events to form a hotspot

    @property
    def raw_dir(self) -> pathlib.Path:
        return self.data_root / "raw"

    @property
    def curated_dir(self) -> pathlib.Path:
        return self.data_root / "curated"

    @property
    def features_dir(self) -> pathlib.Path:
        return self.data_root / "features"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.curated_dir, self.features_dir,
                  self.model_root):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
