"""
Synthetic road network — the reference "HD map" the fleet is matched against.

A network of road segments over the Milan–Turin corridor. Each segment has road
class, speed limit, curvature and length, plus a *latent* hazard level (ground
truth we plant so the learned risk model can be validated). Real deployments
would swap this for an OpenStreetMap / HD-map network; the rest of the pipeline
is unchanged because it only consumes the segment table + a `match()` method.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from . import geo
from .config import Settings, settings as default_settings

ROAD_TYPES = ("highway", "rural", "urban")
# probability, (min_len_m, max_len_m), (min_speed, max_speed), (min_curv, max_curv)
ROAD_SPEC = {
    "highway": (0.18, (2000, 6000), (100, 130), (0.00, 0.15)),
    "rural":   (0.32, (700, 2500), (60, 90), (0.10, 0.45)),
    "urban":   (0.50, (200, 900), (30, 50), (0.15, 0.60)),
}


class RoadNetwork:
    def __init__(self, frame: pl.DataFrame, lat0: float, lon0: float):
        self.frame = frame
        self.lat0, self.lon0 = lat0, lon0
        self._build_index()

    # ------------------------------------------------------------------ build
    def _build_index(self) -> None:
        f = self.frame
        self.seg_id = f["segment_id"].to_numpy()
        ax, ay = geo.to_local_xy(f["a_lat"].to_numpy(), f["a_lon"].to_numpy(),
                                 self.lat0, self.lon0)
        bx, by = geo.to_local_xy(f["b_lat"].to_numpy(), f["b_lon"].to_numpy(),
                                 self.lat0, self.lon0)
        self._ax, self._ay, self._bx, self._by = ax, ay, bx, by
        self._mid = np.column_stack(((ax + bx) / 2, (ay + by) / 2))
        self._tree = cKDTree(self._mid)

    @classmethod
    def generate(cls, cfg: Settings | None = None) -> "RoadNetwork":
        cfg = cfg or default_settings
        rng = np.random.default_rng(cfg.seed)
        lat0 = (cfg.lat_min + cfg.lat_max) / 2
        lon0 = (cfg.lon_min + cfg.lon_max) / 2
        m_lat, m_lon = geo.local_meters_per_degree(lat0)

        # a few "town" clusters so segments group like a real network
        n_towns = 6
        town_lat = rng.uniform(cfg.lat_min, cfg.lat_max, n_towns)
        town_lon = rng.uniform(cfg.lon_min, cfg.lon_max, n_towns)

        types = list(ROAD_SPEC.keys())
        probs = np.array([ROAD_SPEC[t][0] for t in types])
        probs = probs / probs.sum()

        rows = []
        n_haz = int(cfg.hazard_fraction * cfg.n_segments)
        hazard_flags = np.zeros(cfg.n_segments, dtype=bool)
        hazard_flags[rng.choice(cfg.n_segments, n_haz, replace=False)] = True

        for i in range(cfg.n_segments):
            rtype = rng.choice(types, p=probs)
            _, (lmin, lmax), (smin, smax), (cmin, cmax) = ROAD_SPEC[rtype]
            length = rng.uniform(lmin, lmax)
            speed = int(rng.integers(smin, smax + 1))
            curv = rng.uniform(cmin, cmax)

            # anchor near a town for urban/rural, spread out for highway
            t = rng.integers(n_towns)
            spread = 0.18 if rtype == "highway" else 0.05
            a_lat = np.clip(town_lat[t] + rng.normal(0, spread), cfg.lat_min, cfg.lat_max)
            a_lon = np.clip(town_lon[t] + rng.normal(0, spread), cfg.lon_min, cfg.lon_max)
            heading = rng.uniform(0, 2 * np.pi)
            b_lat = np.clip(a_lat + (length * np.sin(heading)) / m_lat,
                            cfg.lat_min, cfg.lat_max)
            b_lon = np.clip(a_lon + (length * np.cos(heading)) / m_lon,
                            cfg.lon_min, cfg.lon_max)

            # Latent hazard: driven by curvature + road class, elevated on the
            # planted-hazard subset, plus noise. This is the signal the risk
            # model must recover from observed events.
            base = 0.12 + 0.5 * curv + (0.15 if rtype == "urban" else 0.0)
            if hazard_flags[i]:
                base += rng.uniform(0.30, 0.55)
            hazard = float(np.clip(base + rng.normal(0, 0.05), 0.02, 0.98))

            rows.append({
                "segment_id": i,
                "road_type": rtype,
                "speed_limit_kph": speed,
                "curvature": round(curv, 4),
                "length_m": round(length, 1),
                "a_lat": a_lat, "a_lon": a_lon, "b_lat": b_lat, "b_lon": b_lon,
                "is_planted_hazard": bool(hazard_flags[i]),
                "hazard_level": round(hazard, 4),
            })

        frame = pl.DataFrame(rows)
        return cls(frame, lat0, lon0)

    # ------------------------------------------------------------------- use
    def match(self, lats, lons, k: int = 8):
        """Map-match points to the nearest road segment.
        Returns (segment_id, distance_m, t) arrays. Uses a KD-tree over segment
        midpoints to shortlist candidates, then exact point-to-segment distance.
        """
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        px, py = geo.to_local_xy(lats, lons, self.lat0, self.lon0)
        k = min(k, len(self.seg_id))
        _, cand = self._tree.query(np.column_stack((px, py)), k=k)
        cand = np.atleast_2d(cand.T).T if cand.ndim == 1 else cand

        best_d = np.full(len(px), np.inf)
        best_seg = np.zeros(len(px), dtype=int)
        best_t = np.zeros(len(px))
        for j in range(cand.shape[1]):
            c = cand[:, j]
            d, t = geo.point_segment_distance_xy(
                px, py, self._ax[c], self._ay[c], self._bx[c], self._by[c])
            better = d < best_d
            best_d = np.where(better, d, best_d)
            best_seg = np.where(better, self.seg_id[c], best_seg)
            best_t = np.where(better, t, best_t)
        return best_seg, best_d, best_t

    # ----------------------------------------------------------------- io
    def save(self, cfg: Settings | None = None) -> None:
        cfg = cfg or default_settings
        cfg.ensure_dirs()
        self.frame.write_parquet(cfg.curated_dir / "road_network.parquet")

    @classmethod
    def load(cls, cfg: Settings | None = None) -> "RoadNetwork":
        cfg = cfg or default_settings
        frame = pl.read_parquet(cfg.curated_dir / "road_network.parquet")
        lat0 = (cfg.lat_min + cfg.lat_max) / 2
        lon0 = (cfg.lon_min + cfg.lon_max) / 2
        return cls(frame, lat0, lon0)
