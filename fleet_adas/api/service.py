"""
Serving layer.

Loads the trained model, the scored segment table, the road network, and the
hotspot table once, then answers risk / recommendation queries. Segment risk is
pre-computed by the pipeline, so requests are simple lookups (fast, no model
inference on the hot path — inference happens offline in the pipeline).
"""
from __future__ import annotations

import logging

import polars as pl

from ..config import Settings, settings as default_settings
from ..modeling.recommender import recommend_adas, recommend_for_route
from ..road_network import RoadNetwork

log = logging.getLogger("fleet_adas.api")


class FleetService:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or default_settings
        self.net = RoadNetwork.load(self.cfg)
        self.seg_risk = pl.read_parquet(self.cfg.features_dir / "segment_risk.parquet")
        self._seg_by_id = {r["segment_id"]: r
                           for r in self.seg_risk.iter_rows(named=True)}
        try:
            self.hotspots = pl.read_parquet(self.cfg.features_dir / "hotspots.parquet")
        except FileNotFoundError:
            self.hotspots = pl.DataFrame()
        log.info("service ready: %d segments, %d hotspots",
                 len(self._seg_by_id), self.hotspots.height)

    # ------------------------------------------------------------------ risk
    def risk_at(self, lat: float, lon: float) -> dict:
        seg_id, dist, _ = self.net.match([lat], [lon])
        sid = int(seg_id[0])
        row = self._seg_by_id[sid]
        rec = recommend_adas(row["risk_score"], row["curvature"],
                             row["speed_limit_kph"])
        return {
            "lat": lat, "lon": lon,
            "matched_segment_id": sid,
            "match_distance_m": round(float(dist[0]), 1),
            "road_type": row["road_type"],
            "speed_limit_kph": int(row["speed_limit_kph"]),
            "curvature": round(float(row["curvature"]), 4),
            "risk_score": round(float(row["risk_score"]), 4),
            "risk_category": row["risk_category"],
            "adas": rec.as_dict(),
        }

    def segment_risk(self, sid: int) -> dict | None:
        row = self._seg_by_id.get(sid)
        if row is None:
            return None
        return {
            "segment_id": sid,
            "risk_score": round(float(row["risk_score"]), 4),
            "risk_category": row["risk_category"],
            "curvature": round(float(row["curvature"]), 4),
            "speed_limit_kph": int(row["speed_limit_kph"]),
        }

    # -------------------------------------------------------------- hotspots
    def top_hotspots(self, limit: int = 20) -> list[dict]:
        if self.hotspots.height == 0:
            return []
        cols = ["hotspot_id", "lat", "lon", "n_events", "dominant_type",
                "mean_severity", "hotspot_score"]
        return (self.hotspots.sort("hotspot_score", descending=True)
                .head(limit).select(cols).to_dicts())

    # --------------------------------------------------------------- routes
    def route_recommendation(self, waypoints: list[tuple[float, float]]) -> dict:
        lats = [w[0] for w in waypoints]
        lons = [w[1] for w in waypoints]
        seg_ids, _, _ = self.net.match(lats, lons)

        seen, ordered = set(), []
        for sid in seg_ids:
            sid = int(sid)
            if sid not in seen:
                seen.add(sid)
                ordered.append(self.segment_risk(sid))

        result = recommend_for_route(ordered)
        result["per_segment"] = ordered
        return result


_service: FleetService | None = None


def get_service(cfg: Settings | None = None) -> FleetService:
    """Lazy singleton so artifacts load once per process."""
    global _service
    if _service is None:
        _service = FleetService(cfg)
    return _service
