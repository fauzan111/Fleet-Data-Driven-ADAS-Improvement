"""Geospatial helpers — great-circle distance and point→segment matching."""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres. Scalar or numpy-broadcastable."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def local_meters_per_degree(lat_deg: float):
    """Metres per degree of latitude / longitude at a given latitude.
    Good enough to treat a small region as a local planar frame."""
    lat = np.radians(lat_deg)
    m_per_deg_lat = 111_132.92 - 559.82 * np.cos(2 * lat) + 1.175 * np.cos(4 * lat)
    m_per_deg_lon = 111_412.84 * np.cos(lat) - 93.5 * np.cos(3 * lat)
    return m_per_deg_lat, m_per_deg_lon


def to_local_xy(lat, lon, lat0, lon0):
    """Project lat/lon to local planar metres around (lat0, lon0)."""
    m_lat, m_lon = local_meters_per_degree(lat0)
    x = (np.asarray(lon) - lon0) * m_lon
    y = (np.asarray(lat) - lat0) * m_lat
    return x, y


def point_segment_distance_xy(px, py, ax, ay, bx, by):
    """Distance from point(s) P to segment AB, all in a planar frame.
    Returns (distance, t) where t in [0,1] is the projection parameter."""
    px, py = np.asarray(px), np.asarray(py)
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    denom = np.where(denom == 0, 1e-9, denom)  # guard zero-length segments
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = np.clip(t, 0.0, 1.0)
    cx, cy = ax + t * abx, ay + t * aby
    return np.hypot(px - cx, py - cy), t
