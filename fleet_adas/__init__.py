"""
fleet_adas — Fleet-data-driven ADAS improvement.

An end-to-end pipeline that ingests anonymised fleet driving events, detects
hazard hotspots, learns a per-road-segment risk score, and recommends ADAS
parameter tuning — served over a FastAPI backend.

Inspired by how Volkswagen Group (Cariad / VW–Mobileye) and Stellantis use
anonymised connected-vehicle data across Europe to refine driver-assistance
systems.
"""

__version__ = "0.1.0"
