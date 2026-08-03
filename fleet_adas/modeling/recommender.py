"""
ADAS recommendation policy.

Translates a segment's risk score (and road context) into concrete driver-
assistance parameter tuning — the actionable output a car maker would push to
the fleet. The policy is deliberately transparent (bounded, monotonic, and
explainable) rather than a black box, because ADAS calibration changes must be
justifiable for safety sign-off.

Baselines reflect typical passenger-car ADAS defaults; risk shifts them toward
earlier warnings, larger following gaps, and more assertive intervention.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

# Nominal ADAS defaults (low-risk baseline)
FCW_BASE_S = 2.0        # forward-collision-warning lead time
HEADWAY_BASE_S = 1.4    # adaptive-cruise following gap (time headway)
LKA_BASE_GAIN = 0.4     # lane-keeping assist steering gain [0..1]


@dataclass
class AdasRecommendation:
    risk_score: float
    risk_category: str
    fcw_lead_time_s: float          # earlier warning as risk rises
    headway_time_s: float           # larger following gap as risk rises
    aeb_sensitivity: str            # standard | elevated | high
    lka_gain: float                 # firmer lane-keeping in curvy/risky spots
    speed_advisory_kph: int | None  # advisory cap on high-risk segments
    rationale: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


def _category(score: float) -> str:
    return "high" if score >= 0.66 else "medium" if score >= 0.33 else "low"


def recommend_adas(risk_score: float, curvature: float = 0.2,
                   speed_limit_kph: float | None = None) -> AdasRecommendation:
    r = float(max(0.0, min(1.0, risk_score)))
    cat = _category(r)

    fcw = FCW_BASE_S + 1.5 * r                         # 2.0 .. 3.5 s
    headway = HEADWAY_BASE_S + 1.1 * r                 # 1.4 .. 2.5 s
    lka = min(1.0, LKA_BASE_GAIN + 0.5 * r * (0.5 + curvature))
    aeb = "high" if r >= 0.66 else "elevated" if r >= 0.33 else "standard"

    speed_advisory = None
    if speed_limit_kph is not None and r >= 0.5:
        # advise up to 15% below the limit on the riskiest segments
        speed_advisory = int(round(speed_limit_kph * (1 - 0.15 * r)))

    rationale = [f"risk score {r:.2f} ({cat})",
                 f"forward-collision warning brought forward to {fcw:.1f}s",
                 f"following gap widened to {headway:.1f}s time headway",
                 f"AEB sensitivity set to '{aeb}'"]
    if curvature >= 0.35:
        rationale.append(f"lane-keeping firmed up for high curvature ({curvature:.2f})")
    if speed_advisory is not None:
        rationale.append(f"advisory speed {speed_advisory} km/h "
                         f"(below the {speed_limit_kph:.0f} km/h limit)")

    return AdasRecommendation(
        risk_score=round(r, 4), risk_category=cat,
        fcw_lead_time_s=round(fcw, 2), headway_time_s=round(headway, 2),
        aeb_sensitivity=aeb, lka_gain=round(lka, 3),
        speed_advisory_kph=speed_advisory, rationale=rationale)


def recommend_for_route(segment_risks: list[dict]) -> dict:
    """Aggregate a route's segments into one conservative recommendation.

    ``segment_risks`` items: {segment_id, risk_score, curvature, speed_limit_kph}.
    We tune ADAS for the *riskiest* segment on the route (safety is limited by
    its worst point) and also report the mean risk for context.
    """
    if not segment_risks:
        raise ValueError("no segments provided")
    worst = max(segment_risks, key=lambda s: s["risk_score"])
    mean_risk = sum(s["risk_score"] for s in segment_risks) / len(segment_risks)
    rec = recommend_adas(worst["risk_score"], worst.get("curvature", 0.2),
                         worst.get("speed_limit_kph"))
    return {
        "n_segments": len(segment_risks),
        "mean_risk_score": round(mean_risk, 4),
        "max_risk_score": round(worst["risk_score"], 4),
        "critical_segment_id": worst.get("segment_id"),
        "adas_recommendation": rec.as_dict(),
    }
