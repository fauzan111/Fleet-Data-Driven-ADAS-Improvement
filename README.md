# Fleet-Data-Driven ADAS Improvement

An end-to-end system that turns **anonymised fleet driving data** into
**per-road-segment risk scores** and **concrete ADAS (Advanced Driver-Assistance
System) parameter recommendations** — served over a production-style FastAPI
backend.

> **Why this exists.** Volkswagen Group (via **Cariad** and the **VW–Mobileye**
> partnership) and **Stellantis** collect anonymised connected-vehicle data
> across Europe — hard-braking, evasive swerves, abrupt stops — to find where
> and why drivers hit the limits of their assistance systems, and to retune those
> systems (earlier collision warnings, longer following gaps, more assertive
> intervention). This project reproduces that data-to-decision loop: **ingest →
> detect hazards → learn risk → recommend ADAS tuning → serve**.

---

## Results (synthetic Milan–Turin fleet)

Reproducible with `python -m fleet_adas.cli all` (fixed seeds, deterministic model):

| Stage | Output |
|---|---|
| Raw telematics ingested | **383,462** samples over 30 days, 6,000 trips |
| Safety-critical events derived | **12,125** (hard-brake / swerve / abrupt-stop) |
| Hazard hotspots detected (DBSCAN) | **273** |
| Risk model — ranks latent hazard | **Spearman 0.84** (predicted risk vs. hidden ground-truth hazard) |
| Risk model — hotspot targeting | **81% precision@top-decile** (top-10% risk-scored segments that are truly high-hazard) |
| Risk model — rate regression | R² 0.32, MAE 11 events/1k samples on held-out **future** window |

The model never sees the ground-truth hazard or the future events — it forecasts
future risk from **historical fleet events + road context**, then is validated
against the hidden hazard it was never trained on.

---

## Architecture

```
                         ┌──────────────────────── data lake (S3-style zones) ───────────────────────┐
                         │  raw/date=YYYY-MM-DD/   curated/   features/                                │
                         └───────────────────────────────────────────────────────────────────────────┘
                              ▲              ▲             ▲              ▲
  ┌───────────────┐    ┌──────┴─────┐  ┌─────┴──────┐ ┌───┴────────┐ ┌───┴──────────┐   ┌──────────────┐
  │  synthetic    │──▶ │  generate  │─▶│    ETL     │▶│  segment   │▶│ hotspot +    │──▶│  FastAPI     │
  │  fleet + HD   │    │ telematics │  │ derive     │ │ map-match  │ │ risk model   │   │  service     │
  │  road network │    │  (no labels)│ │ safety flag│ │ + aggregate│ │ (LightGBM)   │   │ /risk-score  │
  └───────────────┘    └────────────┘  └────────────┘ └────────────┘ └──────────────┘   │ /hotspots    │
                                                                                         │ /recommend…  │
                                                                                         └──────────────┘
```

Everything runs locally (folders + LightGBM + Uvicorn) but is laid out to lift
directly into AWS — see [Cloud-ready design](#cloud-ready-design).

---

## How it works

### 1. Data generation (`fleet_adas/generation`)
A synthetic **road network** over the Milan–Turin corridor (segments with road
class, speed limit, curvature, and a *hidden* latent hazard) plus a **telematics
generator** that simulates trips over it. Hazardous segments, rain and night
raise the rate of hard-braking / swerve / abrupt-stop **physics** — but the raw
feed carries **no labels**, exactly like real onboard-sensor data.

### 2. ETL (`fleet_adas/pipeline/etl.py`)
Reads the date-partitioned raw zone and derives the safety-critical flags from
the sensor physics (`decel ≥ 0.40 g` → hard brake, `|yaw rate| ≥ 0.35 rad/s` →
swerve, …), classifies each event, and computes a severity.

### 3. Map-matching + aggregation (`pipeline/segments.py`)
Each event is snapped to the nearest road segment with a KD-tree + exact
point-to-segment distance (the pipeline re-derives segment IDs; it never sees the
generator's). Per-segment aggregates (event rates, speed stats, weather/night
mix) form the model's feature table.

### 4. Hotspot detection (`pipeline/hotspots.py`)
**DBSCAN** spatially clusters safety-critical events into hazard hotspots —
density-based, so it needs no preset cluster count and discards one-off events as
noise.

### 5. Risk model (`fleet_adas/modeling/risk_model.py`)
A **LightGBM** regressor forecasts a segment's *future* safety-critical rate from
road character, driving context, **and the segment's own historical event rate**
(a temporal train/test split, so it's genuine forecasting, not leakage). The rate
is mapped to a 0–1 `risk_score`.

### 6. ADAS recommender (`modeling/recommender.py`)
Maps risk (+ curvature, speed limit) to concrete, bounded, **explainable** tuning:
forward-collision-warning lead time, following-gap time headway, AEB sensitivity,
lane-keeping gain, and an advisory speed on the riskiest segments — with a
rationale for each change (ADAS calibration must be justifiable for safety
sign-off).

---

## API

```bash
uvicorn fleet_adas.api.main:app --reload      # http://localhost:8000/docs
```

| Endpoint | Description |
|---|---|
| `GET /health` | liveness + artifact status |
| `GET /risk-score?lat=&lon=` | risk score + ADAS tuning for a location |
| `GET /hotspots?limit=` | ranked hazard hotspots |
| `GET /segments/{id}` | risk for a specific road segment |
| `POST /recommendations` | route-level ADAS recommendation (tuned to the route's riskiest segment) |

Example — risk + ADAS tuning at a high-risk location:

```jsonc
GET /risk-score?lat=45.31&lon=9.02
{
  "matched_segment_id": 31, "risk_score": 1.0, "risk_category": "high",
  "road_type": "urban", "speed_limit_kph": 34, "curvature": 0.58,
  "adas": {
    "fcw_lead_time_s": 3.5, "headway_time_s": 2.5, "aeb_sensitivity": "high",
    "lka_gain": 0.94, "speed_advisory_kph": 29,
    "rationale": [
      "risk score 1.00 (high)",
      "forward-collision warning brought forward to 3.5s",
      "following gap widened to 2.5s time headway",
      "AEB sensitivity set to 'high'",
      "lane-keeping firmed up for high curvature (0.58)",
      "advisory speed 29 km/h (below the 34 km/h limit)"
    ]
  }
}
```

---

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |  Unix: source .venv/bin/activate
pip install -r requirements.txt

python -m fleet_adas.cli all         # generate -> etl -> features -> hotspots -> train (~10s)
uvicorn fleet_adas.api.main:app --reload
```

### Docker (fully self-contained)

```bash
docker compose up --build            # builds the lake + model on first boot, then serves :8000
```

### Tests

```bash
pytest -q                            # 15 tests: pipeline physics, model, recommender, API
```

CI runs the suite + a small-fleet pipeline smoke test on every push
(`.github/workflows/ci.yml`).

---

## Cloud-ready design

The local layout maps 1:1 onto AWS; nothing in the code changes (paths and knobs
are environment-driven via `FLEET_*`):

| Local | AWS | Role |
|---|---|---|
| `data/raw|curated|features/` | **S3** (partitioned prefixes) | data lake zones |
| `python -m fleet_adas.cli` stages | **AWS Batch / ECS task** or **Glue** | scheduled batch ETL + training |
| LightGBM artifact in `models/` | **S3** + **SageMaker** (optional) | model registry / training |
| FastAPI + Uvicorn | **ECS Fargate** behind an **ALB** | serving |
| logs | **CloudWatch** | monitoring |

**Streaming ingestion** (the real-time path): the generator stands in for a
**Kinesis / Kafka** topic of live telematics; a consumer would land micro-batches
into `raw/`, and the same ETL + scoring runs incrementally. The batch pipeline
here is the backbone; the streaming layer is an additive front-end (roadmap).

---

## Project layout

```
fleet_adas/
  config.py                 env-driven settings + data-lake paths
  geo.py                    haversine + point→segment matching
  road_network.py           synthetic HD-map road network
  generation/generator.py   telematics event generator (no labels)
  pipeline/
    etl.py                  raw → curated (derive safety flags)
    segments.py             map-match + per-segment aggregation
    hotspots.py             DBSCAN hazard-hotspot detection
  modeling/
    features.py             feature engineering (no target leakage)
    risk_model.py           LightGBM risk model (temporal holdout)
    recommender.py          risk → ADAS parameter policy
  api/                      FastAPI app, schemas, serving layer
  cli.py                    pipeline orchestration
tests/                      pytest suite (pipeline, model, API)
Dockerfile / docker-compose.yml / .github/workflows/ci.yml
```

## Roadmap
- Streaming ingestion (Kinesis/Kafka) + incremental scoring.
- Terraform IaC for the AWS mapping above.
- Real open telematics/crash data adapter alongside the synthetic generator.
- Prometheus/Grafana dashboards for data-drift and model monitoring.

## Context
Inspired by Volkswagen Cariad, the VW–Mobileye ADAS data partnership, and
Stellantis' connected-vehicle / "AI from design to driving" initiatives.
Data is fully synthetic; no real vehicle data is used.
