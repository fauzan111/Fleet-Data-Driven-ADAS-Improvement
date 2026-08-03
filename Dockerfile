FROM python:3.12-slim

# libgomp1 is required by LightGBM at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fleet_adas ./fleet_adas
COPY docker/entrypoint.sh ./docker/entrypoint.sh
RUN chmod +x ./docker/entrypoint.sh

ENV FLEET_DATA_ROOT=/app/data \
    FLEET_MODEL_ROOT=/app/models \
    FLEET_N_TRIPS=4000 \
    PYTHONUNBUFFERED=1

# Bake the data lake + trained model into the image at build time, so the
# container serves instantly with no per-boot pipeline (keeps the runtime well
# within a 512 MB free-tier instance and avoids slow cold starts).
RUN python -m fleet_adas.cli all

EXPOSE 8000

# Serve the API (artifacts already baked). Honors $PORT for PaaS hosts (Render).
ENTRYPOINT ["./docker/entrypoint.sh"]
