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
    PYTHONUNBUFFERED=1

EXPOSE 8000

# On first start, build the data lake + train the model, then serve the API.
ENTRYPOINT ["./docker/entrypoint.sh"]
