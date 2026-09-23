# Multi-stage Dockerfile for SatQuery AI FastAPI Backend
FROM python:3.12-slim AS builder

WORKDIR /app

# Install GDAL and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gdal-bin \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY backend/pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn pydantic pydantic-settings \
    rasterio geopandas shapely numpy scipy opencv-python-headless \
    torch torchvision transformers python-multipart sqlalchemy aiosqlite pytest

FROM python:3.12-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal32 \
    gdal-bin \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY backend/ /app/backend/
COPY data/ /app/data/
COPY storage/ /app/storage/

ENV PYTHONPATH=/app
ENV SATQUERY_STORAGE_ROOT=/app/storage
ENV SATQUERY_DEMO_DATA_ROOT=/app/data/demo

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
