# syntax=docker/dockerfile:1.7
ARG VERSION=1.0.0

FROM python:3.10-slim AS trainer
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt
COPY app/ ./app/
ARG EPOCHS=10
RUN --mount=type=cache,target=/build/data \
    python -m app.train --classes yes no up down --epochs ${EPOCHS} --seed 42 \
        --data-dir /build/data --out /build/artifacts

FROM python:3.10-slim AS runtime
ARG VERSION
WORKDIR /app
LABEL org.opencontainers.image.title="kws-api" \
      org.opencontainers.image.version="${VERSION}"
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY app/ ./app/
COPY --from=trainer /build/artifacts ./artifacts

ENV MODEL_PATH=/app/artifacts/model.pt \
    MODEL_VERSION=${VERSION} \
    NUM_THREADS=1 \
    LOG_LEVEL=info \
    PORT=8000 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os, urllib.request as u; u.urlopen('http://localhost:%s/health' % os.environ['PORT'])"
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT}"]