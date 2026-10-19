FROM python:3.11.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system adapts \
    && useradd --system --gid adapts --home-dir /app adapts

COPY requirements-production.txt ./
RUN pip install --requirement requirements-production.txt

COPY . .
RUN mkdir -p backups buffers exports logs repro_snapshots \
    && chown -R adapts:adapts /app

USER adapts

EXPOSE 8000

CMD ["gunicorn", "--bind=0.0.0.0:8000", "--workers=1", "--threads=1", "--timeout=120", "--access-logfile=-", "--error-logfile=-", "run:app"]
