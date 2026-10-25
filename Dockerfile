FROM python:3.12-slim

LABEL maintainer="Shadow AI"
LABEL version="1.0.0"
LABEL description="Shadow AI - Personal AI Assistant"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV SHADOW_DATA_DIR=/app/data
ENV SHADOW_PORT=8765
ENV SHADOW_HOST=0.0.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN id shadow 2>/dev/null || (groupadd -r shadowgroup && useradd --create-home --shell /bin/bash -g shadowgroup shadow)

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY shadowd/ ./shadowd/
COPY shikigami/ ./shikigami/
COPY sensors/ ./sensors/
COPY summon/ ./summon/
COPY phone/ ./phone/
COPY web/ ./web/
COPY scripts/entry.sh ./scripts/entry.sh

RUN mkdir -p /app/data/memory /app/data/logs /app/workspace && \
    chown -R shadow:shadowgroup /app

USER shadow

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${SHADOW_PORT}/ || exit 1

ENTRYPOINT ["bash", "/app/scripts/entry.sh"]
CMD ["python", "-m", "shadowd.shadowd"]