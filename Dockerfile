FROM python:3.13-slim

WORKDIR /srv/legalintel

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm || true

COPY app ./app
COPY README.md .

ENV PYTHONUNBUFFERED=1
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD uvicorn app.api_gateway:app --host 0.0.0.0 --port ${PORT} --workers 1
