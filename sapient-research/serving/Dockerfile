# Secure Sapient sandbox — CPU image, mock engine by default (no GPU, no weights).
# Build:  docker build -t sapient-sandbox .
# Run:    docker run -p 8000:8000 -v "$PWD/keys.json:/data/keys.json:ro" sapient-sandbox
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip

# Copy only what the service needs (NOT .venv / keys.json / .env — see .dockerignore)
COPY pyproject.toml README.md ./
COPY sapient_serving ./sapient_serving
COPY playground ./playground
COPY demo ./demo
COPY scripts ./scripts
RUN pip install --no-cache-dir -e .

ENV SAPIENT_ENGINE=mock \
    SAPIENT_DOCS=0 \
    SAPIENT_KEYS_FILE=/data/keys.json \
    SAPIENT_RESULT_TTL=900 \
    SAPIENT_RATE_LIMIT_PER_MIN=12 \
    SAPIENT_DAILY_QUOTA=200

EXPOSE 8000
# Honor $PORT when the host sets it (Render/Fly), else 8000.
CMD ["sh", "-c", "uvicorn sapient_serving.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
