# humanizer service — production image.
#
# Build:   docker build -t humanizer:latest .
# Run:     docker run -p 8000:8000 \
#            -e OPENAI_API_KEY=... \
#            -e ORIGINALITY_API_KEY=... \
#            -e HUMANIZER_API_KEY=... \
#            humanizer:latest
#
# Notes:
#   - Slim base + only `serve` + `openai` extras. No torch/transformers
#     baked in: the default `--judge auto` falls back to roberta-large
#     ONLY if no paid keys are set, which would require torch. For paid-
#     judge-only deployments this image stays small (~250MB).
#   - To support `--judge roberta` fallback, see Dockerfile.full (TODO).
#
FROM python:3.11-slim AS build

# Build deps for any wheels that need compilation (numpy etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY humanizer ./humanizer
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir '.[serve,openai]'

FROM python:3.11-slim AS runtime
COPY --from=build /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=build /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY --from=build /usr/local/bin/humanizer /usr/local/bin/humanizer

# Sensible defaults; override at `docker run` time.
ENV HUMANIZER_JUDGE=auto \
    HUMANIZER_REJECT_N=8 \
    HUMANIZER_REJECT_ROUNDS=4 \
    HUMANIZER_REJECT_THRESHOLD=0.05 \
    HUMANIZER_MAX_CHARS=20000

# Healthcheck hits /health which is auth-free.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3).read()"

EXPOSE 8000
CMD ["uvicorn", "humanizer.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
