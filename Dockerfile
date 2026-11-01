# ---- build stage: wheels only, no compilers in the final image ----
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /w
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt

# ---- runtime stage ----
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 SPAMGUARD_HOME=/app SPAMGUARD_LOG_LEVEL=INFO
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 10001 appuser
WORKDIR /app
COPY --from=build /wheels /wheels
RUN pip install --no-index --find-links=/wheels /wheels/* && rm -rf /wheels
COPY pyproject.toml README.md params.yaml ./
COPY src ./src
RUN pip install --no-deps -e . && mkdir -p models logs data/feedback reports && chown -R appuser /app
# Models are mounted (-v ./models:/app/models) or baked in with a build-arg-driven COPY in CI/CD.
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "spamguard.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
