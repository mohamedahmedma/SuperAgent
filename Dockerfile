# syntax=docker/dockerfile:1.7
FROM ghcr.io/astral-sh/uv:0.8.14 AS uv
FROM python:3.12.11-slim-bookworm AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
# --locked fails the build when uv.lock does not match pyproject.toml. With --frozen a
# dependency added to pyproject but never locked is silently left out, and the image
# builds, ships and then dies at import — which is how a missing snowballstemmer reached
# production. The check costs nothing and turns that into a failed build.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project
COPY backend ./backend
COPY schoolauth ./schoolauth
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev

FROM python:3.12.11-slim-bookworm
ENV PATH=/app/.venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN groupadd --system app && useradd --system --gid app --home /app app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv ./.venv
COPY --chown=app:app backend ./backend
COPY --chown=app:app schoolauth ./schoolauth
# /app/hf-cache is a mount point for the HuggingFace model cache. Creating it here,
# owned by app, means the named volume mounted over it inherits that ownership —
# a volume mounted onto a path absent from the image would land root-owned and be
# unwritable by the non-root user.
RUN mkdir -p /app/data /app/uploads /app/hf-cache \
    && chown -R app:app /app/data /app/uploads /app/hf-cache
USER app
EXPOSE 8000
# Leave room for cold-start database/schema checks and slow container hosts. The model
# itself warms in the background, so /health normally becomes reachable within seconds.
HEALTHCHECK --interval=15s --timeout=5s --start-period=300s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
