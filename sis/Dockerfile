# syntax=docker/dockerfile:1.7
FROM node:24.19.0-alpine AS web-builder
WORKDIR /src/frontend
COPY sis/frontend/package.json sis/frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci
COPY sis/frontend/ ./
RUN npm run build

FROM python:3.12.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN groupadd --system app && useradd --system --gid app --home /app app
WORKDIR /app
COPY sis/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir -r /tmp/requirements.txt
COPY --chown=app:app sis ./sis
RUN rm -rf ./sis/web
COPY --from=web-builder --chown=app:app /src/web ./sis/web
RUN mkdir -p /app/data && chown app:app /app/data
USER app
EXPOSE 8300
HEALTHCHECK --interval=15s --timeout=5s --start-period=25s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8300/health', timeout=3)"
CMD ["sh", "-c", "alembic -c sis/alembic.ini upgrade head && exec uvicorn sis.app:app --host 0.0.0.0 --port 8300"]
