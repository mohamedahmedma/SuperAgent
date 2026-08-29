# syntax=docker/dockerfile:1.7
FROM python:3.12.11-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN groupadd --system app && useradd --system --gid app --home /app app
WORKDIR /app
COPY records/requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir -r /tmp/requirements.txt
COPY --chown=app:app records ./records
COPY --chown=app:app schoolauth ./schoolauth
USER app
EXPOSE 8100
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/health', timeout=3)"
CMD ["uvicorn", "records.app:app", "--host", "0.0.0.0", "--port", "8100"]
