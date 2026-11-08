FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependency layer changes less often than source, so Docker can cache it.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

RUN groupadd --system visionguard && useradd --system --gid visionguard --create-home visionguard

COPY webapp/ webapp/
COPY serve.py .
RUN mkdir -p /app/instance /app/model && chown -R visionguard:visionguard /app

USER visionguard
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=90s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["python", "serve.py"]
