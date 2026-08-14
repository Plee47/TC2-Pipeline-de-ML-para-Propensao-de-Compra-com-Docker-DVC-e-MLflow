FROM python:3.10-slim AS builder

WORKDIR /build

# poetry-plugin-export is required: `poetry export` left Poetry core in 2.x.
RUN pip install --no-cache-dir "poetry==1.8.5" "poetry-plugin-export==1.8.0"

COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt --only main --without-hashes -o requirements.txt


FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

COPY --from=builder /build/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY params.yaml ./
COPY models/ ./models/

RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "ecommerce_buy_predictor.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
