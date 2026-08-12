FROM python:3.10-slim as builder

WORKDIR /build
RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock* ./
RUN poetry export -f requirements.txt --only main --no-interaction --no-ansi > requirements.txt


FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "ecommerce_buy_predictor.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
