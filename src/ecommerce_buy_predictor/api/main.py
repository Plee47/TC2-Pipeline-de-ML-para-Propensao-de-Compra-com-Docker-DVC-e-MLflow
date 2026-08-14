import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from fastapi import FastAPI, HTTPException

from ecommerce_buy_predictor.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from ecommerce_buy_predictor.config import settings
from ecommerce_buy_predictor.features.schema import to_model_frame
from ecommerce_buy_predictor.models.train import ModelTrainer

logger = logging.getLogger(__name__)


@dataclass
class ModelState:
    """Model currently served, plus where it came from."""

    model: Any = None
    source: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None


state = ModelState()


def load_model() -> ModelState:
    """Load the serving model: Registry first, local pickle as fallback.

    The fallback keeps the API usable in a plain ``docker run`` where no
    MLflow server is reachable.
    """
    registry_uri = f"models:/{settings.model_name}@{settings.model_alias}"
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        return ModelState(mlflow.sklearn.load_model(registry_uri), registry_uri)
    except Exception as error:
        logger.warning(
            "Could not load %s from the Model Registry: %s", registry_uri, error
        )

    local_path = Path(settings.model_dir) / "model.pkl"
    if local_path.exists():
        try:
            return ModelState(ModelTrainer.load_model(str(local_path)), str(local_path))
        except Exception:
            logger.exception("Could not load local model %s", local_path)

    logger.error("No model available: /predict will answer 503")
    return ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global state
    state = load_model()
    if state.is_loaded:
        logger.info("Model loaded from %s", state.source)
    yield
    state = ModelState()


app = FastAPI(
    title="Online Shoppers Prediction API",
    description="Predicts the purchase intention of an e-commerce session.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Liveness probe that also reports whether a model is being served."""
    return HealthResponse(
        status="healthy",
        model_loaded=state.is_loaded,
        model_source=state.source,
    )


def _score(payloads: list[PredictionRequest]) -> list[PredictionResponse]:
    """Run the model over validated payloads."""
    if not state.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the pipeline and register a model first.",
        )

    frame = to_model_frame([payload.model_dump() for payload in payloads])

    try:
        probabilities = state.model.predict_proba(frame)[:, 1]
    except Exception as error:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed") from error

    return [
        PredictionResponse(prediction=int(probability >= 0.5), probability=float(probability))
        for probability in probabilities
    ]


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Score a single session."""
    return _score([request])[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Score up to 1000 sessions in one call."""
    return BatchPredictionResponse(predictions=_score(request.sessions))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
