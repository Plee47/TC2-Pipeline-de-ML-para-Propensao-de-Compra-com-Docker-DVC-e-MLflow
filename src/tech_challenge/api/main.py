from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import mlflow
import pandas as pd
import numpy as np
from tech_challenge.config import settings
from tech_challenge.api.schemas import PredictionRequest, PredictionResponse


model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        model = mlflow.pyfunc.load_model(
            f"models:/online_shoppers_intention/Production"
        )
    except Exception:
        print("Warning: Could not load model from registry, using None")
    yield


app = FastAPI(title="Online Shoppers Prediction API", lifespan=lifespan)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    global model

    if model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please train and register model first."
        )

    try:
        X = pd.DataFrame([request.features]).astype(np.float32)
        prediction = model.predict(X)[0]
        probability = float(prediction) if isinstance(prediction, (int, float)) else float(prediction[1])

        return PredictionResponse(prediction=int(prediction), probability=probability)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
