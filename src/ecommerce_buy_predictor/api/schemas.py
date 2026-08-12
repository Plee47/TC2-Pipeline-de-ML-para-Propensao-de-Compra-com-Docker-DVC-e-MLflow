from pydantic import BaseModel, Field
from typing import List


class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="List of feature values")


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class (0 or 1)")
    probability: float = Field(..., description="Probability of positive class")
