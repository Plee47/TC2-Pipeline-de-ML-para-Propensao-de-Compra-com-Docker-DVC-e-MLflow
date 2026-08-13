
from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """One browsing session, with the raw feature names of the dataset.

    Values are the *raw* ones: scaling and encoding happen inside the model
    pipeline, so callers never need to know how the model was trained.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Administrative": 2,
                "Administrative_Duration": 80.0,
                "Informational": 0,
                "Informational_Duration": 0.0,
                "ProductRelated": 31,
                "ProductRelated_Duration": 1200.5,
                "BounceRates": 0.01,
                "ExitRates": 0.03,
                "PageValues": 12.4,
                "SpecialDay": 0.0,
                "Month": "Nov",
                "OperatingSystems": 2,
                "Browser": 2,
                "Region": 1,
                "TrafficType": 3,
                "VisitorType": "Returning_Visitor",
                "Weekend": False,
            }
        }
    )

    Administrative: float = Field(..., ge=0, description="Account pages visited")
    Administrative_Duration: float = Field(..., ge=0, description="Seconds on account pages")
    Informational: float = Field(..., ge=0, description="Informational pages visited")
    Informational_Duration: float = Field(..., ge=0, description="Seconds on informational pages")
    ProductRelated: float = Field(..., ge=0, description="Product pages visited")
    ProductRelated_Duration: float = Field(..., ge=0, description="Seconds on product pages")
    BounceRates: float = Field(..., ge=0, le=1, description="Average bounce rate")
    ExitRates: float = Field(..., ge=0, le=1, description="Average exit rate")
    PageValues: float = Field(..., ge=0, description="Average page value")
    SpecialDay: float = Field(..., ge=0, le=1, description="Closeness to a special day")
    Month: str = Field(..., description="Month abbreviation, e.g. 'Feb', 'June', 'Nov'")
    OperatingSystems: int = Field(..., description="Operating system code")
    Browser: int = Field(..., description="Browser code")
    Region: int = Field(..., description="Region code")
    TrafficType: int = Field(..., description="Traffic source code")
    VisitorType: str = Field(
        ..., description="'Returning_Visitor', 'New_Visitor' or 'Other'"
    )
    Weekend: bool = Field(..., description="Whether the session happened on a weekend")


class BatchPredictionRequest(BaseModel):
    """Several sessions scored in a single call."""

    sessions: list[PredictionRequest] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class: 1 = likely to buy")
    probability: float = Field(
        ..., ge=0, le=1, description="Probability of the positive class"
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_source: str | None = Field(
        None, description="Where the served model was loaded from"
    )
