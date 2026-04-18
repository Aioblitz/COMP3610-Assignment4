from fastapi import FastAPI
from pydantic import BaseModel, Field
import mlflow.sklearn
import uuid
import os
import pandas as pd
import joblib
import numpy as np
from fastapi.responses import JSONResponse
from fastapi.requests import Request
#
# Part 2: Model Serving with FastAPI
#

#create app
app = FastAPI()

model = joblib.load("models/random_forest_reg.pkl")

#create feature list and define
features_list = joblib.load("models/features.pkl")
class TripFeatures(BaseModel):
    passenger_count: int = Field(..., ge=1, le=6)
    trip_distance: float = Field(..., gt=0)
    trip_duration_minutes: float = Field(..., gt=0)
    pickup_hour: int = Field(..., ge=0, le=23)
    is_weekend: int = Field(..., ge=0, le=1)

    pickup_borough_label: int
    dropoff_borough_label: int

    fare_amount: float = Field(..., gt=0)
    extra: float = Field(..., ge=0)
    mta_tax: float = Field(..., ge=0)
    tolls_amount: float = Field(..., ge=0)
    congestion_surcharge: float = Field(..., ge=0)
    Airport_fee: float = Field(..., ge=0)



@app.post("/predict")
def predict(features: TripFeatures):

    data = features.model_dump()

    # --- Derived features ---
    data["log_trip_distance"] = np.log1p(data["trip_distance"])
    duration_hours = data["trip_duration_minutes"] / 60

    data["trip_speed_mph"] = data["trip_distance"] / duration_hours if duration_hours > 0 else 0
    data["fare_per_mile"] = data["fare_amount"] / data["trip_distance"] if data["trip_distance"] > 0 else 0
    data["fare_per_minute"] = data["fare_amount"] / data["trip_duration_minutes"] if data["trip_duration_minutes"] > 0 else 0

    data["fare_per_mile"] = data["fare_amount"] / data["trip_distance"]
    data["fare_per_minute"] = data["fare_amount"] / data["trip_duration_minutes"]

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Ensure correct feature order
    df = df[features_list]

    pred = model.predict(df)[0]

    return {
        "prediction_id": str(uuid.uuid4()),
        "tip_amount": round(float(pred), 2),
        "model_version": "1"
    }

@app.post("/predict/batch")
def predict_batch(features_list_input: list[TripFeatures]):

    if len(features_list_input) > 100:
        return {"error": "Maximum 100 records allowed"}

    data_list = []

    for features in features_list_input:
        data = features.model_dump()

        # Derived features
        data["log_trip_distance"] = np.log1p(data["trip_distance"])

        duration_hours = data["trip_duration_minutes"] / 60

        data["trip_speed_mph"] = data["trip_distance"] / duration_hours if duration_hours > 0 else 0
        data["fare_per_mile"] = data["fare_amount"] / data["trip_distance"] if data["trip_distance"] > 0 else 0
        data["fare_per_minute"] = data["fare_amount"] / data["trip_duration_minutes"] if data["trip_duration_minutes"] > 0 else 0

        data_list.append(data)

    df = pd.DataFrame(data_list)
    df = df[features_list]

    preds = model.predict(df)

    return {
        "predictions": [round(float(p), 2) for p in preds],
        "model_version": "1"
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_version": "1"
    }

@app.get("/model/info")
def model_info():
    return {
        "model_name": "taxi-tip-regressor",
        "model_version": "1",
        "features": features_list
    }



@app.exception_handler(Exception)
def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "Something went wrong"
        },
    )