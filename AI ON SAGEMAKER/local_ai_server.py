"""
local_ai_server.py
------------------
Local HTTP inference server for the factory anomaly-detection models.

This replaces the SageMaker endpoint for local/demo use.

Expected model files in ./model/:
    isolation_forest_vibration_moisture.joblib
    scaler_vibration_moisture.joblib
    isolation_forest_thermal_pressure.joblib
    scaler_thermal_pressure.joblib
    score_calibration.joblib

Run:
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install fastapi uvicorn joblib pandas scikit-learn
    uvicorn local_ai_server:app --host 0.0.0.0 --port 8000

Test locally:
    curl -X POST http://127.0.0.1:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"model":"vibration_moisture","vibration_1":2.4,"vibration_2":2.6,"moisture_1":12.1,"moisture_2":11.8}'
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"

FEATURES_BY_MODEL = {
    "vibration_moisture": [
        "vibration_1",
        "vibration_2",
        "moisture_1",
        "moisture_2",
    ],
    "thermal_pressure": [
        "temperature_1",
        "temperature_2",
        "temperature_3",
        "pressure_1",
        "pressure_2",
        "ultrasonic",
    ],
}

app = FastAPI(
    title="Factory Local AI Inference Server",
    version="1.0.0",
)

# Loaded once when the process starts.
BUNDLES: dict[str, dict[str, Any]] = {}
LOAD_ERROR: str | None = None


def load_models() -> None:
    global BUNDLES, LOAD_ERROR

    try:
        bundles: dict[str, dict[str, Any]] = {}

        for name in FEATURES_BY_MODEL:
            model_path = MODEL_DIR / f"isolation_forest_{name}.joblib"
            scaler_path = MODEL_DIR / f"scaler_{name}.joblib"

            if not model_path.exists():
                raise FileNotFoundError(f"Missing model file: {model_path}")
            if not scaler_path.exists():
                raise FileNotFoundError(f"Missing scaler file: {scaler_path}")

            bundles[name] = {
                "model": joblib.load(model_path),
                "scaler": joblib.load(scaler_path),
            }

        calibration_path = MODEL_DIR / "score_calibration.joblib"
        if not calibration_path.exists():
            raise FileNotFoundError(
                f"Missing calibration file: {calibration_path}"
            )

        calibration = joblib.load(calibration_path)

        for name, (low, high) in calibration.items():
            if name not in bundles:
                continue
            bundles[name]["score_low"] = float(low)
            bundles[name]["score_high"] = float(high)

        # Make sure every model has calibration values.
        for name in FEATURES_BY_MODEL:
            if "score_low" not in bundles[name] or "score_high" not in bundles[name]:
                raise ValueError(
                    f"No score calibration found for model '{name}'. "
                    "Re-run train_local.py."
                )

        BUNDLES = bundles
        LOAD_ERROR = None

    except Exception as exc:
        BUNDLES = {}
        LOAD_ERROR = str(exc)


def pick_model_by_keys(data_keys) -> str | None:
    keys_lower = {str(k).lower() for k in data_keys}

    for name, features in FEATURES_BY_MODEL.items():
        feature_names = {f.lower() for f in features}
        if keys_lower & feature_names:
            return name

    return None


def choose_model(data: dict[str, Any]) -> str:
    explicit = data.get("model")

    if explicit:
        model_name = str(explicit).strip()
        if model_name not in FEATURES_BY_MODEL:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Known models: {list(FEATURES_BY_MODEL)}"
            )
        return model_name

    model_name = pick_model_by_keys(data.keys())

    if model_name is None:
        raise ValueError(
            'Could not determine the model. Pass "model": '
            '"vibration_moisture" or "model": "thermal_pressure", '
            'or include sensor feature names.'
        )

    return model_name


def predict(data: dict[str, Any]) -> dict[str, Any]:
    if not BUNDLES:
        raise RuntimeError(
            LOAD_ERROR
            or "Models are not loaded. Check the ./model directory."
        )

    model_name = choose_model(data)
    features = FEATURES_BY_MODEL[model_name]
    bundle = BUNDLES[model_name]

    # Match inference(2).py: missing features become 0.0.
    values = {}
    for feature in features:
        try:
            values[feature] = float(data.get(feature, 0.0))
        except (TypeError, ValueError):
            raise ValueError(
                f"Feature '{feature}' must be numeric."
            )

    row = [values[f] for f in features]
    X_scaled = bundle["scaler"].transform(
        pd.DataFrame([row], columns=features)
    )

    score = float(bundle["model"].decision_function(X_scaled)[0])
    prediction = int(bundle["model"].predict(X_scaled)[0])

    score_low = float(bundle["score_low"])
    score_high = float(bundle["score_high"])

    if score_high == score_low:
        health = 50.0
    else:
        pct = (score - score_low) / (score_high - score_low)
        health = round(max(0.0, min(100.0, pct * 100.0)), 1)

    return {
        "model_used": model_name,
        "health_score": health,
        "anomaly": prediction == -1,
        "score": round(score, 4),
        "severity": (
            "critical"
            if health < 40
            else "warning"
            if health < 70
            else "normal"
        ),
    }


@app.on_event("startup")
def startup_event():
    load_models()


@app.get("/")
def root():
    return {
        "service": "Factory Local AI Inference Server",
        "status": "ok" if BUNDLES else "model_error",
        "models": list(BUNDLES.keys()),
        "predict_endpoint": "/predict",
    }


@app.get("/health")
def health():
    if not BUNDLES:
        return {
            "status": "error",
            "models_loaded": False,
            "error": LOAD_ERROR,
        }

    return {
        "status": "ok",
        "models_loaded": True,
        "models": list(BUNDLES.keys()),
    }


@app.post("/predict")
async def predict_endpoint(request: Request):
    try:
        content_type = request.headers.get("content-type", "").lower()

        if "application/json" not in content_type:
            raise HTTPException(
                status_code=415,
                detail="Content-Type must be application/json.",
            )

        data = await request.json()

        if not isinstance(data, dict):
            raise HTTPException(
                status_code=400,
                detail="JSON body must be an object.",
            )

        result = predict(data)
        return result

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {exc}",
        )
