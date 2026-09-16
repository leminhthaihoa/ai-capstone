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

# from pathlib import Path
# from typing import Any

# import joblib
# import pandas as pd
# from fastapi import FastAPI, HTTPException, Request

# APP_DIR = Path(__file__).resolve().parent
# MODEL_DIR = APP_DIR / "model"

# FEATURES_BY_MODEL = {
#     "vibration_moisture": [
#         "vibration_1",
#         "vibration_2",
#         "moisture_1",
#         "moisture_2",
#     ],
#     "thermal_pressure": [
#         "temperature_1",
#         "temperature_2",
#         "temperature_3",
#         "pressure_1",
#         "pressure_2",
#         "ultrasonic",
#     ],
# }

# app = FastAPI(
#     title="Factory Local AI Inference Server",
#     version="1.0.0",
# )

# # Loaded once when the process starts.
# BUNDLES: dict[str, dict[str, Any]] = {}
# LOAD_ERROR: str | None = None


# def load_models() -> None:
#     global BUNDLES, LOAD_ERROR

#     try:
#         bundles: dict[str, dict[str, Any]] = {}

#         for name in FEATURES_BY_MODEL:
#             model_path = MODEL_DIR / f"isolation_forest_{name}.joblib"
#             scaler_path = MODEL_DIR / f"scaler_{name}.joblib"

#             if not model_path.exists():
#                 raise FileNotFoundError(f"Missing model file: {model_path}")
#             if not scaler_path.exists():
#                 raise FileNotFoundError(f"Missing scaler file: {scaler_path}")

#             bundles[name] = {
#                 "model": joblib.load(model_path),
#                 "scaler": joblib.load(scaler_path),
#             }

#         calibration_path = MODEL_DIR / "score_calibration.joblib"
#         if not calibration_path.exists():
#             raise FileNotFoundError(
#                 f"Missing calibration file: {calibration_path}"
#             )

#         calibration = joblib.load(calibration_path)

#         for name, (low, high) in calibration.items():
#             if name not in bundles:
#                 continue
#             bundles[name]["score_low"] = float(low)
#             bundles[name]["score_high"] = float(high)

#         # Make sure every model has calibration values.
#         for name in FEATURES_BY_MODEL:
#             if "score_low" not in bundles[name] or "score_high" not in bundles[name]:
#                 raise ValueError(
#                     f"No score calibration found for model '{name}'. "
#                     "Re-run train_local.py."
#                 )

#         BUNDLES = bundles
#         LOAD_ERROR = None

#     except Exception as exc:
#         BUNDLES = {}
#         LOAD_ERROR = str(exc)


# def pick_model_by_keys(data_keys) -> str | None:
#     keys_lower = {str(k).lower() for k in data_keys}

#     for name, features in FEATURES_BY_MODEL.items():
#         feature_names = {f.lower() for f in features}
#         if keys_lower & feature_names:
#             return name

#     return None


# def choose_model(data: dict[str, Any]) -> str:
#     explicit = data.get("model")

#     if explicit:
#         model_name = str(explicit).strip()
#         if model_name not in FEATURES_BY_MODEL:
#             raise ValueError(
#                 f"Unknown model '{model_name}'. "
#                 f"Known models: {list(FEATURES_BY_MODEL)}"
#             )
#         return model_name

#     model_name = pick_model_by_keys(data.keys())

#     if model_name is None:
#         raise ValueError(
#             'Could not determine the model. Pass "model": '
#             '"vibration_moisture" or "model": "thermal_pressure", '
#             'or include sensor feature names.'
#         )

#     return model_name


# def predict(data: dict[str, Any]) -> dict[str, Any]:
#     if not BUNDLES:
#         raise RuntimeError(
#             LOAD_ERROR
#             or "Models are not loaded. Check the ./model directory."
#         )

#     model_name = choose_model(data)
#     features = FEATURES_BY_MODEL[model_name]
#     bundle = BUNDLES[model_name]

#     # Match inference(2).py: missing features become 0.0.
#     values = {}
#     for feature in features:
#         try:
#             values[feature] = float(data.get(feature, 0.0))
#         except (TypeError, ValueError):
#             raise ValueError(
#                 f"Feature '{feature}' must be numeric."
#             )

#     row = [values[f] for f in features]
#     X_scaled = bundle["scaler"].transform(
#         pd.DataFrame([row], columns=features)
#     )

#     score = float(bundle["model"].decision_function(X_scaled)[0])
#     prediction = int(bundle["model"].predict(X_scaled)[0])

#     score_low = float(bundle["score_low"])
#     score_high = float(bundle["score_high"])

#     if score_high == score_low:
#         health = 50.0
#     else:
#         pct = (score - score_low) / (score_high - score_low)
#         health = round(max(0.0, min(100.0, pct * 100.0)), 1)

#     return {
#         "model_used": model_name,
#         "health_score": health,
#         "anomaly": prediction == -1,
#         "score": round(score, 4),
#         "severity": (
#             "critical"
#             if health < 40
#             else "warning"
#             if health < 70
#             else "normal"
#         ),
#     }


# @app.on_event("startup")
# def startup_event():
#     load_models()


# @app.get("/")
# def root():
#     return {
#         "service": "Factory Local AI Inference Server",
#         "status": "ok" if BUNDLES else "model_error",
#         "models": list(BUNDLES.keys()),
#         "predict_endpoint": "/predict",
#     }


# @app.get("/health")
# def health():
#     if not BUNDLES:
#         return {
#             "status": "error",
#             "models_loaded": False,
#             "error": LOAD_ERROR,
#         }

#     return {
#         "status": "ok",
#         "models_loaded": True,
#         "models": list(BUNDLES.keys()),
#     }


# @app.post("/predict")
# async def predict_endpoint(request: Request):
#     try:
#         content_type = request.headers.get("content-type", "").lower()

#         if "application/json" not in content_type:
#             raise HTTPException(
#                 status_code=415,
#                 detail="Content-Type must be application/json.",
#             )

#         data = await request.json()

#         if not isinstance(data, dict):
#             raise HTTPException(
#                 status_code=400,
#                 detail="JSON body must be an object.",
#             )

#         result = predict(data)
#         return result

#     except HTTPException:
#         raise
#     except ValueError as exc:
#         raise HTTPException(status_code=400, detail=str(exc))
#     except RuntimeError as exc:
#         raise HTTPException(status_code=503, detail=str(exc))
#     except Exception as exc:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Inference error: {exc}",
#         )





"""
local_ai_server.py
------------------
Local HTTP inference server for the factory anomaly-detection models.

This replaces the SageMaker endpoint for local/demo use.

CHANGED: routes by FACTORY now, not by station-type ("vibration_moisture"/
"thermal_pressure"). Station 1 and Station 3 both report the same 4
vibration/moisture feature names but train_local.py now trains a SEPARATE
model for each — verified that their independently-tuned thresholds are
different enough (sometimes opposite: one station tight on vibration_1 and
loose on vibration_2, the other the reverse) that sharing one model, even
with averaged thresholds, didn't serve either station well. Since factory1
and factory3 share identical feature NAMES, this server can no longer
auto-detect which one a request is for from the feature keys alone — the
caller must always pass an explicit "model" field naming the factory
(e.g. "factory1", "factory2", "factory3"). ai_inference.py already does
this correctly.

Loads whichever factories are actually present in features_by_model.joblib
at startup — it does NOT hardcode a factory list, so this file doesn't need
editing again if train_local.py's set of trained factories ever changes.

Expected files in ./model/ (produced by train_local.py):
    features_by_model.joblib   -- {factory: [feature names]}
    isolation_forest_<factory>.joblib   (one per factory)
    scaler_<factory>.joblib             (one per factory)
    score_calibration.joblib   -- {factory: (score_low, score_high)}

Run:
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install fastapi uvicorn joblib pandas scikit-learn
    uvicorn local_ai_server:app --host 0.0.0.0 --port 8000

Test locally:
    curl -X POST http://127.0.0.1:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"model":"factory1","vibration_1":2.4,"vibration_2":2.6,"moisture_1":12.1,"moisture_2":11.8}'
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"

app = FastAPI(
    title="Factory Local AI Inference Server",
    version="2.0.0",
)

# Loaded once when the process starts.
FEATURES_BY_MODEL: dict[str, list[str]] = {}   # factory -> feature names, loaded from disk
BUNDLES: dict[str, dict[str, Any]] = {}        # factory -> {model, scaler, score_low, score_high}
LOAD_ERROR: str | None = None


def load_models() -> None:
    global FEATURES_BY_MODEL, BUNDLES, LOAD_ERROR

    try:
        features_path = MODEL_DIR / "features_by_model.joblib"
        if not features_path.exists():
            raise FileNotFoundError(f"Missing {features_path} — run train_local.py first.")
        features_by_model = joblib.load(features_path)

        calibration_path = MODEL_DIR / "score_calibration.joblib"
        if not calibration_path.exists():
            raise FileNotFoundError(f"Missing calibration file: {calibration_path}")
        calibration = joblib.load(calibration_path)

        bundles: dict[str, dict[str, Any]] = {}
        for factory in features_by_model:
            model_path = MODEL_DIR / f"isolation_forest_{factory}.joblib"
            scaler_path = MODEL_DIR / f"scaler_{factory}.joblib"

            if not model_path.exists():
                raise FileNotFoundError(f"Missing model file: {model_path}")
            if not scaler_path.exists():
                raise FileNotFoundError(f"Missing scaler file: {scaler_path}")
            if factory not in calibration:
                raise ValueError(
                    f"No score calibration found for '{factory}'. Re-run train_local.py."
                )

            low, high = calibration[factory]
            bundles[factory] = {
                "model":      joblib.load(model_path),
                "scaler":     joblib.load(scaler_path),
                "score_low":  float(low),
                "score_high": float(high),
            }

        FEATURES_BY_MODEL = features_by_model
        BUNDLES = bundles
        LOAD_ERROR = None

    except Exception as exc:
        FEATURES_BY_MODEL = {}
        BUNDLES = {}
        LOAD_ERROR = str(exc)


def choose_model(data: dict[str, Any]) -> str:
    """Returns the factory this request is for. Always requires an explicit
    "model" field now — factory1 and factory3 share identical feature
    names, so there's no reliable way to infer which one a request means
    just from which keys are present, unlike the old station-type version
    of this function."""
    explicit = data.get("model")

    if not explicit:
        raise ValueError(
            'Missing "model" field — pass the factory this request is for, '
            f'e.g. "model": "factory1". Known factories: {list(FEATURES_BY_MODEL)}'
        )

    factory = str(explicit).strip()
    if factory not in FEATURES_BY_MODEL:
        raise ValueError(
            f"Unknown factory '{factory}'. Known factories: {list(FEATURES_BY_MODEL)}"
        )
    return factory


def predict(data: dict[str, Any]) -> dict[str, Any]:
    if not BUNDLES:
        raise RuntimeError(
            LOAD_ERROR
            or "Models are not loaded. Check the ./model directory."
        )

    factory = choose_model(data)
    features = FEATURES_BY_MODEL[factory]
    bundle = BUNDLES[factory]

    # Match inference.py: missing features become 0.0.
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

    score_low = bundle["score_low"]
    score_high = bundle["score_high"]

    if score_high == score_low:
        health = 50.0
    else:
        pct = (score - score_low) / (score_high - score_low)
        health = round(max(0.0, min(100.0, pct * 100.0)), 1)

    return {
        "model_used": factory,
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


@app.api_route("/", methods=["GET", "HEAD"])
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