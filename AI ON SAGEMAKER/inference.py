"""
inference.py — SageMaker serving script (per-station-type models)
=====================================================================
This is NOT run directly. deploy_endpoint.py packages this alongside your
trained model.tar.gz and SageMaker calls these four functions automatically
whenever the endpoint receives a request.

CHANGED FROM THE ORIGINAL: the old version assumed every station reports the
same 7 sensors. They don't anymore — Station 1/3 report vibration_1/2 +
moisture_1/2 (4 values), Station 2 reports temperature_1/2/3 + pressure_1/2
+ ultrasonic (6 values). Those aren't reconcilable into one feature vector,
so this file now bundles TWO separate model/scaler pairs and picks the
right one per request. The endpoint itself is still ONE endpoint under the
same name — only what's inside model.tar.gz changed (see train_local.py).

Request formats:
  - JSON (recommended): a flat object of {feature_name: value}, e.g.
      {"vibration_1": 2.4, "vibration_2": 2.6, "moisture_1": 12.1, "moisture_2": 11.8}
    The right model is chosen automatically from which feature names are
    present. You can also pass an explicit "model" field to be unambiguous:
      {"model": "vibration_moisture", "vibration_1": 2.4, ...}
  - CSV (legacy): a comma-separated list of values, with NO names attached.
    Since the two models have different feature counts (4 vs 6), the model
    is chosen by length: 4 values -> vibration_moisture, 6 values ->
    thermal_pressure. This only works because the two sets happen not to
    share a length — if that ever changes, switch the caller to JSON.

Response: JSON with keys model_used, health_score, anomaly, score, severity
"""

import os
import json
import joblib
import pandas as pd

FEATURES_BY_MODEL = {
    "vibration_moisture": ["vibration_1", "vibration_2", "moisture_1", "moisture_2"],
    "thermal_pressure":   ["temperature_1", "temperature_2", "temperature_3", "pressure_1", "pressure_2", "ultrasonic"],
}


def model_fn(model_dir):
    """Called once when the endpoint container starts. Loads both model/scaler
    pairs plus the score->health calibration computed at training time (see
    train_local.py — this replaces a fixed (score+0.5)*100 formula that
    verifiably mislabeled the vast majority of genuinely normal data as
    "warning" severity)."""
    bundles = {}
    for name in FEATURES_BY_MODEL:
        bundles[name] = {
            "model":  joblib.load(os.path.join(model_dir, f"isolation_forest_{name}.joblib")),
            "scaler": joblib.load(os.path.join(model_dir, f"scaler_{name}.joblib")),
        }
    calibration = joblib.load(os.path.join(model_dir, "score_calibration.joblib"))
    for name, (low, high) in calibration.items():
        bundles[name]["score_low"] = low
        bundles[name]["score_high"] = high
    return bundles


def _pick_model_by_keys(data_keys):
    """Auto-detect which model a JSON request is for, from which feature
    names showed up as keys. Returns None if it can't tell."""
    keys_lower = {k.lower() for k in data_keys}
    for name, feats in FEATURES_BY_MODEL.items():
        if keys_lower & {f.lower() for f in feats}:
            return name
    return None


def input_fn(request_body, request_content_type):
    if request_content_type == "text/csv":
        values = [float(x) for x in request_body.strip().split(",")]
        if len(values) == len(FEATURES_BY_MODEL["vibration_moisture"]):
            model_name = "vibration_moisture"
        elif len(values) == len(FEATURES_BY_MODEL["thermal_pressure"]):
            model_name = "thermal_pressure"
        else:
            raise ValueError(
                f"CSV has {len(values)} values — expected "
                f"{len(FEATURES_BY_MODEL['vibration_moisture'])} (vibration_moisture) or "
                f"{len(FEATURES_BY_MODEL['thermal_pressure'])} (thermal_pressure). "
                f"Use JSON with named keys instead if you need to be explicit."
            )
        features = FEATURES_BY_MODEL[model_name]
        return {"model_name": model_name, "values": dict(zip(features, values))}

    elif request_content_type == "application/json":
        data = json.loads(request_body)
        if not isinstance(data, dict):
            raise ValueError('JSON input must be an object of {"feature_name": value, ...}')

        model_name = data.get("model") or _pick_model_by_keys(data.keys())
        if model_name not in FEATURES_BY_MODEL:
            raise ValueError(
                'Could not determine which model to use. Pass "model": "vibration_moisture" '
                'or "model": "thermal_pressure" explicitly, or include at least one of that '
                f"model's feature names as a key. Known models: {list(FEATURES_BY_MODEL.keys())}"
            )
        features = FEATURES_BY_MODEL[model_name]
        values = {f: float(data.get(f, 0.0)) for f in features}
        return {"model_name": model_name, "values": values}

    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_bundles):
    """Same scoring formula as train_local.py's predict() function, applied
    to whichever model input_fn picked."""
    model_name = input_data["model_name"]
    features   = FEATURES_BY_MODEL[model_name]
    bundle     = model_bundles[model_name]
    model, scaler = bundle["model"], bundle["scaler"]
    score_low, score_high = bundle["score_low"], bundle["score_high"]

    row  = [input_data["values"][f] for f in features]
    X_s  = scaler.transform(pd.DataFrame([row], columns=features))
    score = float(model.decision_function(X_s)[0])
    pred  = int(model.predict(X_s)[0])

    # Percentile-calibrated, not a fixed (score+0.5)*100 formula — see
    # train_local.py's train_one() for why the fixed version was wrong.
    if score_high == score_low:
        health = 50.0
    else:
        pct = (score - score_low) / (score_high - score_low)
        health = round(max(0, min(100, pct * 100)), 1)

    return {
        "model_used":   model_name,
        "health_score": health,
        "anomaly":      pred == -1,
        "score":        round(score, 4),
        "severity":     "critical" if health < 40 else "warning" if health < 70 else "normal",
    }


def output_fn(prediction, response_content_type):
    """Always return JSON regardless of what was requested."""
    return json.dumps(prediction), "application/json"
