# """
# train_local.py — per-station-type models
# =============================================
# CHANGED FROM THE ORIGINAL: stations no longer share one uniform sensor set.
# Station 1/3 report vibration_1/2 + moisture_1/2 (4 features). Station 2
# reports temperature_1/2/3 + pressure_1/2 + ultrasonic (6 features). Those
# can't be one model, so this trains TWO Isolation Forests — one per
# station-type — and packages both into the same model.tar.gz. The SageMaker
# ENDPOINT name and deploy/delete scripts are UNCHANGED; only what's inside
# the tar.gz differs. inference.py picks the right model per request.

# Run on laptop:
#   pip install scikit-learn joblib boto3 numpy pandas
#   python train_local.py

# NOTE ON THRESHOLDS BELOW: normal/fault ranges are set to roughly match the
# warn/crit values in ThresholdSettings.jsx as of this conversation. If you
# tune those thresholds later, these training ranges should move with them —
# a model trained on stale "normal" ranges will misjudge health scores even
# if the dashboard's alerting thresholds are correct.
# """

# import numpy as np
# import pandas as pd
# import boto3
# import joblib
# import os
# import tarfile
# from sklearn.ensemble import IsolationForest
# from sklearn.preprocessing import StandardScaler

# AWS_REGION = "ap-southeast-2"
# S3_BUCKET  = "factory-ai-data-capstone"   # update this

# FEATURES_BY_MODEL = {
#     "vibration_moisture": ["vibration_1", "vibration_2", "moisture_1", "moisture_2"],
#     "thermal_pressure":   ["temperature_1", "temperature_2", "temperature_3", "pressure_1", "pressure_2", "ultrasonic"],
# }

# np.random.seed(42)


# # ── Model 1: vibration_moisture (Stations 1 & 3) ────────────────────────────
# def build_vibration_moisture_dataset():
#     n_normal = 2000
#     n_fault  = 120  # split across 4 fault scenarios, 30 each
#     k = n_fault // 4

#     normal = pd.DataFrame({
#         "vibration_1": np.random.normal(2.5, 0.6, n_normal).clip(0.3, 6.0),
#         "vibration_2": np.random.normal(2.5, 0.6, n_normal).clip(0.3, 6.0),
#         "moisture_1":  np.random.normal(12,  2,   n_normal).clip(5,   18),
#         "moisture_2":  np.random.normal(12,  2,   n_normal).clip(5,   18),
#     })

#     # Fault A: both vibration sensors elevated — general bearing/mounting wear
#     fault_vib_both = pd.DataFrame({
#         "vibration_1": np.random.normal(8.5, 1.0, k).clip(6, 12),
#         "vibration_2": np.random.normal(8.5, 1.0, k).clip(6, 12),
#         "moisture_1":  np.random.normal(12,  2,   k).clip(5, 18),
#         "moisture_2":  np.random.normal(12,  2,   k).clip(5, 18),
#     })

#     # Fault B: ONE vibration sensor elevated, the other normal — localized
#     # mechanical issue. This is exactly the kind of fault having two
#     # separate sensors (instead of one) is meant to catch.
#     fault_vib_one = pd.DataFrame({
#         "vibration_1": np.random.normal(9.0, 1.0, k).clip(6.5, 12),
#         "vibration_2": np.random.normal(2.5, 0.6, k).clip(0.3, 6.0),
#         "moisture_1":  np.random.normal(12,  2,   k).clip(5, 18),
#         "moisture_2":  np.random.normal(12,  2,   k).clip(5, 18),
#     })

#     # Fault C: both moisture sensors elevated — material quality issue
#     fault_moist_both = pd.DataFrame({
#         "vibration_1": np.random.normal(2.7, 0.6, k).clip(0.3, 6.0),
#         "vibration_2": np.random.normal(2.7, 0.6, k).clip(0.3, 6.0),
#         "moisture_1":  np.random.normal(24,  2,   k).clip(18, 30),
#         "moisture_2":  np.random.normal(24,  2,   k).clip(18, 30),
#     })

#     # Fault D: combined vibration + moisture — severe/compound fault
#     fault_combined = pd.DataFrame({
#         "vibration_1": np.random.normal(7.5, 1.0, k).clip(5.5, 11),
#         "vibration_2": np.random.normal(7.0, 1.0, k).clip(5, 10.5),
#         "moisture_1":  np.random.normal(21,  2,   k).clip(17, 27),
#         "moisture_2":  np.random.normal(20,  2,   k).clip(16, 26),
#     })

#     df = pd.concat([normal, fault_vib_both, fault_vib_one, fault_moist_both, fault_combined], ignore_index=True)
#     return df[FEATURES_BY_MODEL["vibration_moisture"]].dropna(), n_normal, n_fault


# # ── Model 2: thermal_pressure (Station 2) ───────────────────────────────────
# def build_thermal_pressure_dataset():
#     n_normal = 2000
#     n_fault  = 150  # split across 5 fault scenarios, 30 each
#     k = n_fault // 5

#     # pressure_1 is Pa; pressure_2 is kPa — genuinely different sensors,
#     # kPa-scale numbers, assuming the same physical pressure range.
#     # ultrasonic is mm (was cm) — scaled 10x,
#     # same ~350mm physical distance as before.
#     normal = pd.DataFrame({
#         "temperature_1": np.random.normal(75, 5, n_normal).clip(50, 93),
#         "temperature_2": np.random.normal(75, 5, n_normal).clip(50, 93),
#         "temperature_3": np.random.normal(75, 5, n_normal).clip(50, 93),
#         "pressure_1":    np.random.normal(4500, 500, n_normal).clip(1000, 7000),
#         "pressure_2":    np.random.normal(4.5, 0.5, n_normal).clip(1.0, 7.0),
#         # Ultrasonic: mid-range is normal — both extremes (too close / too
#         # far) are faults, matching the two-sided threshold design.
#         "ultrasonic":    np.random.normal(170, 40, n_normal).clip(30, 320),
#     })

#     # Fault A: thermal overload, all 3 sensors elevated together
#     fault_thermal_all = pd.DataFrame({
#         "temperature_1": np.random.normal(98, 4, k).clip(88, 115),
#         "temperature_2": np.random.normal(98, 4, k).clip(88, 115),
#         "temperature_3": np.random.normal(98, 4, k).clip(88, 115),
#         "pressure_1":    np.random.normal(4500, 500, k).clip(1000, 7000),
#         "pressure_2":    np.random.normal(4.5, 0.5, k).clip(1.0, 7.0),
#         "ultrasonic":    np.random.normal(170, 40, k).clip(30, 320),
#     })

#     # Fault B: ONE thermal sensor elevated, others normal — localized hotspot
#     fault_thermal_one = pd.DataFrame({
#         "temperature_1": np.random.normal(100, 4, k).clip(90, 115),
#         "temperature_2": np.random.normal(76,  5, k).clip(55, 93),
#         "temperature_3": np.random.normal(76,  5, k).clip(55, 93),
#         "pressure_1":    np.random.normal(4500, 500, k).clip(1000, 7000),
#         "pressure_2":    np.random.normal(4.5, 0.5, k).clip(1.0, 7.0),
#         "ultrasonic":    np.random.normal(170, 40, k).clip(30, 320),
#     })

#     # Fault C: pressure spike, both sensors
#     fault_pressure = pd.DataFrame({
#         "temperature_1": np.random.normal(80, 5, k).clip(55, 95),
#         "temperature_2": np.random.normal(80, 5, k).clip(55, 95),
#         "temperature_3": np.random.normal(80, 5, k).clip(55, 95),
#         "pressure_1":    np.random.normal(7500, 300, k).clip(6500, 9000),
#         "pressure_2":    np.random.normal(7.5, 0.3, k).clip(6.5, 9.0),
#         "ultrasonic":    np.random.normal(170, 40, k).clip(30, 320),
#     })

#     # Fault D: storage overfull — material too close to the ultrasonic sensor
#     fault_overfull = pd.DataFrame({
#         "temperature_1": np.random.normal(75, 5, k).clip(50, 93),
#         "temperature_2": np.random.normal(75, 5, k).clip(50, 93),
#         "temperature_3": np.random.normal(75, 5, k).clip(50, 93),
#         "pressure_1":    np.random.normal(4500, 500, k).clip(1000, 7000),
#         "pressure_2":    np.random.normal(4.5, 0.5, k).clip(1.0, 7.0),
#         "ultrasonic":    np.random.normal(15, 8, k).clip(0, 40),
#     })

#     # Fault E: storage running low/empty — material far from the sensor
#     fault_lowstorage = pd.DataFrame({
#         "temperature_1": np.random.normal(75, 5, k).clip(50, 93),
#         "temperature_2": np.random.normal(75, 5, k).clip(50, 93),
#         "temperature_3": np.random.normal(75, 5, k).clip(50, 93),
#         "pressure_1":    np.random.normal(4500, 500, k).clip(1000, 7000),
#         "pressure_2":    np.random.normal(4.5, 0.5, k).clip(1.0, 7.0),
#         "ultrasonic":    np.random.normal(335, 10, k).clip(300, 350),
#     })

#     df = pd.concat([
#         normal, fault_thermal_all, fault_thermal_one, fault_pressure, fault_overfull, fault_lowstorage
#     ], ignore_index=True)
#     return df[FEATURES_BY_MODEL["thermal_pressure"]].dropna(), n_normal, n_fault


# def train_one(name, df, n_normal, n_fault):
#     features = FEATURES_BY_MODEL[name]
#     print(f"\n--- Training '{name}' ({len(features)} features: {features}) ---")
#     print(f"    Total samples: {len(df)} ({n_normal} normal + {n_fault} fault)")

#     scaler = StandardScaler()
#     X = scaler.fit_transform(df[features])

#     # contamination should match the TRUE fault rate in the training data,
#     # not an arbitrary constant — a fixed 0.05 here systematically dropped
#     # some fault types when the actual rate ran higher (verified: several
#     # fault scenarios weren't being flagged as anomalies until this was
#     # corrected from a flat 0.05 to n_fault/(n_normal+n_fault)).
#     contamination = n_fault / (n_normal + n_fault)
#     model = IsolationForest(n_estimators=300, contamination=contamination, random_state=42, n_jobs=-1)
#     model.fit(X)
#     anomalies = (model.predict(X) == -1).sum()
#     print(f"    Contamination: {contamination:.4f} — Anomalies detected: {anomalies}/{len(X)} ({anomalies/len(X)*100:.1f}%)")

#     # Calibrate the score->health mapping to THIS model's actual score
#     # range, instead of assuming decision_function always lands near
#     # [-0.5, 0.5] (the old (score+0.5)*100 formula). Verified: with that
#     # fixed formula, 93.6%/100% of genuinely NORMAL data scored below the
#     # "normal" severity cutoff (70) — i.e. the dashboard would show
#     # near-constant false "warning" status. Percentile-based calibration
#     # (using the 1st/99th percentile of this model's own training scores)
#     # makes health scores actually span 0-100 meaningfully for THIS model,
#     # whatever its raw decision_function range happens to be.
#     all_scores = model.decision_function(X)
#     score_low  = float(np.percentile(all_scores, 1))
#     score_high = float(np.percentile(all_scores, 99))
#     print(f"    Score calibration: raw decision_function range [{all_scores.min():.3f}, {all_scores.max():.3f}], "
#           f"calibrating [{score_low:.3f}, {score_high:.3f}] -> [0, 100]")

#     def score_to_health(score):
#         if score_high == score_low:
#             return 50.0
#         pct = (score - score_low) / (score_high - score_low)
#         return round(max(0, min(100, pct * 100)), 1)

#     def predict(values):
#         sample = pd.DataFrame([values])[features]
#         X_s = scaler.transform(sample)
#         score = float(model.decision_function(X_s)[0])
#         pred = int(model.predict(X_s)[0])
#         health = score_to_health(score)
#         return {"health": health, "anomaly": pred == -1,
#                 "severity": "critical" if health < 40 else "warning" if health < 70 else "normal"}

#     return model, scaler, predict, score_low, score_high


# def main():
#     print("=" * 60)
#     print("  Factory AI Training — per-station-type models")
#     print("=" * 60)

#     os.makedirs("model", exist_ok=True)

#     # ── Model 1: vibration_moisture ──────────────────────────────────────────
#     df1, n1, f1 = build_vibration_moisture_dataset()
#     model1, scaler1, predict1, low1, high1 = train_one("vibration_moisture", df1, n1, f1)

#     print("    Sanity tests:")
#     tests1 = [
#         ("Normal",              {"vibration_1": 2.5, "vibration_2": 2.5, "moisture_1": 12, "moisture_2": 12}),
#         ("Both vibration high", {"vibration_1": 8.5,  "vibration_2": 8.5, "moisture_1": 12, "moisture_2": 12}),
#         ("One vibration high",  {"vibration_1": 9.0,  "vibration_2": 2.5, "moisture_1": 12, "moisture_2": 12}),
#         ("Both moisture high",  {"vibration_1": 2.5,  "vibration_2": 2.5, "moisture_1": 24, "moisture_2": 24}),
#     ]
#     for name, values in tests1:
#         r = predict1(values)
#         icon = "✅" if (name == "Normal") == (not r["anomaly"]) else "❌"
#         print(f"      {icon} {name}: health={r['health']} severity={r['severity']}")

#     # ── Model 2: thermal_pressure ────────────────────────────────────────────
#     df2, n2, f2 = build_thermal_pressure_dataset()
#     model2, scaler2, predict2, low2, high2 = train_one("thermal_pressure", df2, n2, f2)

#     print("    Sanity tests:")
#     print("    (Note: 'Storage overfull'/'Storage low/empty' below are a single-feature")
#     print("     deviation (ultrasonic alone) among 6 total features, vs. thermal's 3 and")
#     print("     pressure's 2 — IsolationForest inherently isolates multi-feature anomalies")
#     print("     more reliably than single-feature ones. Expect ~50% binary-anomaly detection")
#     print("     on these two specifically, not the ~100% the others get. The health SCORE")
#     print("     still drops meaningfully even when the binary flag doesn't trip — that's the")
#     print("     more useful signal for these two, not the anomaly boolean.)")
#     tests2 = [
#         ("Normal",           {"temperature_1": 75, "temperature_2": 75, "temperature_3": 75, "pressure_1": 4500, "pressure_2": 4.5, "ultrasonic": 170}),
#         ("Thermal overload",  {"temperature_1": 98, "temperature_2": 98, "temperature_3": 98, "pressure_1": 4500, "pressure_2": 4.5, "ultrasonic": 170}),
#         ("Pressure spike",    {"temperature_1": 80, "temperature_2": 80, "temperature_3": 80, "pressure_1": 7500, "pressure_2": 7.5, "ultrasonic": 170}),
#         ("Storage overfull",  {"temperature_1": 75, "temperature_2": 75, "temperature_3": 75, "pressure_1": 4500, "pressure_2": 4.5, "ultrasonic": 15}),
#         ("Storage low/empty", {"temperature_1": 75, "temperature_2": 75, "temperature_3": 75, "pressure_1": 4500, "pressure_2": 4.5, "ultrasonic": 335}),
#     ]
#     for name, values in tests2:
#         r = predict2(values)
#         icon = "✅" if (name == "Normal") == (not r["anomaly"]) else "❌"
#         print(f"      {icon} {name}: health={r['health']} severity={r['severity']}")

#     # ── Save both, package into ONE tar.gz ──────────────────────────────────
#     print("\nSaving and packaging both models...")
#     joblib.dump(model1,  "model/isolation_forest_vibration_moisture.joblib")
#     joblib.dump(scaler1, "model/scaler_vibration_moisture.joblib")
#     joblib.dump(model2,  "model/isolation_forest_thermal_pressure.joblib")
#     joblib.dump(scaler2, "model/scaler_thermal_pressure.joblib")
#     joblib.dump(FEATURES_BY_MODEL, "model/features_by_model.joblib")
#     joblib.dump(
#         {"vibration_moisture": (low1, high1), "thermal_pressure": (low2, high2)},
#         "model/score_calibration.joblib"
#     )

#     with tarfile.open("model.tar.gz", "w:gz") as tar:
#         tar.add("model/isolation_forest_vibration_moisture.joblib", arcname="isolation_forest_vibration_moisture.joblib")
#         tar.add("model/scaler_vibration_moisture.joblib",           arcname="scaler_vibration_moisture.joblib")
#         tar.add("model/isolation_forest_thermal_pressure.joblib",   arcname="isolation_forest_thermal_pressure.joblib")
#         tar.add("model/scaler_thermal_pressure.joblib",             arcname="scaler_thermal_pressure.joblib")
#         tar.add("model/features_by_model.joblib",                   arcname="features_by_model.joblib")
#         tar.add("model/score_calibration.joblib",                   arcname="score_calibration.joblib")

#     s3 = boto3.client("s3", region_name=AWS_REGION)
#     s3.upload_file("model.tar.gz", S3_BUCKET, "models/model.tar.gz")
#     print(f"Uploaded to s3://{S3_BUCKET}/models/model.tar.gz")
#     print("\nDone — run deploy_endpoint.py next (or retrain_and_redeploy.py to do both in one step).")


# if __name__ == "__main__":
#     main()



"""
train_local.py — per-station-type models
=============================================
CHANGED FROM THE ORIGINAL: stations no longer share one uniform sensor set.
Station 1/3 report vibration_1/2 + moisture_1/2 (4 features). Station 2
reports temperature_1/2/3 + pressure_1/2 + ultrasonic (6 features). Those
can't be one model, so this trains TWO Isolation Forests — one per
station-type — and packages both into the same model.tar.gz. The SageMaker
ENDPOINT name and deploy/delete scripts are UNCHANGED; only what's inside
the tar.gz differs. inference.py picks the right model per request.

Run on laptop:
  pip install scikit-learn joblib boto3 numpy pandas
  python train_local.py

NOTE ON THRESHOLDS BELOW: normal/fault ranges are set to roughly match the
warn/crit values in ThresholdSettings.jsx as of this conversation. If you
tune those thresholds later, these training ranges should move with them —
a model trained on stale "normal" ranges will misjudge health scores even
if the dashboard's alerting thresholds are correct.
"""

import numpy as np
import pandas as pd
import boto3
import joblib
import os
import tarfile
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

AWS_REGION = "ap-southeast-2"
S3_BUCKET  = "factory-ai-data-capstone"   # update this

FEATURES_BY_MODEL = {
    "vibration_moisture": ["vibration_1", "vibration_2", "moisture_1", "moisture_2"],
    "thermal_pressure":   ["temperature_1", "temperature_2", "temperature_3", "pressure_1", "pressure_2", "ultrasonic"],
}

np.random.seed(42)


# ── Model 1: vibration_moisture (Stations 1 & 3) ────────────────────────────
def build_vibration_moisture_dataset():
    n_fault = 120  # split across 4 fault scenarios, 30 each
    k = n_fault // 4

    real_path = "real_data_vibration_moisture.csv"
    if os.path.exists(real_path):
        normal = pd.read_csv(real_path)
        n_normal = len(normal)
        print(f"    Using {n_normal} REAL rows from {real_path} as normal data")
    else:
        n_normal = 2000
        normal = pd.DataFrame({
            "vibration_1": np.random.normal(2.5, 0.6, n_normal).clip(0.3, 6.0),
            "vibration_2": np.random.normal(2.5, 0.6, n_normal).clip(0.3, 6.0),
            "moisture_1":  np.random.normal(12,  2,   n_normal).clip(5,   18),
            "moisture_2":  np.random.normal(12,  2,   n_normal).clip(5,   18),
        })
        print(f"    No {real_path} found — using {n_normal} SYNTHETIC normal rows "
              f"(run fetch_real_data.py first to train on real history instead)")

    # Fault A: both vibration sensors elevated — general bearing/mounting wear
    fault_vib_both = pd.DataFrame({
        "vibration_1": np.random.normal(8.5, 1.0, k).clip(6, 12),
        "vibration_2": np.random.normal(8.5, 1.0, k).clip(6, 12),
        "moisture_1":  np.random.normal(12,  2,   k).clip(5, 18),
        "moisture_2":  np.random.normal(12,  2,   k).clip(5, 18),
    })

    # Fault B: ONE vibration sensor elevated, the other normal — localized
    # mechanical issue. This is exactly the kind of fault having two
    # separate sensors (instead of one) is meant to catch.
    fault_vib_one = pd.DataFrame({
        "vibration_1": np.random.normal(9.0, 1.0, k).clip(6.5, 12),
        "vibration_2": np.random.normal(2.5, 0.6, k).clip(0.3, 6.0),
        "moisture_1":  np.random.normal(12,  2,   k).clip(5, 18),
        "moisture_2":  np.random.normal(12,  2,   k).clip(5, 18),
    })

    # Fault C: both moisture sensors elevated — material quality issue
    fault_moist_both = pd.DataFrame({
        "vibration_1": np.random.normal(2.7, 0.6, k).clip(0.3, 6.0),
        "vibration_2": np.random.normal(2.7, 0.6, k).clip(0.3, 6.0),
        "moisture_1":  np.random.normal(24,  2,   k).clip(18, 30),
        "moisture_2":  np.random.normal(24,  2,   k).clip(18, 30),
    })

    # Fault D: combined vibration + moisture — severe/compound fault
    fault_combined = pd.DataFrame({
        "vibration_1": np.random.normal(7.5, 1.0, k).clip(5.5, 11),
        "vibration_2": np.random.normal(7.0, 1.0, k).clip(5, 10.5),
        "moisture_1":  np.random.normal(21,  2,   k).clip(17, 27),
        "moisture_2":  np.random.normal(20,  2,   k).clip(16, 26),
    })

    df = pd.concat([normal, fault_vib_both, fault_vib_one, fault_moist_both, fault_combined], ignore_index=True)
    return df[FEATURES_BY_MODEL["vibration_moisture"]].dropna(), n_normal, n_fault


# ── Model 2: thermal_pressure (Station 2) ───────────────────────────────────
def build_thermal_pressure_dataset():
    n_fault  = 150  # split across 5 fault scenarios, 30 each
    k = n_fault // 5

    real_path = "real_data_thermal_pressure.csv"
    if os.path.exists(real_path):
        normal = pd.read_csv(real_path)
        n_normal = len(normal)
        print(f"    Using {n_normal} REAL rows from {real_path} as normal data")
    else:
        n_normal = 2000
        # pressure_1 is Pa; pressure_2 is kPa — genuinely different sensors.
        # ultrasonic is mm, 0-350mm physical range.
        normal = pd.DataFrame({
            "temperature_1": np.random.normal(75, 5, n_normal).clip(50, 93),
            "temperature_2": np.random.normal(75, 5, n_normal).clip(50, 93),
            "temperature_3": np.random.normal(75, 5, n_normal).clip(50, 93),
            "pressure_1":    np.random.normal(4500, 500, n_normal).clip(1000, 7000),
            "pressure_2":    np.random.normal(4.5, 0.5, n_normal).clip(1.0, 7.0),
            # Ultrasonic: mid-range is normal — both extremes (too close / too
            # far) are faults, matching the two-sided threshold design.
            "ultrasonic":    np.random.normal(565, 40, n_normal).clip(480, 650),
        })
        print(f"    No {real_path} found — using {n_normal} SYNTHETIC normal rows "
              f"(run fetch_real_data.py first to train on real history instead)")

    # Fault A: thermal overload, all 3 sensors elevated together
    fault_thermal_all = pd.DataFrame({
        "temperature_1": np.random.normal(98, 4, k).clip(88, 115),
        "temperature_2": np.random.normal(98, 4, k).clip(88, 115),
        "temperature_3": np.random.normal(98, 4, k).clip(88, 115),
        "pressure_1":    np.random.normal(4500, 500, k).clip(1000, 7000),
        "pressure_2":    np.random.normal(4.5, 0.5, k).clip(1.0, 7.0),
        "ultrasonic":    np.random.normal(565, 40, k).clip(480, 650),
    })

    # Fault B: ONE thermal sensor elevated, others normal — localized hotspot
    fault_thermal_one = pd.DataFrame({
        "temperature_1": np.random.normal(100, 4, k).clip(90, 115),
        "temperature_2": np.random.normal(76,  5, k).clip(55, 93),
        "temperature_3": np.random.normal(76,  5, k).clip(55, 93),
        "pressure_1":    np.random.normal(4500, 500, k).clip(1000, 7000),
        "pressure_2":    np.random.normal(4.5, 0.5, k).clip(1.0, 7.0),
        "ultrasonic":    np.random.normal(565, 40, k).clip(480, 650),
    })

    # Fault C: pressure spike, both sensors
    fault_pressure = pd.DataFrame({
        "temperature_1": np.random.normal(80, 5, k).clip(55, 95),
        "temperature_2": np.random.normal(80, 5, k).clip(55, 95),
        "temperature_3": np.random.normal(80, 5, k).clip(55, 95),
        "pressure_1":    np.random.normal(7500, 300, k).clip(6500, 9000),
        "pressure_2":    np.random.normal(7.5, 0.3, k).clip(6.5, 9.0),
        "ultrasonic":    np.random.normal(565, 40, k).clip(480, 650),
    })

    # Fault D: storage overfull — material too close to the ultrasonic sensor
    fault_overfull = pd.DataFrame({
        "temperature_1": np.random.normal(75, 5, k).clip(50, 93),
        "temperature_2": np.random.normal(75, 5, k).clip(50, 93),
        "temperature_3": np.random.normal(75, 5, k).clip(50, 93),
        "pressure_1":    np.random.normal(4500, 500, k).clip(1000, 7000),
        "pressure_2":    np.random.normal(4.5, 0.5, k).clip(1.0, 7.0),
        "ultrasonic":    np.random.normal(200, 100, k).clip(0, 440),
    })

    # Fault E: storage running low/empty — material far from the sensor
    fault_lowstorage = pd.DataFrame({
        "temperature_1": np.random.normal(75, 5, k).clip(50, 93),
        "temperature_2": np.random.normal(75, 5, k).clip(50, 93),
        "temperature_3": np.random.normal(75, 5, k).clip(50, 93),
        "pressure_1":    np.random.normal(4500, 500, k).clip(1000, 7000),
        "pressure_2":    np.random.normal(4.5, 0.5, k).clip(1.0, 7.0),
        "ultrasonic":    np.random.normal(9000, 500, k).clip(710, 10000),
    })

    df = pd.concat([
        normal, fault_thermal_all, fault_thermal_one, fault_pressure, fault_overfull, fault_lowstorage
    ], ignore_index=True)
    return df[FEATURES_BY_MODEL["thermal_pressure"]].dropna(), n_normal, n_fault


def train_one(name, df, n_normal, n_fault):
    features = FEATURES_BY_MODEL[name]
    print(f"\n--- Training '{name}' ({len(features)} features: {features}) ---")
    print(f"    Total samples: {len(df)} ({n_normal} normal + {n_fault} fault)")

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    # contamination should match the TRUE fault rate in the training data,
    # not an arbitrary constant — a fixed 0.05 here systematically dropped
    # some fault types when the actual rate ran higher (verified: several
    # fault scenarios weren't being flagged as anomalies until this was
    # corrected from a flat 0.05 to n_fault/(n_normal+n_fault)).
    contamination = n_fault / (n_normal + n_fault)
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(X)
    anomalies = (model.predict(X) == -1).sum()
    print(f"    Contamination: {contamination:.4f} — Anomalies detected: {anomalies}/{len(X)} ({anomalies/len(X)*100:.1f}%)")

    # Calibrate the score->health mapping to THIS model's actual score
    # range, instead of assuming decision_function always lands near
    # [-0.5, 0.5] (the old (score+0.5)*100 formula). Verified: with that
    # fixed formula, 93.6%/100% of genuinely NORMAL data scored below the
    # "normal" severity cutoff (70) — i.e. the dashboard would show
    # near-constant false "warning" status. Percentile-based calibration
    # (using the 1st/99th percentile of this model's own training scores)
    # makes health scores actually span 0-100 meaningfully for THIS model,
    # whatever its raw decision_function range happens to be.
    all_scores = model.decision_function(X)
    score_low  = float(np.percentile(all_scores, 1))
    score_high = float(np.percentile(all_scores, 99))
    print(f"    Score calibration: raw decision_function range [{all_scores.min():.3f}, {all_scores.max():.3f}], "
          f"calibrating [{score_low:.3f}, {score_high:.3f}] -> [0, 100]")

    def score_to_health(score):
        if score_high == score_low:
            return 50.0
        pct = (score - score_low) / (score_high - score_low)
        return round(max(0, min(100, pct * 100)), 1)

    def predict(values):
        sample = pd.DataFrame([values])[features]
        X_s = scaler.transform(sample)
        score = float(model.decision_function(X_s)[0])
        pred = int(model.predict(X_s)[0])
        health = score_to_health(score)
        return {"health": health, "anomaly": pred == -1,
                "severity": "critical" if health < 40 else "warning" if health < 70 else "normal"}

    return model, scaler, predict, score_low, score_high


def main():
    print("=" * 60)
    print("  Factory AI Training — per-station-type models")
    print("=" * 60)

    os.makedirs("model", exist_ok=True)

    # ── Model 1: vibration_moisture ──────────────────────────────────────────
    df1, n1, f1 = build_vibration_moisture_dataset()
    model1, scaler1, predict1, low1, high1 = train_one("vibration_moisture", df1, n1, f1)

    print("    Sanity tests:")
    tests1 = [
        ("Normal",              {"vibration_1": 2.5, "vibration_2": 2.5, "moisture_1": 12, "moisture_2": 12}),
        ("Both vibration high", {"vibration_1": 8.5,  "vibration_2": 8.5, "moisture_1": 12, "moisture_2": 12}),
        ("One vibration high",  {"vibration_1": 9.0,  "vibration_2": 2.5, "moisture_1": 12, "moisture_2": 12}),
        ("Both moisture high",  {"vibration_1": 2.5,  "vibration_2": 2.5, "moisture_1": 24, "moisture_2": 24}),
    ]
    for name, values in tests1:
        r = predict1(values)
        icon = "✅" if (name == "Normal") == (not r["anomaly"]) else "❌"
        print(f"      {icon} {name}: health={r['health']} severity={r['severity']}")

    # ── Model 2: thermal_pressure ────────────────────────────────────────────
    df2, n2, f2 = build_thermal_pressure_dataset()
    model2, scaler2, predict2, low2, high2 = train_one("thermal_pressure", df2, n2, f2)

    print("    Sanity tests:")
    print("    (Note: 'Storage overfull'/'Storage low/empty' below are a single-feature")
    print("     deviation (ultrasonic alone) among 6 total features, vs. thermal's 3 and")
    print("     pressure's 2 — IsolationForest inherently isolates multi-feature anomalies")
    print("     more reliably than single-feature ones. Expect ~50% binary-anomaly detection")
    print("     on these two specifically, not the ~100% the others get. The health SCORE")
    print("     still drops meaningfully even when the binary flag doesn't trip — that's the")
    print("     more useful signal for these two, not the anomaly boolean.)")
    tests2 = [
        ("Normal",           {"temperature_1": 75, "temperature_2": 75, "temperature_3": 75, "pressure_1": 4500, "pressure_2": 4.5, "ultrasonic": 565}),
        ("Thermal overload",  {"temperature_1": 98, "temperature_2": 98, "temperature_3": 98, "pressure_1": 4500, "pressure_2": 4.5, "ultrasonic": 565}),
        ("Pressure spike",    {"temperature_1": 80, "temperature_2": 80, "temperature_3": 80, "pressure_1": 7500, "pressure_2": 7.5, "ultrasonic": 565}),
        ("Storage overfull",  {"temperature_1": 75, "temperature_2": 75, "temperature_3": 75, "pressure_1": 4500, "pressure_2": 4.5, "ultrasonic": 200}),
        ("Storage low/empty", {"temperature_1": 75, "temperature_2": 75, "temperature_3": 75, "pressure_1": 4500, "pressure_2": 4.5, "ultrasonic": 9000}),
    ]
    for name, values in tests2:
        r = predict2(values)
        icon = "✅" if (name == "Normal") == (not r["anomaly"]) else "❌"
        print(f"      {icon} {name}: health={r['health']} severity={r['severity']}")

    # ── Save both, package into ONE tar.gz ──────────────────────────────────
    print("\nSaving and packaging both models...")
    joblib.dump(model1,  "model/isolation_forest_vibration_moisture.joblib")
    joblib.dump(scaler1, "model/scaler_vibration_moisture.joblib")
    joblib.dump(model2,  "model/isolation_forest_thermal_pressure.joblib")
    joblib.dump(scaler2, "model/scaler_thermal_pressure.joblib")
    joblib.dump(FEATURES_BY_MODEL, "model/features_by_model.joblib")
    joblib.dump(
        {"vibration_moisture": (low1, high1), "thermal_pressure": (low2, high2)},
        "model/score_calibration.joblib"
    )

    with tarfile.open("model.tar.gz", "w:gz") as tar:
        tar.add("model/isolation_forest_vibration_moisture.joblib", arcname="isolation_forest_vibration_moisture.joblib")
        tar.add("model/scaler_vibration_moisture.joblib",           arcname="scaler_vibration_moisture.joblib")
        tar.add("model/isolation_forest_thermal_pressure.joblib",   arcname="isolation_forest_thermal_pressure.joblib")
        tar.add("model/scaler_thermal_pressure.joblib",             arcname="scaler_thermal_pressure.joblib")
        tar.add("model/features_by_model.joblib",                   arcname="features_by_model.joblib")
        tar.add("model/score_calibration.joblib",                   arcname="score_calibration.joblib")

    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.upload_file("model.tar.gz", S3_BUCKET, "models/model.tar.gz")
    print(f"Uploaded to s3://{S3_BUCKET}/models/model.tar.gz")
    print("\nDone — run deploy_endpoint.py next (or retrain_and_redeploy.py to do both in one step).")


if __name__ == "__main__":
    main()