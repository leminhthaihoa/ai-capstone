"""
fetch_real_data.py — pulls real historical readings from DynamoDB
=====================================================================
Saves one CSV per station-type (vibration_moisture, thermal_pressure) for
train_local.py to use as REAL "normal" training data, instead of purely
synthetic np.random data.

This does NOT replace the synthetic fault scenarios in train_local.py —
IsolationForest is unsupervised and doesn't need labeled faults to train,
but real fault EVENTS may be rare or nonexistent in your operating history
so far. This script only grounds the "normal" distribution in reality;
train_local.py keeps generating synthetic fault examples so the model still
has something to learn the boundary against.

Completely optional and backward-compatible: if you don't run this,
train_local.py falls back to exactly its current synthetic-normal
behavior. Re-run this periodically as more real history accumulates.

Run:
  pip install boto3 pandas --break-system-packages
  python3 fetch_real_data.py
"""

import os
import boto3
import pandas as pd
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from boto3.dynamodb.conditions import Key

AWS_REGION = "ap-southeast-2"
TABLE_NAME = os.environ.get("DYNAMODB_TABLE", "sensor-data")
DAYS_BACK  = 14  # how much history to pull — adjust to how much you actually have

FEATURES_BY_MODEL = {
    "vibration_moisture": ["vibration_1", "vibration_2", "moisture_1", "moisture_2"],
    "thermal_pressure":   ["temperature_1", "temperature_2", "temperature_3", "pressure_1", "pressure_2", "ultrasonic"],
}
STATION_TYPE = {
    "factory1": "vibration_moisture",
    "factory2": "thermal_pressure",
    "factory3": "vibration_moisture",
}

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
table = dynamodb.Table(TABLE_NAME)


def to_float(v):
    return float(v) if isinstance(v, Decimal) else v


def fetch_factory(factory, since_ms):
    """Paginated query — table.query() caps at 1MB per call, so a real
    multi-day pull needs to follow LastEvaluatedKey across pages."""
    items = []
    kwargs = {
        "KeyConditionExpression": Key("factory").eq(factory) & Key("timestamp").gte(since_ms),
    }
    while True:
        result = table.query(**kwargs)
        items.extend(result.get("Items", []))
        if "LastEvaluatedKey" not in result:
            break
        kwargs["ExclusiveStartKey"] = result["LastEvaluatedKey"]
    return items


def main():
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)).timestamp() * 1000)
    rows_by_type = {name: [] for name in FEATURES_BY_MODEL}

    for factory, station_type in STATION_TYPE.items():
        print(f"Fetching {factory} ({station_type}), last {DAYS_BACK} days...")
        items = fetch_factory(factory, since_ms)
        print(f"  {len(items)} raw messages")

        features = FEATURES_BY_MODEL[station_type]
        kept = 0
        for item in items:
            readings = {r["tag"]: to_float(r["value"]) for r in item.get("readings", []) if "tag" in r}
            # Only keep rows where EVERY feature this station-type needs was
            # actually present in this message — a partial row would corrupt
            # the training data with missing/zero-filled values.
            if all(f in readings for f in features):
                rows_by_type[station_type].append({f: readings[f] for f in features})
                kept += 1
        print(f"  {kept} complete rows kept ({len(items) - kept} incomplete/skipped)")

    print()
    for station_type, rows in rows_by_type.items():
        if not rows:
            print(f"WARNING: no complete rows for '{station_type}' — "
                  f"train_local.py will fall back to synthetic normal data for this model.")
            continue
        df = pd.DataFrame(rows)
        path = f"real_data_{station_type}.csv"
        df.to_csv(path, index=False)
        print(f"Wrote {len(df)} rows to {path}")
        print(df.describe())
        print()

    print("Done. Run train_local.py next — it will automatically pick up any "
          "real_data_*.csv files sitting next to it.")


if __name__ == "__main__":
    main()
