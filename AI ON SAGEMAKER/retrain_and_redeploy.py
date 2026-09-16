"""
retrain_and_redeploy.py — retrain, then redeploy to the SAME endpoint name
==============================================================================
Meant to run on a SageMaker Notebook Instance (or anywhere with an AWS role/
credentials configured). Does three things in sequence:

  1. Runs train_local.py — generates fresh training data, trains BOTH
     per-station-type models, uploads model.tar.gz to S3
  2. If an endpoint named ENDPOINT_NAME already exists, deletes it and waits
     for AWS to fully tear it down (SageMaker won't let you create a new
     endpoint under a name that's still being deleted)
  3. Deploys the freshly trained models under that exact same endpoint name

Endpoint name never changes — nothing downstream (your Lambda's
SAGEMAKER_ENDPOINT env var, any client code) needs updating, ever.

Must be run from the same directory as train_local.py, inference.py.

Setup (one-time):
  pip install sagemaker boto3

Run:
  python retrain_and_redeploy.py
"""

import subprocess
import sys
import time

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

AWS_REGION    = "ap-southeast-2"
S3_BUCKET     = "factory-ai-data-capstone"
MODEL_S3_PATH = f"s3://{S3_BUCKET}/models/model.tar.gz"
ROLE_ARN      = "arn:aws:iam::811162362071:role/service-role/AmazonSageMaker-ExecutionRole-20260517T132971"  # update this
ENDPOINT_NAME = "factory-anomaly-detector"    # existing endpoint — never changes
INSTANCE_TYPE = "ml.t2.medium"
DELETE_WAIT_TIMEOUT_S = 300


def run_training():
    print("=" * 60)
    print("STEP 1/3 — Training (both models)")
    print("=" * 60)
    result = subprocess.run([sys.executable, "train_local.py"])
    if result.returncode != 0:
        raise RuntimeError("train_local.py failed — aborting before touching the endpoint")


def wait_until_gone(sm, endpoint_name, timeout=DELETE_WAIT_TIMEOUT_S):
    start = time.time()
    while time.time() - start < timeout:
        try:
            sm.describe_endpoint(EndpointName=endpoint_name)
            print("   ...still deleting, waiting")
            time.sleep(10)
        except sm.exceptions.ClientError:
            return  # endpoint is gone
    raise TimeoutError(f"Endpoint '{endpoint_name}' still exists after {timeout}s — check the console")


def teardown_existing_endpoint(sm):
    print("\n" + "=" * 60)
    print("STEP 2/3 — Removing existing endpoint (if any)")
    print("=" * 60)
    try:
        desc = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
    except sm.exceptions.ClientError:
        print(f"No existing endpoint named '{ENDPOINT_NAME}' — this will be a first-time deployment.")
        return

    config_name = desc["EndpointConfigName"]
    print(f"Found existing endpoint '{ENDPOINT_NAME}' — deleting it before redeploy...")
    sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
    sm.delete_endpoint_config(EndpointConfigName=config_name)
    wait_until_gone(sm, ENDPOINT_NAME)
    print("Old endpoint fully removed.")


def deploy_fresh(session):
    print("\n" + "=" * 60)
    print("STEP 3/3 — Deploying freshly trained models")
    print("=" * 60)
    model = SKLearnModel(
        model_data=MODEL_S3_PATH,
        role=ROLE_ARN,
        entry_point="inference.py",   # must be in this same directory
        framework_version="1.4-2",
        py_version="py3",
        sagemaker_session=session,
    )
    print(f"Deploying to '{ENDPOINT_NAME}' (this takes a few minutes)...")
    model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
    )
    print(f"\nDone. Endpoint '{ENDPOINT_NAME}' is live with both newly trained models.")


def main():
    session = sagemaker.Session(boto3.Session(region_name=AWS_REGION))
    sm      = boto3.client("sagemaker", region_name=AWS_REGION)

    run_training()
    teardown_existing_endpoint(sm)
    deploy_fresh(session)

    print("\nAll done. This endpoint now bundles TWO models — inference.py picks the right one per request.")
    print("\nTest the vibration_moisture model (Stations 1 & 3):")
    print(f"  aws sagemaker-runtime invoke-endpoint --endpoint-name {ENDPOINT_NAME} \\")
    print("    --content-type application/json \\")
    print('    --body \'{"model":"vibration_moisture","vibration_1":2.5,"vibration_2":2.5,"moisture_1":12,"moisture_2":12}\' \\')
    print("    output.json --region " + AWS_REGION + " && cat output.json")
    print("\nTest the thermal_pressure model (Station 2):")
    print(f"  aws sagemaker-runtime invoke-endpoint --endpoint-name {ENDPOINT_NAME} \\")
    print("    --content-type application/json \\")
    print('    --body \'{"model":"thermal_pressure","temperature_1":75,"temperature_2":75,"temperature_3":75,"pressure_1":4.5,"pressure_2":4.5,"ultrasonic":17}\' \\')
    print("    output.json --region " + AWS_REGION + " && cat output.json")


if __name__ == "__main__":
    main()
