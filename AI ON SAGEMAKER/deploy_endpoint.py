"""
deploy_endpoint.py — creates the SageMaker real-time inference endpoint
===========================================================================
Run this after training (train_local.py) has uploaded a fresh model.tar.gz
to S3. It packages inference.py as the serving script and stands up a
SageMaker endpoint that your Lambda can call via invoke_endpoint().

Must be run from the same directory as inference.py.

Setup (one-time):
  pip install sagemaker boto3

If the endpoint already exists (you're redeploying a retrained model), run
delete_endpoint.py FIRST — this script creates fresh, it does not update
an existing endpoint in place.

Run:
  python deploy_endpoint.py
"""

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

AWS_REGION    = "ap-southeast-2"
S3_BUCKET     = "factory-ai-data-capstone"                    # same bucket your training scripts upload to
MODEL_S3_PATH = f"s3://{S3_BUCKET}/models/model.tar.gz"
ROLE_ARN      = "arn:aws:iam::811162362071:role/service-role/AmazonSageMaker-ExecutionRole-20260517T132971"  # update this
ENDPOINT_NAME = "factory-anomaly-detector"                     # matches SAGEMAKER_ENDPOINT in ai_inference.py — do not change
INSTANCE_TYPE = "ml.t2.medium"                                 # smallest general-purpose instance; fine for this model size

session = sagemaker.Session(boto3.Session(region_name=AWS_REGION))

model = SKLearnModel(
    model_data=MODEL_S3_PATH,
    role=ROLE_ARN,
    entry_point="inference.py",     # must sit next to this script when you run it
    framework_version="1.4-2",
    py_version="py3",
    sagemaker_session=session,
)

print(f"Deploying '{MODEL_S3_PATH}' to endpoint '{ENDPOINT_NAME}'...")
print("(this takes a few minutes — SageMaker is spinning up a hosting instance)")

predictor = model.deploy(
    initial_instance_count=1,
    instance_type=INSTANCE_TYPE,
    endpoint_name=ENDPOINT_NAME,
)

print(f"\nDone. Endpoint '{ENDPOINT_NAME}' is live and billing hourly until you delete it.")
print("This endpoint now bundles TWO models — inference.py picks the right one per request.")
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
print("\nWhen you're done testing/demoing, run delete_endpoint.py to stop the hourly charges.")
