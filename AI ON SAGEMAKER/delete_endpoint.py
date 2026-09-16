"""
delete_endpoint.py — tears down the SageMaker real-time inference endpoint
==============================================================================
SageMaker real-time endpoints bill by the hour for as long as they exist,
whether or not you're actively using them — run this whenever you're done
testing/demoing. Run deploy_endpoint.py again any time you need it back.

Note: only the Endpoint itself is billed. The EndpointConfig and Model
resources this also cleans up are free metadata records — deleting them
just keeps your SageMaker console tidy, it isn't required to stop charges.

Setup (one-time):
  pip install boto3

Run:
  python delete_endpoint.py
"""

import boto3

AWS_REGION    = "ap-southeast-2"
ENDPOINT_NAME = "factory-anomaly-detector"   # matches SAGEMAKER_ENDPOINT in ai_inference.py — do not change

sm = boto3.client("sagemaker", region_name=AWS_REGION)

try:
    desc        = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
    config_name = desc["EndpointConfigName"]
except sm.exceptions.ClientError:
    print(f"Endpoint '{ENDPOINT_NAME}' not found — nothing to delete (already torn down?)")
    config_name = None

if config_name:
    sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
    print(f"Deleted endpoint: {ENDPOINT_NAME}  (this stops the hourly billing)")

    sm.delete_endpoint_config(EndpointConfigName=config_name)
    print(f"Deleted endpoint config: {config_name}")

print("Done.")
