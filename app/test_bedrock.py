import boto3

from app.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_SESSION_TOKEN,
    AWS_REGION,
    MODEL_REGISTRY,
)


client = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    aws_session_token=AWS_SESSION_TOKEN,
)

response = client.converse(
    modelId=MODEL_REGISTRY.get("nova-pro"),
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "text": "Say hello in one sentence."
                }
            ],
        }
    ],
)

print(response["output"]["message"]["content"][0]["text"])