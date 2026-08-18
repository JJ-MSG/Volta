import os

from dotenv import load_dotenv

load_dotenv()


AWS_ACCESS_KEY_ID = os.getenv("aws_access_key_id")
AWS_SECRET_ACCESS_KEY = os.getenv("aws_secret_access_key")
AWS_SESSION_TOKEN = os.getenv("aws_session_token")
AWS_REGION = os.getenv("aws_region")
MODEL_REGISTRY = {
    "nova-pro": os.getenv("nova_pro_model_id"),
}
print("MODEL_REGISTRY:", MODEL_REGISTRY)