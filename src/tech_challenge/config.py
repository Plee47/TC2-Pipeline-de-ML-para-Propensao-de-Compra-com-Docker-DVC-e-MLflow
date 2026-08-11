from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    kaggle_username: str = ""
    kaggle_key: str = ""
    data_raw_path: str = "data/raw/online_shoppers_intention.csv"
    data_processed_path: str = "data/processed"
    model_registry_uri: str = "models"
    random_seed: int = 42
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
