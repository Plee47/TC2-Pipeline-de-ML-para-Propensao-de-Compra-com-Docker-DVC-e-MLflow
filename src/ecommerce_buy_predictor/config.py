from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "online_shoppers_intention"
    kaggle_username: str = ""
    kaggle_key: str = ""
    data_raw_path: str = "data/raw/online_shoppers_intention.csv"
    data_processed_path: str = "data/processed"
    #: Directory holding the local copy of the best model (DVC output).
    model_dir: str = "models"
    #: Name the model is registered under in the MLflow Model Registry.
    model_name: str = "online_shoppers_intention"
    #: Registry alias pointing at the serving version (replaces stages).
    model_alias: str = "champion"
    random_seed: int = 42
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"


settings = Settings()
