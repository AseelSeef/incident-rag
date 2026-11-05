from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/incident_rag"
    # Shown as '**********' when printed; read it with .get_secret_value().
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = "claude-haiku-5-5"


settings = Settings()
