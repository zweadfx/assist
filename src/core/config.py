from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    OPENAI_API_KEY: str
    CHROMA_DB_PATH: str = "./data/chroma"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


# Create a single, reusable instance of the settings
settings = Settings()
