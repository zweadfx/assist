from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    OPENAI_API_KEY: str
    CHROMA_DB_PATH: str = "./data/chroma"
    DATABASE_URL: str = "sqlite:///./data/assist.db"
    SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


# Create a single, reusable instance of the settings
settings = Settings()
