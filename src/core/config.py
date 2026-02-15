from pydantic import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    OPENAI_API_KEY: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create a single, reusable instance of the settings
settings = Settings()
