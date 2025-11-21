from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Конфигурация приложения, загружаемая из .env файла.
    """

    BOT_TOKEN: str
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    MAX_HISTORY_MESSAGES: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

