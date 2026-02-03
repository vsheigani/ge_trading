from pydantic import BaseSettings


class Settings(BaseSettings):
    database_hostname: str = "localhost"
    database_port: str = "5432"
    database_password: str = "postgres"
    database_name: str = "price_db"
    database_username: str = "postgres"
    finnhub_apikey: str = "bshvnnnrh5r8b9vl73jg"
    polygon_apikey: str = "68IWDMZoiB7twYRbsJjGl1XZ1wxEFpFD"
    # secret_key: str
    # algorithm: str
    # access_token_expire_minutes: int

    class Config:
        env_file = ".env"


settings = Settings()
