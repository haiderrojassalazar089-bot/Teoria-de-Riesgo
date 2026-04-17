"""
backend/app/config.py
Configuración centralizada con BaseSettings.
Las variables se cargan desde el archivo .env
"""

from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── API Keys (opcionales — yfinance no requiere key) ──
    alpha_vantage_key: str = Field(default="demo", description="API key de Alpha Vantage")
    fred_api_key: str      = Field(default="",     description="API key de FRED (Federal Reserve)")

    # ── Activos del portafolio ──
    tickers: list[str] = Field(
        default=["AAPL", "JPM", "XOM", "JNJ", "AMZN"],
        description="Lista de tickers del portafolio"
    )
    benchmark: str = Field(default="^GSPC", description="Ticker del benchmark")
    rf_ticker: str = Field(default="^IRX",  description="Ticker tasa libre de riesgo")

    # ── Parámetros del modelo ──
    default_years: int     = Field(default=3,    ge=1, le=10,  description="Años de historia por defecto")
    var_confidence: float  = Field(default=0.95, ge=0.9, le=0.999, description="Nivel de confianza VaR")
    mc_simulations: int    = Field(default=10000, ge=1000,      description="Simulaciones Montecarlo")
    sma_period: int        = Field(default=20,   ge=5, le=200,  description="Período SMA por defecto")
    ema_period: int        = Field(default=21,   ge=5, le=100,  description="Período EMA por defecto")
    rsi_period: int        = Field(default=14,   ge=5, le=30,   description="Período RSI por defecto")

    # ── Cache ──
    cache_ttl_seconds: int = Field(default=1800, description="TTL del caché en segundos")

    # ── CORS ──
    allowed_origins: list[str] = Field(
        default=["http://localhost:8501", "http://localhost:3000", "*"],
        description="Orígenes permitidos para CORS"
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """Retorna instancia única de Settings (singleton con lru_cache)."""
    return Settings()