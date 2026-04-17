"""
backend/app/dependencies.py
Inyección de dependencias con Depends() de FastAPI.
"""

from functools import lru_cache
from fastapi import Depends
from .config import Settings, get_settings
from .services import DataService, TechnicalIndicators, RiskCalculator, PortfolioAnalyzer, AlertasService


# ── Dependencia: configuración ────────────────────────────────
def get_config(settings: Settings = Depends(get_settings)) -> Settings:
    """Inyecta la configuración de la aplicación."""
    return settings


# ── Dependencia: servicio de datos ────────────────────────────
@lru_cache
def _data_service_singleton() -> DataService:
    return DataService()

def get_data_service() -> DataService:
    """Inyecta el servicio de descarga de datos (singleton)."""
    return _data_service_singleton()


# ── Dependencia: indicadores técnicos ─────────────────────────
def get_technical_indicators() -> TechnicalIndicators:
    """Inyecta el calculador de indicadores técnicos."""
    return TechnicalIndicators()


# ── Dependencia: calculador de riesgo ─────────────────────────
def get_risk_calculator(
    data_service: DataService = Depends(get_data_service),
) -> RiskCalculator:
    """Inyecta el calculador de VaR/CVaR."""
    return RiskCalculator(data_service)


# ── Dependencia: analizador de portafolio ─────────────────────
def get_portfolio_analyzer(
    data_service: DataService = Depends(get_data_service),
) -> PortfolioAnalyzer:
    """Inyecta el analizador CAPM y Markowitz."""
    return PortfolioAnalyzer(data_service)


# ── Dependencia: servicio de alertas ──────────────────────────
def get_alertas_service(
    data_service: DataService = Depends(get_data_service),
) -> AlertasService:
    """Inyecta el servicio de alertas y señales."""
    return AlertasService(data_service)