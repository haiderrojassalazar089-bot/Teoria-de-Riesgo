"""
backend/app/models.py
Modelos Pydantic para request y response de todos los endpoints.
"""

from __future__ import annotations
from typing import Optional
from datetime import date
from pydantic import BaseModel, Field, field_validator, model_validator


# ════════════════════════════════════════════════
# REQUEST MODELS
# ════════════════════════════════════════════════

class PortfolioRequest(BaseModel):
    """Request para cálculo de VaR/CVaR."""
    tickers: list[str] = Field(
        description="Lista de tickers del portafolio",
        min_length=1,
        max_length=20,
        examples=[["AAPL", "JPM", "XOM", "JNJ", "AMZN"]]
    )
    weights: list[float] = Field(
        description="Pesos del portafolio (deben sumar 1.0)",
        examples=[[0.2, 0.2, 0.2, 0.2, 0.2]]
    )
    confidence: float = Field(
        default=0.95,
        ge=0.90, le=0.999,
        description="Nivel de confianza del VaR (ej: 0.95 = 95%)"
    )
    years: int = Field(
        default=3,
        ge=1, le=10,
        description="Años de historia para el cálculo"
    )

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, v: list[str]) -> list[str]:
        for ticker in v:
            if len(ticker) < 1 or len(ticker) > 10:
                raise ValueError(f"Ticker '{ticker}' inválido: debe tener entre 1 y 10 caracteres")
            if not ticker.replace("^", "").replace(".", "").isalnum():
                raise ValueError(f"Ticker '{ticker}' contiene caracteres inválidos")
        return [t.upper() for t in v]

    @field_validator("weights")
    @classmethod
    def validate_weights_range(cls, v: list[float]) -> list[float]:
        for w in v:
            if w < 0 or w > 1:
                raise ValueError("Cada peso debe estar entre 0 y 1")
        return v

    @model_validator(mode="after")
    def validate_tickers_weights_match(self) -> "PortfolioRequest":
        if len(self.tickers) != len(self.weights):
            raise ValueError("El número de tickers debe coincidir con el número de pesos")
        total = sum(self.weights)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Los pesos deben sumar 1.0 (suma actual: {total:.4f})")
        return self


class FronteraRequest(BaseModel):
    """Request para cálculo de frontera eficiente."""
    tickers: list[str] = Field(
        description="Tickers para la optimización",
        min_length=2,
        max_length=20,
        examples=[["AAPL", "JPM", "XOM", "JNJ", "AMZN"]]
    )
    years: int = Field(default=3, ge=1, le=10, description="Años de historia")
    n_portfolios: int = Field(
        default=10000,
        ge=1000, le=50000,
        description="Número de portafolios a simular"
    )
    target_return: Optional[float] = Field(
        default=None,
        ge=0.0, le=5.0,
        description="Rendimiento objetivo anual (opcional, ej: 0.15 = 15%)"
    )

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, v: list[str]) -> list[str]:
        return [t.upper() for t in v]


# ════════════════════════════════════════════════
# RESPONSE MODELS
# ════════════════════════════════════════════════

class ActivoInfo(BaseModel):
    """Información básica de un activo."""
    ticker:  str = Field(description="Símbolo del activo")
    sector:  str = Field(description="Sector económico")
    nombre:  str = Field(description="Nombre de la empresa")
    ultimo:  float = Field(description="Último precio de cierre")
    cambio_hoy: float = Field(description="Cambio porcentual del día")


class ActivosResponse(BaseModel):
    """Response para GET /activos"""
    activos:   list[ActivoInfo] = Field(description="Lista de activos del portafolio")
    benchmark: str = Field(description="Ticker del benchmark")
    total:     int = Field(description="Total de activos")


class PrecioItem(BaseModel):
    """Un punto de precio OHLCV."""
    fecha:   str   = Field(description="Fecha en formato YYYY-MM-DD")
    open:    float = Field(description="Precio de apertura")
    high:    float = Field(description="Precio máximo")
    low:     float = Field(description="Precio mínimo")
    close:   float = Field(description="Precio de cierre ajustado")
    volume:  int   = Field(description="Volumen negociado")


class PreciosResponse(BaseModel):
    """Response para GET /precios/{ticker}"""
    ticker:      str            = Field(description="Ticker consultado")
    start_date:  str            = Field(description="Fecha inicial")
    end_date:    str            = Field(description="Fecha final")
    n_dias:      int            = Field(description="Número de días de datos")
    precios:     list[PrecioItem] = Field(description="Serie de precios OHLCV")


class RendimientoStats(BaseModel):
    """Estadísticas de rendimientos."""
    media_diaria:       float = Field(description="Media diaria")
    media_anualizada:   float = Field(description="Media anualizada (×252)")
    std_diaria:         float = Field(description="Desviación estándar diaria")
    std_anualizada:     float = Field(description="Desviación estándar anualizada")
    asimetria:          float = Field(description="Asimetría (skewness)")
    curtosis:           float = Field(description="Curtosis en exceso")
    minimo:             float = Field(description="Rendimiento mínimo")
    maximo:             float = Field(description="Rendimiento máximo")
    jarque_bera_stat:   float = Field(description="Estadístico Jarque-Bera")
    jarque_bera_p:      float = Field(description="p-valor Jarque-Bera")
    n_obs:              int   = Field(description="Número de observaciones")


class RendimientosResponse(BaseModel):
    """Response para GET /rendimientos/{ticker}"""
    ticker:        str             = Field(description="Ticker consultado")
    log_returns:   list[float]     = Field(description="Serie de log-rendimientos")
    simple_returns: list[float]    = Field(description="Serie de rendimientos simples")
    fechas:        list[str]       = Field(description="Fechas correspondientes")
    stats_log:     RendimientoStats = Field(description="Estadísticas de log-rendimientos")
    stats_simple:  RendimientoStats = Field(description="Estadísticas de rendimientos simples")


class IndicadoresResponse(BaseModel):
    """Response para GET /indicadores/{ticker}"""
    ticker:    str         = Field(description="Ticker consultado")
    fechas:    list[str]   = Field(description="Fechas")
    close:     list[float] = Field(description="Precios de cierre")
    sma_20:    list[Optional[float]] = Field(description="SMA 20 períodos")
    sma_50:    list[Optional[float]] = Field(description="SMA 50 períodos")
    ema_21:    list[Optional[float]] = Field(description="EMA 21 períodos")
    bb_upper:  list[Optional[float]] = Field(description="Banda de Bollinger superior")
    bb_lower:  list[Optional[float]] = Field(description="Banda de Bollinger inferior")
    rsi:       list[Optional[float]] = Field(description="RSI 14 períodos")
    macd:      list[Optional[float]] = Field(description="Línea MACD")
    macd_signal: list[Optional[float]] = Field(description="Línea de señal MACD")
    macd_hist: list[Optional[float]] = Field(description="Histograma MACD")
    stoch_k:   list[Optional[float]] = Field(description="Oscilador estocástico %K")
    stoch_d:   list[Optional[float]] = Field(description="Oscilador estocástico %D")


class VaRResponse(BaseModel):
    """Response para POST /var"""
    tickers:   list[str]  = Field(description="Tickers del portafolio")
    weights:   list[float] = Field(description="Pesos utilizados")
    confidence: float      = Field(description="Nivel de confianza")
    var_parametrico_95:  float = Field(description="VaR paramétrico al 95%")
    var_parametrico_99:  float = Field(description="VaR paramétrico al 99%")
    var_historico_95:    float = Field(description="VaR histórico al 95%")
    var_historico_99:    float = Field(description="VaR histórico al 99%")
    var_montecarlo_95:   float = Field(description="VaR Montecarlo al 95%")
    var_montecarlo_99:   float = Field(description="VaR Montecarlo al 99%")
    cvar_95:             float = Field(description="CVaR (Expected Shortfall) al 95%")
    cvar_99:             float = Field(description="CVaR (Expected Shortfall) al 99%")
    var_anualizado_95:   float = Field(description="VaR paramétrico anualizado al 95%")
    var_anualizado_99:   float = Field(description="VaR paramétrico anualizado al 99%")
    distribucion:        list[float] = Field(description="Distribución de rendimientos del portafolio")


class CAPMItem(BaseModel):
    """CAPM para un activo."""
    ticker:          str   = Field(description="Ticker del activo")
    sector:          str   = Field(description="Sector")
    beta:            float = Field(description="Beta calculado por MCO")
    alpha_anual:     float = Field(description="Alpha de Jensen anualizado")
    r_cuadrado:      float = Field(description="R² de la regresión")
    retorno_esperado: float = Field(description="Retorno esperado anual según CAPM")
    clasificacion:   str   = Field(description="Agresivo / Neutro / Defensivo")
    riesgo_sistematico_pct: float = Field(description="% del riesgo total que es sistemático")
    riesgo_idiosincratico_pct: float = Field(description="% del riesgo total idiosincrático")


class CAPMResponse(BaseModel):
    """Response para GET /capm"""
    rf_display:  str         = Field(description="Tasa libre de riesgo (texto)")
    rf_annual:   float       = Field(description="Tasa libre de riesgo anual")
    rf_source:   str         = Field(description="Fuente de la tasa libre de riesgo")
    rf_date:     str         = Field(description="Fecha de actualización de la Rf")
    benchmark:   str         = Field(description="Benchmark utilizado")
    rm_annual:   float       = Field(description="Rendimiento anualizado del mercado")
    activos:     list[CAPMItem] = Field(description="CAPM por activo")


class PortafolioOptimo(BaseModel):
    """Un portafolio óptimo de la frontera eficiente."""
    nombre:     str        = Field(description="Nombre del portafolio")
    pesos:      list[float] = Field(description="Pesos de cada activo")
    retorno:    float      = Field(description="Retorno esperado anual")
    volatilidad: float     = Field(description="Volatilidad anual")
    sharpe:     float      = Field(description="Ratio de Sharpe")


class FronteraResponse(BaseModel):
    """Response para POST /frontera-eficiente"""
    tickers:         list[str]    = Field(description="Tickers utilizados")
    n_simulaciones:  int          = Field(description="Portafolios simulados")
    retornos:        list[float]  = Field(description="Retornos de portafolios simulados")
    volatilidades:   list[float]  = Field(description="Volatilidades de portafolios simulados")
    sharpes:         list[float]  = Field(description="Sharpes de portafolios simulados")
    pesos_simulados: list[list[float]] = Field(description="Pesos de cada portafolio simulado")
    min_varianza:    PortafolioOptimo  = Field(description="Portafolio de mínima varianza")
    max_sharpe:      PortafolioOptimo  = Field(description="Portafolio de máximo Sharpe")
    objetivo:        Optional[PortafolioOptimo] = Field(
        default=None, description="Portafolio con rendimiento objetivo (si se especificó)"
    )


class AlertaItem(BaseModel):
    """Alerta de señal para un activo."""
    ticker:      str  = Field(description="Ticker del activo")
    indicador:   str  = Field(description="Indicador que generó la señal")
    señal:       str  = Field(description="COMPRA / VENTA / NEUTRAL")
    descripcion: str  = Field(description="Descripción de la señal en lenguaje simple")
    valor:       float = Field(description="Valor actual del indicador")
    umbral:      Optional[float] = Field(default=None, description="Umbral configurado")


class AlertasResponse(BaseModel):
    """Response para GET /alertas"""
    fecha:   str              = Field(description="Fecha de generación de alertas")
    alertas: list[AlertaItem] = Field(description="Lista de alertas activas")
    resumen: dict[str, int]   = Field(description="Resumen: cuántas COMPRA/VENTA/NEUTRAL")


class MacroIndicador(BaseModel):
    """Un indicador macroeconómico."""
    nombre:  str   = Field(description="Nombre del indicador")
    valor:   float = Field(description="Valor actual")
    display: str   = Field(description="Valor formateado para mostrar")
    fuente:  str   = Field(description="Fuente del dato")
    fecha:   str   = Field(description="Fecha de actualización")
    unidad:  str   = Field(description="Unidad de medida")


class MacroResponse(BaseModel):
    """Response para GET /macro"""
    tasa_libre_riesgo: MacroIndicador = Field(description="T-Bill 3M (^IRX)")
    benchmark_retorno: MacroIndicador = Field(description="Retorno anualizado S&P 500")
    benchmark_vol:     MacroIndicador = Field(description="Volatilidad anualizada S&P 500")


# ── Error model ──────────────────────────────────────────────
class ErrorResponse(BaseModel):
    """Response de error estándar."""
    error:   str = Field(description="Tipo de error")
    detalle: str = Field(description="Descripción detallada del error")
    codigo:  int = Field(description="Código HTTP")