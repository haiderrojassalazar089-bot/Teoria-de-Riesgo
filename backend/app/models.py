"""
backend/app/models.py
Modelos Pydantic — request y response de todos los endpoints.
Incluye validación dinámica contra el S&P 500.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ════════════════════════════════════════════════
# REQUEST MODELS
# ════════════════════════════════════════════════

class SP500TickerRequest(BaseModel):
    """Request base que valida tickers contra el S&P 500 en tiempo real."""
    tickers: list[str] = Field(
        description="Lista de tickers — deben cotizar en el S&P 500",
        min_length=1,
        max_length=10,
    )

    @field_validator("tickers")
    @classmethod
    def validate_sp500_tickers(cls, v: list[str]) -> list[str]:
        from .sp500_service import validate_tickers_sp500
        cleaned = [t.strip().upper().replace(".", "-") for t in v]
        ok, invalid = validate_tickers_sp500(cleaned)
        if not ok:
            raise ValueError(
                f"Los siguientes tickers no pertenecen al S&P 500: {invalid}. "
                f"Consulta GET /sp500/tickers para ver la lista completa."
            )
        return cleaned


class PortfolioRequest(SP500TickerRequest):
    """Request para cálculo de VaR/CVaR con tickers dinámicos del S&P 500."""
    weights: list[float] = Field(
        description="Pesos del portafolio (deben sumar 1.0)",
    )
    confidence: float = Field(
        default=0.95,
        ge=0.90, le=0.999,
        description="Nivel de confianza del VaR (ej: 0.95 = 95%)"
    )
    years: int = Field(
        default=3,
        ge=1, le=10,
        description="Años de historia"
    )

    @field_validator("weights")
    @classmethod
    def validate_weights_range(cls, v: list[float]) -> list[float]:
        for w in v:
            if w < 0 or w > 1:
                raise ValueError(f"Cada peso debe estar entre 0 y 1, recibido: {w}")
        return v

    @model_validator(mode="after")
    def validate_tickers_weights_match(self) -> "PortfolioRequest":
        if len(self.tickers) != len(self.weights):
            raise ValueError(
                f"len(tickers)={len(self.tickers)} ≠ len(weights)={len(self.weights)}"
            )
        total = sum(self.weights)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Los pesos deben sumar 1.0 (suma actual: {total:.4f})")
        return self


class FronteraRequest(SP500TickerRequest):
    """Request para frontera eficiente con tickers dinámicos."""
    years: int = Field(default=3, ge=1, le=10)
    n_portfolios: int = Field(default=10000, ge=1000, le=50000)
    target_return: Optional[float] = Field(default=None, ge=0.0, le=5.0)

    @field_validator("tickers")
    @classmethod
    def at_least_two(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError("Se necesitan al menos 2 tickers para la frontera eficiente.")
        return v


class DynamicAnalysisRequest(SP500TickerRequest):
    """Request genérico para análisis con tickers dinámicos."""
    years: int = Field(default=3, ge=1, le=10)
    benchmark: str = Field(default="^GSPC", description="Ticker del benchmark")
    start_date: Optional[str] = Field(
        default=None,
        description="Fecha inicio YYYY-MM-DD (para Máquina del Tiempo)"
    )
    end_date: Optional[str] = Field(
        default=None,
        description="Fecha fin YYYY-MM-DD (para Máquina del Tiempo)"
    )

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def validate_date(cls, v):
        if v is None:
            return v
        from datetime import datetime
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Fecha inválida '{v}', formato requerido: YYYY-MM-DD")
        return v

    @model_validator(mode="after")
    def validate_date_range(self) -> "DynamicAnalysisRequest":
        from datetime import datetime
        if self.start_date and self.end_date:
            s = datetime.strptime(self.start_date, "%Y-%m-%d")
            e = datetime.strptime(self.end_date,   "%Y-%m-%d")
            if s >= e:
                raise ValueError("start_date debe ser anterior a end_date")
            delta = (e - s).days
            if delta < 30:
                raise ValueError("El rango mínimo es 30 días")
            if delta > 3650:
                raise ValueError("El rango máximo es 10 años (3650 días)")
        return self


class MonteCarloRequest(SP500TickerRequest):
    """Request para simulación de Monte Carlo visual."""
    weights: list[float] = Field(description="Pesos del portafolio")
    horizon_days: int = Field(default=252, ge=30, le=1260, description="Horizonte en días hábiles")
    n_simulations: int = Field(default=500, ge=100, le=2000, description="Número de trayectorias")
    years_history: int = Field(default=3, ge=1, le=10)

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v):
        for w in v:
            if w < 0 or w > 1:
                raise ValueError(f"Peso {w} fuera de [0,1]")
        return v

    @model_validator(mode="after")
    def match_lengths(self):
        if len(self.tickers) != len(self.weights):
            raise ValueError("len(tickers) ≠ len(weights)")
        if abs(sum(self.weights) - 1.0) > 0.01:
            raise ValueError(f"Σpesos = {sum(self.weights):.4f} ≠ 1.0")
        return self


class DueloRequest(BaseModel):
    """Request para duelo de dos portafolios."""
    portafolio_a: PortfolioRequest = Field(description="Portafolio A")
    portafolio_b: PortfolioRequest = Field(description="Portafolio B")
    years: int = Field(default=3, ge=1, le=10)


# ════════════════════════════════════════════════
# RESPONSE MODELS (los mismos de antes + nuevos)
# ════════════════════════════════════════════════

class ActivoInfo(BaseModel):
    ticker:  str
    sector:  str
    nombre:  str
    ultimo:  float
    cambio_hoy: float


class ActivosResponse(BaseModel):
    activos:   list[ActivoInfo]
    benchmark: str
    total:     int


class PrecioItem(BaseModel):
    fecha:   str
    open:    float
    high:    float
    low:     float
    close:   float
    volume:  int


class PreciosResponse(BaseModel):
    ticker:     str
    start_date: str
    end_date:   str
    n_dias:     int
    precios:    list[PrecioItem]


class RendimientoStats(BaseModel):
    media_diaria:      float
    media_anualizada:  float
    std_diaria:        float
    std_anualizada:    float
    asimetria:         float
    curtosis:          float
    minimo:            float
    maximo:            float
    jarque_bera_stat:  float
    jarque_bera_p:     float
    n_obs:             int


class RendimientosResponse(BaseModel):
    ticker:         str
    log_returns:    list[float]
    simple_returns: list[float]
    fechas:         list[str]
    stats_log:      RendimientoStats
    stats_simple:   RendimientoStats


class IndicadoresResponse(BaseModel):
    ticker:      str
    fechas:      list[str]
    close:       list[float]
    sma_20:      list[Optional[float]]
    sma_50:      list[Optional[float]]
    ema_21:      list[Optional[float]]
    bb_upper:    list[Optional[float]]
    bb_lower:    list[Optional[float]]
    rsi:         list[Optional[float]]
    macd:        list[Optional[float]]
    macd_signal: list[Optional[float]]
    macd_hist:   list[Optional[float]]
    stoch_k:     list[Optional[float]]
    stoch_d:     list[Optional[float]]


class VaRResponse(BaseModel):
    tickers:             list[str]
    weights:             list[float]
    confidence:          float
    var_parametrico_95:  float
    var_parametrico_99:  float
    var_historico_95:    float
    var_historico_99:    float
    var_montecarlo_95:   float
    var_montecarlo_99:   float
    cvar_95:             float
    cvar_99:             float
    var_anualizado_95:   float
    var_anualizado_99:   float
    distribucion:        list[float]


class CAPMItem(BaseModel):
    ticker:                   str
    sector:                   str
    beta:                     float
    alpha_anual:              float
    r_cuadrado:               float
    retorno_esperado:         float
    clasificacion:            str
    riesgo_sistematico_pct:   float
    riesgo_idiosincratico_pct: float


class CAPMResponse(BaseModel):
    rf_display: str
    rf_annual:  float
    rf_source:  str
    rf_date:    str
    benchmark:  str
    rm_annual:  float
    activos:    list[CAPMItem]


class PortafolioOptimo(BaseModel):
    nombre:      str
    pesos:       list[float]
    retorno:     float
    volatilidad: float
    sharpe:      float


class FronteraResponse(BaseModel):
    tickers:          list[str]
    n_simulaciones:   int
    retornos:         list[float]
    volatilidades:    list[float]
    sharpes:          list[float]
    pesos_simulados:  list[list[float]]
    min_varianza:     PortafolioOptimo
    max_sharpe:       PortafolioOptimo
    objetivo:         Optional[PortafolioOptimo] = None


class AlertaItem(BaseModel):
    ticker:      str
    indicador:   str
    señal:       str
    descripcion: str
    valor:       float
    umbral:      Optional[float] = None


class AlertasResponse(BaseModel):
    fecha:   str
    alertas: list[AlertaItem]
    resumen: dict[str, int]


class MacroIndicador(BaseModel):
    nombre:  str
    valor:   float
    display: str
    fuente:  str
    fecha:   str
    unidad:  str


class MacroResponse(BaseModel):
    tasa_libre_riesgo: MacroIndicador
    benchmark_retorno: MacroIndicador
    benchmark_vol:     MacroIndicador


# ── Nuevos response models ─────────────────────────────────

class SP500TickerInfo(BaseModel):
    ticker: str
    name:   str
    sector: str


class SP500ListResponse(BaseModel):
    total:   int
    tickers: list[SP500TickerInfo]


class MonteCarloResponse(BaseModel):
    tickers:          list[str]
    weights:          list[float]
    horizon_days:     int
    n_simulations:    int
    trayectorias:     list[list[float]]   # [sim_i][dia_j] — valores del portafolio
    fechas_sim:       list[str]
    percentil_5:      list[float]
    percentil_50:     list[float]
    percentil_95:     list[float]
    prob_perdida:     float
    retorno_esperado: float
    var_horizonte:    float


class DueloMetricas(BaseModel):
    nombre:        str
    tickers:       list[str]
    weights:       list[float]
    retorno_anual: float
    volatilidad:   float
    sharpe:        float
    max_drawdown:  float
    var_95:        float
    beta:          float
    alpha:         float
    ganador_metricas: dict[str, str]   # métrica → "A" | "B" | "empate"


class DueloResponse(BaseModel):
    portafolio_a:  DueloMetricas
    portafolio_b:  DueloMetricas
    veredicto:     str   # "A" | "B" | "empate"
    puntos_a:      int
    puntos_b:      int
    resumen:       str


class MaquinaTiempoResponse(BaseModel):
    tickers:        list[str]
    start_date:     str
    end_date:       str
    n_dias:         int
    retornos_norm:  dict[str, list[float]]   # ticker → valores normalizados base 100
    fechas:         list[str]
    estadisticas:   dict[str, dict]          # ticker → stats del período
    benchmark_norm: list[float]
    mejor_activo:   str
    peor_activo:    str


class ErrorResponse(BaseModel):
    error:   str
    detalle: str
    codigo:  int