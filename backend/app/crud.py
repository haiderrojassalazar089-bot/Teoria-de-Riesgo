"""
backend/app/crud.py
Operaciones CRUD para los modelos ORM
"""
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json

from .database import PrecioCache, ConsultaLog, PortafolioGuardado, PredictionLog


# ══════════════════════════════════════════════════════════════
# PRECIOS CACHE
# ══════════════════════════════════════════════════════════════

def upsert_precios(
    db: Session,
    ticker: str,
    precios: List[Dict],  # [{fecha, open, high, low, close, volume}, ...]
    ttl_hours: int = 24,
):
    """
    Inserta precios nuevos en el caché. Si ya existen, los actualiza.
    Elimina precios más viejos que ttl_hours.
    """
    cutoff = datetime.utcnow() - timedelta(hours=ttl_hours)
    
    # Limpiar caché antiguo
    db.query(PrecioCache).filter(
        and_(
            PrecioCache.ticker == ticker,
            PrecioCache.created_at < cutoff
        )
    ).delete()
    
    # Insertar/actualizar
    for p in precios:
        fecha_dt = p["fecha"] if isinstance(p["fecha"], datetime) else datetime.strptime(p["fecha"], "%Y-%m-%d")
        
        existing = db.query(PrecioCache).filter(
            and_(
                PrecioCache.ticker == ticker,
                PrecioCache.fecha == fecha_dt
            )
        ).first()
        
        if existing:
            existing.open   = p.get("open")
            existing.high   = p.get("high")
            existing.low    = p.get("low")
            existing.close  = p["close"]
            existing.volume = p.get("volume")
            existing.created_at = datetime.utcnow()
        else:
            db.add(PrecioCache(
                ticker = ticker,
                fecha  = fecha_dt,
                open   = p.get("open"),
                high   = p.get("high"),
                low    = p.get("low"),
                close  = p["close"],
                volume = p.get("volume"),
            ))
    
    db.commit()


def get_precios_cached(
    db: Session,
    ticker: str,
    fecha_desde: datetime,
    fecha_hasta: datetime,
) -> List[PrecioCache]:
    """Obtiene precios del caché para un rango de fechas."""
    return db.query(PrecioCache).filter(
        and_(
            PrecioCache.ticker == ticker,
            PrecioCache.fecha >= fecha_desde,
            PrecioCache.fecha <= fecha_hasta,
        )
    ).order_by(PrecioCache.fecha).all()


# ══════════════════════════════════════════════════════════════
# CONSULTAS LOG
# ══════════════════════════════════════════════════════════════

def log_consulta(
    db: Session,
    endpoint: str,
    tickers: Optional[List[str]] = None,
    params: Optional[Dict] = None,
    status: str = "success",
    error_msg: Optional[str] = None,
):
    """Registra una consulta a la API."""
    db.add(ConsultaLog(
        endpoint  = endpoint,
        tickers   = ",".join(tickers) if tickers else None,
        params    = json.dumps(params) if params else None,
        status    = status,
        error_msg = error_msg,
    ))
    db.commit()


def get_consultas_recientes(
    db: Session,
    limit: int = 100,
    endpoint: Optional[str] = None,
) -> List[ConsultaLog]:
    """Obtiene las últimas N consultas."""
    q = db.query(ConsultaLog)
    if endpoint:
        q = q.filter(ConsultaLog.endpoint == endpoint)
    return q.order_by(desc(ConsultaLog.timestamp)).limit(limit).all()


# ══════════════════════════════════════════════════════════════
# PORTAFOLIOS GUARDADOS
# ══════════════════════════════════════════════════════════════

def crear_portafolio(
    db: Session,
    nombre: str,
    tickers: List[str],
    pesos: List[float],
) -> PortafolioGuardado:
    """Crea un nuevo portafolio guardado."""
    port = PortafolioGuardado(
        nombre  = nombre,
        tickers = ",".join(tickers),
        pesos   = ",".join(str(p) for p in pesos),
    )
    db.add(port)
    db.commit()
    db.refresh(port)
    return port


def get_portafolios(db: Session, limit: int = 50) -> List[PortafolioGuardado]:
    """Lista todos los portafolios guardados."""
    return db.query(PortafolioGuardado).order_by(
        desc(PortafolioGuardado.updated_at)
    ).limit(limit).all()


def delete_portafolio(db: Session, portafolio_id: int) -> bool:
    """Elimina un portafolio por ID."""
    port = db.query(PortafolioGuardado).filter(
        PortafolioGuardado.id == portafolio_id
    ).first()
    if port:
        db.delete(port)
        db.commit()
        return True
    return False


# ══════════════════════════════════════════════════════════════
# PREDICTIONS LOG
# ══════════════════════════════════════════════════════════════

def log_prediction(
    db: Session,
    ticker: str,
    fecha_pred: datetime,
    features: Dict,
    prediction: float,
    confidence: float,
) -> PredictionLog:
    """Registra una predicción del modelo ML."""
    pred = PredictionLog(
        ticker        = ticker,
        fecha_pred    = fecha_pred,
        features_json = json.dumps(features),
        prediction    = prediction,
        confidence    = confidence,
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred


def update_prediction_result(
    db: Session,
    prediction_id: int,
    real_value: float,
    correct: bool,
):
    """Actualiza el resultado real de una predicción."""
    pred = db.query(PredictionLog).filter(
        PredictionLog.id == prediction_id
    ).first()
    if pred:
        pred.real_value = real_value
        pred.correct    = correct
        db.commit()


def get_predictions(
    db: Session,
    ticker: Optional[str] = None,
    limit: int = 100,
) -> List[PredictionLog]:
    """Obtiene predicciones del log."""
    q = db.query(PredictionLog)
    if ticker:
        q = q.filter(PredictionLog.ticker == ticker)
    return q.order_by(desc(PredictionLog.timestamp)).limit(limit).all()