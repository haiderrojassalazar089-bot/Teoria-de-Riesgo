"""
backend/app/database.py
SQLAlchemy setup + Modelos ORM para RiskLab
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()


# ══════════════════════════════════════════════════════════════
# MODELOS ORM
# ══════════════════════════════════════════════════════════════

class PrecioCache(Base):
    """
    Caché de precios históricos de yfinance.
    Evita consultas repetidas a Yahoo Finance.
    """
    __tablename__ = "precios_cache"

    id         = Column(Integer, primary_key=True, index=True)
    ticker     = Column(String(10), nullable=False, index=True)
    fecha      = Column(DateTime, nullable=False, index=True)
    open       = Column(Float)
    high       = Column(Float)
    low        = Column(Float)
    close      = Column(Float, nullable=False)
    volume     = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class ConsultaLog(Base):
    """
    Log de todas las consultas a endpoints de la API.
    Para auditoría y análisis de uso.
    """
    __tablename__ = "consultas_log"

    id         = Column(Integer, primary_key=True, index=True)
    endpoint   = Column(String(100), nullable=False, index=True)
    tickers    = Column(String(200))  # lista separada por comas
    params     = Column(Text)         # JSON con parámetros
    timestamp  = Column(DateTime, default=datetime.utcnow, index=True)
    status     = Column(String(20), default="success")  # success | error
    error_msg  = Column(Text)


class PortafolioGuardado(Base):
    """
    Portafolios guardados por el usuario con sus pesos.
    """
    __tablename__ = "portafolios_guardados"

    id         = Column(Integer, primary_key=True, index=True)
    nombre     = Column(String(100), nullable=False)
    tickers    = Column(String(500), nullable=False)  # separados por coma
    pesos      = Column(String(500), nullable=False)  # separados por coma
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PredictionLog(Base):
    """
    Log de predicciones del modelo ML.
    Almacena features, predicción y resultado real para evaluación posterior.
    """
    __tablename__ = "predictions_log"

    id              = Column(Integer, primary_key=True, index=True)
    ticker          = Column(String(10), nullable=False, index=True)
    fecha_pred      = Column(DateTime, nullable=False)  # fecha de la predicción
    features_json   = Column(Text)                      # JSON con features usados
    prediction      = Column(Float)                     # predicción del modelo (ej: 1=sube, 0=baja)
    confidence      = Column(Float)                     # confianza [0-1]
    real_value      = Column(Float)                     # valor real observado (se llena después)
    correct         = Column(Boolean)                   # si la predicción fue correcta
    timestamp       = Column(DateTime, default=datetime.utcnow)


# ══════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════

def get_engine(database_url: str):
    """Crea el engine de SQLAlchemy."""
    return create_engine(
        database_url,
        connect_args={"check_same_thread": False} if "sqlite" in database_url else {},
        echo=False,  # cambiar a True para debug SQL
    )


def get_session_local(engine):
    """Crea la clase SessionLocal."""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(engine):
    """Crea todas las tablas en la BD."""
    Base.metadata.create_all(bind=engine)