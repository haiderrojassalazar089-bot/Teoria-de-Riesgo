"""
backend/app/routers/predict.py
Endpoint /predict — ML para señales de trading
"""

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List
from sqlalchemy.orm import Session
from datetime import datetime


router = APIRouter(prefix="/ml", tags=["Machine Learning"])


# ══════════════════════════════════════════════════════════════
# SCHEMAS PYDANTIC
# ══════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=10, description="Ticker del activo")
    features: List[float] = Field(
        ...,
        min_items=5,
        max_items=5,
        description="Features: [RSI, MACD_hist, ret_5d, ret_10d, vol_20d]"
    )


class PredictResponse(BaseModel):
    ticker: str
    signal: str  # SELL / HOLD / BUY
    signal_code: int  # 0 / 1 / 2
    confidence: float  # probabilidad de la clase predicha
    probabilities: dict  # {SELL: 0.x, HOLD: 0.y, BUY: 0.z}
    model_version: str
    features_received: List[float]


# ══════════════════════════════════════════════════════════════
# ENDPOINT
# ══════════════════════════════════════════════════════════════

@router.post("/predict", response_model=PredictResponse, summary="Predecir señal de trading")
def predict_signal(req: PredictRequest):
    """
    Predice señal de trading (BUY/HOLD/SELL) basado en features técnicos.
    
    **Features esperados (en orden):**
    1. RSI (14 períodos) — rango [0, 100]
    2. MACD histogram — típicamente [-5, +5]
    3. Retorno 5 días — decimal (ej: 0.02 = +2%)
    4. Retorno 10 días — decimal
    5. Volatilidad 20 días — decimal (ej: 0.25 = 25% anual)
    
    **Señales:**
    - SELL (0): retorno futuro esperado < -1%
    - HOLD (1): retorno futuro esperado entre -1% y +1%
    - BUY  (2): retorno futuro esperado > +1%
    """
    try:
        # Importar aquí para evitar error si no está entrenado
        from ..ml.predictor import ModelPredictor
        predictor = ModelPredictor()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e) + " — El modelo ML no está disponible. Entrena primero con 'python -m app.ml.train'."
        )
    
    # Preparar features
    X = np.array(req.features).reshape(1, -1)
    
    # Predecir
    y_pred = int(predictor.predict(X)[0])
    y_proba = predictor.predict_proba(X)[0]
    
    signal_name = predictor.class_names[y_pred]
    confidence = float(y_proba[y_pred])
    
    probabilities = {
        name: round(float(prob), 4)
        for name, prob in zip(predictor.class_names, y_proba)
    }
    
    # Log en BD (si tienes get_db configurado)
    # Aquí lo dejamos comentado para que no falle si no está la BD
    # db = Depends(get_db)
    # log = PredictionLog(...)
    # db.add(log); db.commit()
    
    return PredictResponse(
        ticker=req.ticker,
        signal=signal_name,
        signal_code=y_pred,
        confidence=confidence,
        probabilities=probabilities,
        model_version=predictor.model_version,
        features_received=req.features,
    )


@router.get("/model-info", summary="Información del modelo")
def model_info():
    """Retorna metadatos del modelo ML."""
    try:
        from ..ml.predictor import ModelPredictor
        predictor = ModelPredictor()
        return {
            "model_version": predictor.model_version,
            "feature_names": predictor.feature_names,
            "class_names": predictor.class_names,
            "status": "loaded",
        }
    except FileNotFoundError:
        return {
            "status": "not_trained",
            "message": "Ejecuta 'python -m app.ml.train' para entrenar el modelo."
        }