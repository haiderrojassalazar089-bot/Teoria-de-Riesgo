"""
backend/app/ml/predictor.py
Singleton para cargar y servir el modelo ML
"""

import joblib
from pathlib import Path
from typing import Any, List
import numpy as np


class ModelPredictor:
    """
    Singleton: garantiza que el modelo se carga UNA sola vez al inicio
    y se reutiliza en todos los requests.
    """
    
    _instance = None
    _model: Any = None
    _version: str = "v1.0.0"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            model_path = Path(__file__).parent / "model.joblib"
            
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Modelo no encontrado en {model_path}. "
                    f"Ejecuta 'python -m app.ml.train' primero."
                )
            
            cls._model = joblib.load(model_path)
            print(f"[ModelPredictor] ✅ Modelo cargado: {type(cls._model).__name__}")
            print(f"[ModelPredictor] Versión: {cls._version}")
        
        return cls._instance
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predice la señal de trading.
        
        Parameters
        ----------
        features : np.ndarray, shape (n_samples, n_features)
            Features: [RSI, MACD_hist, ret_5d, ret_10d, vol_20d]
        
        Returns
        -------
        np.ndarray
            Predicciones: 0=SELL, 1=HOLD, 2=BUY
        """
        return self._model.predict(features)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Probabilidades por clase."""
        return self._model.predict_proba(features)
    
    @property
    def model_version(self) -> str:
        return self._version
    
    @property
    def feature_names(self) -> List[str]:
        return ["RSI", "MACD_hist", "ret_5d", "ret_10d", "vol_20d"]
    
    @property
    def class_names(self) -> List[str]:
        return ["SELL", "HOLD", "BUY"]