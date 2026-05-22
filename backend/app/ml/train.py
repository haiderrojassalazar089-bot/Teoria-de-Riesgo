"""
backend/app/ml/train.py
Entrena un clasificador de señales de trading (BUY / HOLD / SELL)
basado en features técnicos: RSI, MACD, retornos rezagados, volatilidad.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_features_and_labels():
    """
    Construye features sintéticos para demostración.
    En producción, estos vendrían de yfinance + indicadores técnicos reales.
    
    Features:
    - RSI (14)
    - MACD histogram
    - Retorno 5 días
    - Retorno 10 días
    - Volatilidad 20 días
    
    Label:
    - 0: SELL (retorno futuro < -1%)
    - 1: HOLD (retorno futuro entre -1% y +1%)
    - 2: BUY  (retorno futuro > +1%)
    """
    np.random.seed(42)
    n_samples = 2000
    
    # Features sintéticos
    rsi = np.random.uniform(20, 80, n_samples)
    macd_hist = np.random.normal(0, 2, n_samples)
    ret_5d = np.random.normal(0, 0.02, n_samples)
    ret_10d = np.random.normal(0, 0.03, n_samples)
    vol_20d = np.random.uniform(0.10, 0.40, n_samples)
    
    X = np.column_stack([rsi, macd_hist, ret_5d, ret_10d, vol_20d])
    
    # Labels: correlación artificial con RSI y retornos
    future_ret = 0.001 * (rsi - 50) + 0.5 * ret_5d + 0.3 * ret_10d + np.random.normal(0, 0.01, n_samples)
    
    y = np.zeros(n_samples, dtype=int)
    y[future_ret < -0.01] = 0  # SELL
    y[(future_ret >= -0.01) & (future_ret <= 0.01)] = 1  # HOLD
    y[future_ret > 0.01] = 2  # BUY
    
    return X, y


def main():
    print("=" * 60)
    print("ENTRENAMIENTO DEL MODELO ML — RiskLab USTA")
    print("=" * 60)
    
    # 1. Construir features y labels
    print("\n[1/5] Construyendo features...")
    X, y = build_features_and_labels()
    print(f"  → {X.shape[0]} muestras, {X.shape[1]} features")
    print(f"  → Distribución de clases: SELL={sum(y==0)}, HOLD={sum(y==1)}, BUY={sum(y==2)}")
    
    # 2. Split temporal (shuffle=False para series financieras)
    print("\n[2/5] Dividiendo train/test (80/20, sin shuffle)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    print(f"  → Train: {len(X_train)} | Test: {len(X_test)}")
    
    # 3. Pipeline: Scaler + Random Forest
    print("\n[3/5] Entrenando Random Forest...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    print("  → Modelo entrenado ✓")
    
    # 4. Evaluación
    print("\n[4/5] Evaluando en test set...")
    y_pred = pipeline.predict(X_test)
    
    print("\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["SELL", "HOLD", "BUY"],
        digits=4
    ))
    
    print("\n  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # 5. Guardar modelo
    model_path = Path(__file__).parent / "model.joblib"
    print(f"\n[5/5] Guardando modelo en {model_path}...")
    joblib.dump(pipeline, model_path)
    print("  → Modelo guardado ✓")
    
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETO")
    print("=" * 60)
    print(f"\nPara usar el modelo, inicia el servidor FastAPI.")
    print(f"El endpoint /predict cargará automáticamente {model_path.name}")


if __name__ == "__main__":
    main()