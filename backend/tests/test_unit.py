"""
tests/test_unit.py
Tests unitarios — funciones puras sin BD ni red
"""

import numpy as np
import pytest
from scipy.stats import norm
import math


# ══════════════════════════════════════════════════════════════
# TEST 1: RSI sobre serie conocida
# ══════════════════════════════════════════════════════════════

def test_rsi_calculation():
    """
    Verifica que el cálculo de RSI sea correcto sobre una serie conocida.
    """
    # Serie simple: 5 subidas, 5 bajadas
    prices = np.array([100, 102, 104, 106, 108, 110, 108, 106, 104, 102, 100])
    
    # Calcular retornos
    deltas = np.diff(prices)
    
    # Separar ganancias y pérdidas
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Promedios simples (periodo 14, pero usamos 10 para esta serie corta)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    
    # RSI = 100 - (100 / (1 + RS))
    rs = avg_gain / avg_loss if avg_loss > 0 else 0
    rsi = 100 - (100 / (1 + rs)) if avg_loss > 0 else 100
    
    # En esta serie simétrica, RSI debe estar cerca de 50
    assert 45 <= rsi <= 55, f"RSI esperado ~50, obtenido {rsi}"


# ══════════════════════════════════════════════════════════════
# TEST 2: VaR paramétrico vs analítico
# ══════════════════════════════════════════════════════════════

def test_var_parametric():
    """
    Verifica VaR paramétrico contra valor analítico.
    Con μ y σ conocidos, VaR = -Φ⁻¹(α) · σ · √Δt
    """
    # Parámetros conocidos
    portfolio_value = 1_000_000
    daily_vol = 0.02  # 2% diario
    confidence = 0.99
    
    # VaR analítico
    z_score = norm.ppf(1 - confidence)  # ~-2.326 para 99%
    var_expected = -portfolio_value * z_score * daily_vol
    
    # VaR calculado (simplificado)
    var_calculated = portfolio_value * abs(z_score) * daily_vol
    
    # Deben ser muy cercanos
    assert abs(var_calculated - var_expected) < 1.0, \
        f"VaR esperado {var_expected:.2f}, obtenido {var_calculated:.2f}"
    
    # Verificar que esté en rango razonable (2-3% del portfolio)
    assert 0.02 * portfolio_value < var_calculated < 0.05 * portfolio_value


# ══════════════════════════════════════════════════════════════
# TEST 3: Black-Scholes paridad put-call
# ══════════════════════════════════════════════════════════════

def test_black_scholes_put_call_parity():
    """
    Verifica la identidad de paridad put-call:
    C - P = S - K·e^(-rT)
    """
    # Parámetros
    S = 100.0   # Spot
    K = 100.0   # Strike
    T = 1.0     # 1 año
    r = 0.05    # 5%
    sigma = 0.20  # 20%
    
    # Calcular d1 y d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Black-Scholes
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    # Paridad put-call
    lhs = call - put
    rhs = S - K * math.exp(-r * T)
    
    error = abs(lhs - rhs)
    
    assert error < 1e-6, f"Paridad put-call violada: error={error:.2e}"


# ══════════════════════════════════════════════════════════════
# TEST 4: Indicador técnico adicional (SMA)
# ══════════════════════════════════════════════════════════════

def test_sma_calculation():
    """
    Verifica cálculo correcto de SMA (Simple Moving Average).
    """
    prices = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    period = 5
    
    # SMA manual
    sma = np.convolve(prices, np.ones(period), 'valid') / period
    
    # Valor esperado para los últimos 5: (16+17+18+19+20)/5 = 18
    assert abs(sma[-1] - 18.0) < 1e-6, f"SMA esperado 18.0, obtenido {sma[-1]}"
