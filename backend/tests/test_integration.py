"""
tests/test_integration.py
Tests de integración — endpoints con TestClient
"""

import pytest


# ══════════════════════════════════════════════════════════════
# TEST 4: GET /precios/{ticker} retorna 200 y schema correcto
# ══════════════════════════════════════════════════════════════

def test_get_precios_success(client, sample_ticker):
    """
    Verifica que GET /precios/{ticker} retorna 200 y estructura correcta.
    """
    response = client.get(f"/precios/{sample_ticker}")
    
    assert response.status_code == 200, \
        f"Esperado 200, obtenido {response.status_code}"
    
    data = response.json()
    
    # Verificar estructura del response
    assert "ticker" in data
    assert "precios" in data
    assert isinstance(data["precios"], list)
    
    # Si hay precios, verificar estructura de cada item
    if len(data["precios"]) > 0:
        precio = data["precios"][0]
        assert "fecha" in precio
        assert "close" in precio
        assert "volume" in precio


# ══════════════════════════════════════════════════════════════
# TEST 5: POST /var con pesos incorrectos retorna 422
# ══════════════════════════════════════════════════════════════

def test_var_invalid_weights(client):
    """
    Verifica que POST /var con pesos que NO suman 1 retorna HTTP 422.
    """
    # Payload con pesos que suman 1.2 (incorrecto)
    payload = {
        "tickers": ["AAPL", "MSFT"],
        "weights": [0.7, 0.5],  # Suma 1.2, debería ser 1.0
        "confidence": 0.95,
        "years": 2
    }
    
    response = client.post("/var", json=payload)
    
    # Debe retornar 422 (Unprocessable Entity) por validación fallida
    # O 400 si la validación está en el servicio
    assert response.status_code in [400, 422], \
        f"Esperado 400 o 422, obtenido {response.status_code}"


# ══════════════════════════════════════════════════════════════
# TEST ADICIONAL: GET / health check
# ══════════════════════════════════════════════════════════════

def test_root_endpoint(client):
    """
    Verifica que el endpoint raíz retorna 200 y status ok.
    """
    response = client.get("/")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


# ══════════════════════════════════════════════════════════════
# TEST ADICIONAL: GET /activos retorna estructura correcta
# ══════════════════════════════════════════════════════════════

def test_get_activos(client):
    """
    Verifica que GET /activos retorna lista de activos.
    """
    response = client.get("/activos?tickers=AAPL,MSFT")
    
    assert response.status_code == 200
    
    data = response.json()
    assert "activos" in data
    assert isinstance(data["activos"], list)
