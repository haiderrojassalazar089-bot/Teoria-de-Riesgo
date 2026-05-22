"""
tests/conftest.py
Configuración de fixtures para pytest
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base
from app.dependencies import get_db


# ══════════════════════════════════════════════════════════════
# BD EN MEMORIA PARA TESTS AISLADOS
# ══════════════════════════════════════════════════════════════

test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
)

TestSessionLocal = sessionmaker(
    bind=test_engine,
    autocommit=False,
    autoflush=False
)

# Crear todas las tablas en la BD de test
Base.metadata.create_all(bind=test_engine)


def override_get_db():
    """Override del Depends(get_db) para usar BD en memoria."""
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Sobreescribir la dependencia
app.dependency_overrides[get_db] = override_get_db


# ══════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def client():
    """Cliente de prueba para FastAPI."""
    return TestClient(app)


@pytest.fixture
def sample_ticker():
    """Ticker de prueba."""
    return "AAPL"
