# import sys
# from pathlib import Path

# API_DIR = Path(__file__).resolve().parents[1]  # .../api
# sys.path.insert(0, str(API_DIR))

# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from api.app.main import app

@pytest.fixture(scope="session")
def client():
    """
    Client FastAPI utilisé par tous les tests.
    L'API est exécutée en mémoire (pas de serveur).
    """
    return TestClient(app)
