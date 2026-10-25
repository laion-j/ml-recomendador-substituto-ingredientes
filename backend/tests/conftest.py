import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    # `with` dispara o lifespan (carrega modelo e abre o pool).
    with TestClient(app) as c:
        yield c
