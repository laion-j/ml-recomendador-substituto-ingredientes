"""Testes contra o Postgres real (docker compose up -d) e o modelo real."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    # `with` dispara o lifespan (carrega modelo e abre o pool).
    with TestClient(app) as c:
        yield c


def buscar(client, **kw):
    corpo = {"receita_texto": "bolo de cenoura com cobertura", "ingrediente_faltante": "butter", **kw}
    return client.post("/api/buscar-candidatos", json=corpo)


def test_receitas_tolera_lista_vazia(client):
    r = client.get("/api/receitas")
    assert r.status_code == 200
    lista = r.json()
    assert isinstance(lista, list) and len(lista) <= 50
    for item in lista:
        assert {"id", "titulo", "ingredientes"} <= item.keys()


def test_butter_devolve_k_candidatos(client):
    r = buscar(client, k=10)
    assert r.status_code == 200
    d = r.json()
    assert len(d["receita_embedding"]) == 384
    assert len(d["ingrediente_embedding"]) == 384
    assert len(d["candidatos"]) == 10
    assert all(len(c["embedding"]) == 384 for c in d["candidatos"])
    dist = [c["distancia"] for c in d["candidatos"]]
    assert dist == sorted(dist)  # mais próximo primeiro
    assert "butter" not in [c["nome"].lower() for c in d["candidatos"]]


def test_ingrediente_desconhecido_responde(client):
    r = buscar(client, ingrediente_faltante="manteiga de garrafa", k=5)
    assert r.status_code == 200
    assert len(r.json()["candidatos"]) == 5


@pytest.mark.parametrize("kw", [
    {"receita_texto": ""},
    {"receita_texto": "   "},
    {"ingrediente_faltante": ""},
    {"k": 0},
    {"k": 101},
])
def test_validacao_422(client, kw):
    assert buscar(client, **kw).status_code == 422
