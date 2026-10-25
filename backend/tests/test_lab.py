"""Laboratório: validação de parâmetros e o fluxo real treinar -> salvar -> ativar -> desfazer.
O fluxo roda o TensorFlow de verdade (um treino mínimo, ~1 min) em pasta temporária: não toca em data/lab nem no modelo ativo."""
import time

import pytest

from app import lab

PEQUENO = {"camadas": [8], "epocas": 2, "paciencia": 0}   # rede minúscula: só queremos ver o fluxo funcionar


@pytest.mark.parametrize("ruim", [
    {"camadas": []}, {"camadas": [2]}, {"camadas": [2000]}, {"epocas": 0}, {"dropout": 0.95}, {"otimizador": "xyz"},
])
def test_parametros_invalidos_viram_422(client, ruim):
    assert client.post("/api/lab/treinos", json=ruim).status_code == 422


def test_id_que_nao_existe_ou_malformado_da_404(client):
    assert client.get("/api/lab/treinos/nao-existe").status_code == 404
    assert client.get("/api/lab/treinos/20990101-000000-abcd").status_code == 404


def test_resumo_dos_dados(client):
    d = client.get("/api/lab/dados?neg_por_pos=5").json()
    assert d["splits"]["treino"]["pares_unicos"] > d["splits"]["teste"]["pares_unicos"]
    assert d["exemplos"]["treino"]["negativos"] > d["exemplos"]["treino"]["positivos"]   # ~5 negativos por positivo
    assert len(d["amostra"]) == 8 and d["dim_entrada"] == 1958
    assert d["splits"]["treino"]["ocorrencias_externas"] == 0          # Recipe1MSubs é opcional e vem desligada


def test_dados_com_recipe1msubs_so_aumenta_o_treino(client):
    base = client.get("/api/lab/dados?neg_por_pos=5").json()
    com = client.get("/api/lab/dados?neg_por_pos=5&externos=true").json()
    assert com["splits"]["treino"]["ocorrencias_externas"] > 0
    assert com["splits"]["treino"]["ocorrencias"] > base["splits"]["treino"]["ocorrencias"]
    # a avaliação não muda: validação e teste são 100% da nossa mineração
    assert com["splits"]["validacao"] == base["splits"]["validacao"] and com["splits"]["teste"] == base["splits"]["teste"]
    assert com["exemplos"]["validacao"] == base["exemplos"]["validacao"]


def test_fluxo_treinar_salvar_ativar_desfazer(client, tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "LAB", tmp_path / "lab")
    ativos = tmp_path / "models"; ativos.mkdir()
    (ativos / "model.json").write_text("antigo"); (ativos / "group1-shard1of1.bin").write_text("antigo")
    monkeypatch.setattr(lab, "MODELOS_ATIVOS", ativos)

    id = client.post("/api/lab/treinos", json=PEQUENO).json()["id"]
    assert client.post("/api/lab/treinos", json=PEQUENO).status_code == 409      # um treino por vez

    for _ in range(240):                                                          # até 4 min
        e = client.get(f"/api/lab/treinos/{id}").json()
        if e["status"] not in lab.EM_ANDAMENTO:
            break
        time.sleep(1)
    assert e["status"] == "concluido", e.get("erro")
    assert len(e["historico"]) == 2 and e["melhor_epoca"] in (1, 2)
    assert e["metricas"]["sistemas"]["reranker"]["todos"]["r10"] >= 0

    assert client.post(f"/api/lab/treinos/{id}/ativar").status_code == 409        # não salvou ainda
    salvo = client.post(f"/api/lab/treinos/{id}/salvar", json={"nome": "meu teste"}).json()["salvo"]
    assert salvo["nome"] == "meu teste" and salvo["paridade_ok"]

    client.post(f"/api/lab/treinos/{id}/ativar").raise_for_status()
    assert (ativos / "model.json").read_text() != "antigo"
    assert client.get("/api/lab/treinos").json()["ativo"]["id"] == id

    client.post("/api/lab/ativo/desfazer").raise_for_status()
    assert (ativos / "model.json").read_text() == "antigo"                         # voltou ao anterior
    assert client.get("/api/lab/treinos").json()["ativo"] is None
