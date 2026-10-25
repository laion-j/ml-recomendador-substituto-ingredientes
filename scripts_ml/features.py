"""Contrato de features do reranker (compartilhado por 09-13). O front implementa EXATAMENTE a mesma ordem.

Vetor por exemplo (receita r, original o, candidato c):
  [ r(384) | o(384) | c(384) | o*c(384) | |o-c|(384) | dist_cos(1) | cat(c) | funcoes(c) | sabores(c) ]
"""
import os
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

RAIZ = Path(__file__).resolve().parent.parent
K = 30
SEED = 42


def conectar() -> psycopg.Connection:
    # Defaults do docker-compose.yml; sobrescreva por variável de ambiente se mudar.
    con = psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"), port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "ingredientes"), password=os.getenv("POSTGRES_PASSWORD", "ingredientes"),
        dbname=os.getenv("POSTGRES_DB", "ingredientes"))
    register_vector(con)
    return con


def carregar(externos: bool = False) -> dict:
    """Lê o banco uma vez e devolve matrizes numpy indexadas por POSIÇÃO (não pelo id do banco).
    `externos`: inclui as ocorrências da Recipe1MSubs (receita de id negativo, só treino). Desligado por padrão: nos
    testes de 2026-10-25 elas não melhoraram o reranker (ver AGENTS.md)."""
    with conectar() as con:
        ing = con.execute("SELECT id, nome, categoria, funcoes, sabores, embedding FROM ingredientes ORDER BY id").fetchall()
        rec = con.execute("SELECT id, embedding FROM receitas WHERE embedding IS NOT NULL ORDER BY id").fetchall()
        pares = con.execute("SELECT original_id, substituto_id, receita_id, split FROM pares_substituicao "
                            "WHERE receita_id IS NOT NULL AND (%s OR receita_id > 0)", (externos,)).fetchall()
    d = {"ing_ids": [r[0] for r in ing], "nomes": [r[1] for r in ing], "rec_ids": [r[0] for r in rec]}
    d["E_ing"] = np.stack([r[5].to_numpy() for r in ing]).astype(np.float32)          # (300, 384)
    d["E_rec"] = np.stack([r[1].to_numpy() for r in rec]).astype(np.float32)          # (n_receitas, 384)
    # vocabulários multi-hot em ordem alfabética (gravada no feature_spec.json)
    d["categorias"] = sorted({r[2] for r in ing if r[2]})
    d["funcoes"] = sorted({f for r in ing for f in r[3]})
    d["sabores"] = sorted({s for r in ing for s in r[4]})
    # multi-hot por ingrediente: matriz (300, |cat|+|fun|+|sab|), já na ordem do contrato (7, 8, 9)
    tags = np.zeros((len(ing), len(d["categorias"]) + len(d["funcoes"]) + len(d["sabores"])), np.float32)
    nc, nf = len(d["categorias"]), len(d["funcoes"])
    for i, r in enumerate(ing):
        if r[2]: tags[i, d["categorias"].index(r[2])] = 1
        for f in r[3]: tags[i, nc + d["funcoes"].index(f)] = 1
        for s in r[4]: tags[i, nc + nf + d["sabores"].index(s)] = 1
    d["tags"] = tags
    # ids do banco -> posição nas matrizes
    pos_i = {x: i for i, x in enumerate(d["ing_ids"])}
    pos_r = {r[0]: i for i, r in enumerate(rec)}
    d["pares"] = [(pos_i[o], pos_i[s], pos_r[r], sp) for o, s, r, sp in pares if r in pos_r]
    # distância de cosseno (1 - cos) entre todos os ingredientes; idêntica à `<=>` do pgvector
    n = d["E_ing"] / np.linalg.norm(d["E_ing"], axis=1, keepdims=True)
    d["dist"] = (1 - n @ n.T).astype(np.float32)                           # (300, 300)
    # top-30 vizinhos de cada ingrediente (sem ele mesmo), do mais próximo ao mais distante.
    # Busca exata aqui; em produção o pgvector (HNSW) devolve o mesmo conjunto, salvo empates.
    dd = d["dist"] + np.diag(np.full(len(ing), np.inf, np.float32))
    d["topk"] = np.argsort(dd, axis=1, kind="stable")[:, :K]               # (300, 30)
    return d


def montar(d: dict, r: np.ndarray, o: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Índices (n,) de receita, original e candidato -> matriz de features (n, dim_entrada) float32."""
    eo, ec = d["E_ing"][o], d["E_ing"][c]                                  # (n, 384) cada
    return np.concatenate([d["E_rec"][r], eo, ec, eo * ec, np.abs(eo - ec),
                           d["dist"][o, c][:, None], d["tags"][c]], axis=1).astype(np.float32)


def feature_spec(d: dict) -> dict:
    """Contrato lido pelo front (feature_spec.json): ordem das colunas, vocabulários multi-hot e dimensão de entrada."""
    nc, nf, ns = len(d["categorias"]), len(d["funcoes"]), len(d["sabores"])
    return {
        "ordem_features": ["receita(384)", "original(384)", "candidato(384)", "original*candidato(384)",
                           "abs(original-candidato)(384)", "distancia_cosseno(1)",
                           f"multi_hot_categoria_candidato({nc})", f"multi_hot_funcoes_candidato({nf})",
                           f"multi_hot_sabores_candidato({ns})"],
        "dim_entrada": 5 * 384 + 1 + nc + nf + ns,
        "dim_embedding": 384,
        "categorias": d["categorias"], "funcoes": d["funcoes"], "sabores": d["sabores"],   # ordem alfabética
        "k_candidatos": K,
        "versao_embedding": "paraphrase-multilingual-MiniLM-L12-v2 / nome EN + 2x taxonomia (en_tax2)",
        "nota": "A saída (sigmoid) é um score de ranking, não uma probabilidade calibrada.",
    }
