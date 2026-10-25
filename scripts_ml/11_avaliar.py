"""Avalia no split VALIDAÇÃO (nunca no teste): pgvector vs popularidade vs reranker.

Uso: .venv-tf/bin/python scripts_ml/11_avaliar.py [caminho_modelo.h5]
Unidade: par único (original, substituto). Cada ocorrência (receita) gera um ranking dos 30 vizinhos do
original; as métricas das ocorrências são médias dentro do par, depois médias entre pares.
Par não rankeável (substituto fora do top-30) conta 0 em "todos" e fica de fora em "rankeáveis".
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
from collections import Counter, defaultdict

import numpy as np
from tensorflow import keras

import features as F


def metricas(posicao):
    """posicao: rank (1 = topo) do substituto, ou None se fora do top-30 -> (MRR, NDCG@10, R@5, R@10)."""
    if posicao is None: return np.zeros(4)
    # NDCG com 1 item relevante: ganho 1 / log2(rank+1); o ideal é 1, então NDCG = DCG
    return np.array([1 / posicao, 1 / np.log2(posicao + 1) if posicao <= 10 else 0, posicao <= 5, posicao <= 10], float)


def avaliar(modelo, d, nome_modelo=""):
    """Devolve {sistema: (metricas_todos, metricas_rankeaveis)}, n_pares, n_rankeaveis."""
    pop = Counter(s for o, s, r, sp in d["pares"] if sp == "treino")     # frequência do candidato como substituto no TREINO
    por_par = defaultdict(list)
    for o, s, r, sp in d["pares"]:
        if sp == "validacao": por_par[(o, s)].append(r)
    acum = {k: [] for k in ("pgvector", "popularidade", "reranker")}
    rank_ok = []
    for (o, s), receitas in por_par.items():
        top = d["topk"][o]                                               # (30,) já ordenado por distância
        ok = s in top
        rank_ok.append(ok)
        ordens = {"pgvector": list(top),
                  # popularidade: mais frequente primeiro; empate resolvido pela ordem do pgvector (sort estável)
                  "popularidade": sorted(top, key=lambda c: -pop[c])}
        # reranker: um batch de 30 exemplos por ocorrência (mesma receita, mesmo original, 30 candidatos)
        ms = []
        for r in receitas:
            X = F.montar(d, np.full(F.K, r), np.full(F.K, o), top)       # (30, dim_entrada)
            score = modelo.predict(X, verbose=0)[:, 0]                   # (30,) score por candidato
            ms.append(metricas_rank(list(top[np.argsort(-score, kind="stable")]), s))
        for k, ordem in ordens.items():
            acum[k].append(metricas_rank(ordem, s))
        acum["reranker"].append(np.mean(ms, axis=0))
    ok = np.array(rank_ok)
    res = {k: (np.mean(v, axis=0), np.mean(np.array(v)[ok], axis=0)) for k, v in acum.items()}
    return res, len(ok), int(ok.sum())


def metricas_rank(ordem, s):
    return metricas(ordem.index(s) + 1 if s in ordem else None)


if __name__ == "__main__":
    caminho = sys.argv[1] if len(sys.argv) > 1 else str(F.RAIZ / "data" / "reranker.h5")
    d = F.carregar()
    res, n, nr = avaliar(keras.models.load_model(caminho), d)
    print(f"modelo: {caminho} | validação: {n} pares únicos, {nr} rankeáveis ({nr / n:.3f})")
    for rotulo, i in [("TODOS os pares", 0), ("só RANKEÁVEIS", 1)]:
        print(f"\n{rotulo}\n{'sistema':14s} {'MRR':>7s} {'NDCG@10':>8s} {'R@5':>7s} {'R@10':>7s}")
        for k, v in res.items(): print(f"{k:14s} " + " ".join(f"{x:7.3f}" for x in v[i]))
