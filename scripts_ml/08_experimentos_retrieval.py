"""Compara variantes de embedding e de geração de candidatos (recall@30 e @10, validação, por par único).

Uso: .venv/bin/python scripts_ml/08_experimentos_retrieval.py
Em memória (numpy, cosseno exato), sem tocar no banco. Candidatos híbridos usam SÓ pares de treino.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

RAIZ = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("emb", RAIZ / "scripts_ml" / "04_embeddings.py")
emb = importlib.util.module_from_spec(spec); spec.loader.exec_module(emb)

tax = json.load(open(RAIZ / "scripts_ml" / "taxonomia.json", encoding="utf-8"))
vocab = pd.read_csv(RAIZ / "data" / "vocabulario.csv")["ingrediente"].tolist()
idx = {n: i for i, n in enumerate(vocab)}
split = pd.read_csv(RAIZ / "data" / "pares_split.csv")
val = split[split.split == "validacao"]
modelo = SentenceTransformer(emb.MODELO)

# substitutos observados de cada original no TREINO (nunca validação/teste: evitaria vazamento)
obs: dict[str, set] = {}
for r in split[split.split == "treino"].itertuples():
    obs.setdefault(r.original, set()).add(r.substituto)


def candidatos(E, original, k, hibrido):
    s = E @ E[idx[original]]; s[idx[original]] = -9          # cosseno; exclui o próprio
    c = {vocab[i] for i in np.argsort(-s)[:k]}               # top-K vetorial
    return c | obs.get(original, set()) if hibrido else c    # união com os vistos no treino


print(f"{'variante':10} {'candidatos':10} {'r@30':>6} {'|C|@30':>7} {'r@10':>6} {'|C|@10':>7}")
for modo in ["baseline", "taxonomia", "media"]:
    E = emb.textos_embedding(vocab, tax, modo, modelo)
    for hibrido in [False, True]:
        linha = []
        for k in (30, 10):
            cs = [(r.substituto, candidatos(E, r.original, k, hibrido)) for r in val.itertuples()]
            linha += [np.mean([s in c for s, c in cs]), np.mean([len(c) for _, c in cs])]
        print(f"{modo:10} {'hibrido' if hibrido else 'puro':10} {linha[0]:6.3f} {linha[1]:7.1f} {linha[2]:6.3f} {linha[3]:7.1f}")
