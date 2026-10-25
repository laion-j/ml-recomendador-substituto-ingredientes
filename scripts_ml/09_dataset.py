"""Monta os índices do dataset do reranker (treino e validação) a partir do banco.

Uso: .venv-tf/bin/python scripts_ml/09_dataset.py
Saída: data/dataset_reranker.npz  (guarda só índices; as features são montadas por features.montar)
Teste NUNCA é lido para montar nada.
"""
from collections import defaultdict

import numpy as np

import features as F

NEG_POR_POS = 5
rng = np.random.default_rng(F.SEED)
d = F.carregar()

# substitutos observados de cada original, por split (só treino e validação são usados)
subs = {"treino": defaultdict(set), "validacao": defaultdict(set)}
for o, s, r, sp in d["pares"]:
    if sp in subs: subs[sp][o].add(s)


def montar_split(split: str, proibidos: dict, so_rankeaveis: bool):
    """Cada ocorrência (receita real) vira 1 positivo + NEG_POR_POS negativos sorteados do top-30 do original."""
    R, O, C, Y = [], [], [], []
    for o, s, r, sp in d["pares"]:
        if sp != split: continue
        top = d["topk"][o]
        if so_rankeaveis and s not in top: continue            # na validação, só o que o retrieval alcança
        # hard negatives: vizinhos que o pgvector devolveria, mas que não são substitutos conhecidos do original
        negs = [c for c in top if c not in proibidos[o]]
        negs = rng.choice(negs, size=min(NEG_POR_POS, len(negs)), replace=False)
        for c, y in [(s, 1)] + [(c, 0) for c in negs]:
            R.append(r); O.append(o); C.append(c); Y.append(y)
    return [np.array(x, np.int32) for x in (R, O, C, Y)]


# treino: negativos não podem ser substitutos do original NO TREINO (positivos de outras receitas incluídos)
tr = montar_split("treino", subs["treino"], so_rankeaveis=False)
# validação (só para a loss de early stopping): como são rótulos de validação, os negativos também evitam
# os substitutos de validação do original, senão um positivo real viraria negativo. Nada disso vai ao treino.
proib_val = defaultdict(set)
for sp in subs:
    for o, ss in subs[sp].items(): proib_val[o] |= ss
va = montar_split("validacao", proib_val, so_rankeaveis=True)

np.savez(F.RAIZ / "data" / "dataset_reranker.npz",
         tr_r=tr[0], tr_o=tr[1], tr_c=tr[2], tr_y=tr[3], va_r=va[0], va_o=va[1], va_c=va[2], va_y=va[3])
for nome, x in [("treino", tr), ("validacao", va)]:
    print(f"{nome}: {len(x[3])} exemplos | positivos {x[3].sum()} | negativos {(1 - x[3]).sum()}")
