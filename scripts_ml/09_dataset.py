"""Monta os índices do dataset do reranker (treino e validação) a partir do banco.

Uso: .venv-tf/bin/python scripts_ml/09_dataset.py
Saída: data/dataset_reranker.npz  (guarda só índices; as features são montadas por features.montar)
Teste NUNCA é lido para montar nada.
`montar_dataset` também é usado pelo laboratório (lab.py), que varia NEG_POR_POS e a semente.
"""
from collections import defaultdict

import numpy as np

import features as F

NEG_POR_POS = 5


def montar_dataset(d: dict, neg_por_pos: int = NEG_POR_POS, seed: int = F.SEED):
    """Devolve (treino, validacao); cada um = [receita, original, candidato, rótulo], arrays int32."""
    rng = np.random.default_rng(seed)

    # substitutos observados de cada original, por split (só treino e validação são usados)
    subs = {"treino": defaultdict(set), "validacao": defaultdict(set)}
    for o, s, r, sp in d["pares"]:
        if sp in subs: subs[sp][o].add(s)

    def montar_split(split: str, proibidos: dict, so_rankeaveis: bool):
        """Cada ocorrência (receita real) vira 1 positivo + `neg_por_pos` negativos sorteados do top-30 do original."""
        R, O, C, Y = [], [], [], []
        for o, s, r, sp in d["pares"]:
            if sp != split: continue
            top = d["topk"][o]
            if so_rankeaveis and s not in top: continue            # na validação, só o que o retrieval alcança
            # hard negatives: vizinhos que o pgvector devolveria, mas que não são substitutos conhecidos do original
            negs = [c for c in top if c not in proibidos[o]]
            negs = rng.choice(negs, size=min(neg_por_pos, len(negs)), replace=False)
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
    return tr, va


if __name__ == "__main__":
    tr, va = montar_dataset(F.carregar())
    np.savez(F.RAIZ / "data" / "dataset_reranker.npz",
             tr_r=tr[0], tr_o=tr[1], tr_c=tr[2], tr_y=tr[3], va_r=va[0], va_o=va[1], va_c=va[2], va_y=va[3])
    for nome, x in [("treino", tr), ("validacao", va)]:
        print(f"{nome}: {len(x[3])} exemplos | positivos {x[3].sum()} | negativos {(1 - x[3]).sum()}")
