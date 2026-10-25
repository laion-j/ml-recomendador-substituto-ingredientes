"""Experimento (2026-10-25): a Recipe1MSubs como BASE DE CONHECIMENTO (candidatos + feature), não como mais linhas de treino.

Uso: TF_USE_LEGACY_KERAS=1 .venv-tf/bin/python scripts_ml/15_experimento_kb.py     (~8 min)
Mede nos 503 pares de validação (par fora dos candidatos conta 0, como no 11_avaliar). Resultado em AGENTS.md.
A = atual; B = top-30 + feature da KB; C = candidatos híbridos (top-30 + substitutos que a KB conhece); D = C + feature da KB.
"""
import sys, pickle, collections, importlib.util
from pathlib import Path
R = str(Path(__file__).resolve().parent.parent) + "/"
sys.path.insert(0, R + "scripts_ml")
import numpy as np, pandas as pd
import lab as L, features as F
from tensorflow import keras
d = F.carregar()
with F.conectar() as con: en = [r[0] for r in con.execute("SELECT nome_en FROM ingredientes ORDER BY id")]
pos = {n: i for i, n in enumerate(en)}; N = len(en)
m = L.script("02_minerar_pares"); norm = lambda s: m.normalizar(m.TOKEN.findall(s.replace("_", " ").lower()))
class S(pickle.Unpickler):
    def find_class(self, a, b): raise pickle.UnpicklingError(b)
KB = np.zeros((N, N), np.float32)
for f in ("train", "val", "test"):
    for x in S(open(R + f"data/raw/gismo/{f}_comments_subs.pkl", "rb")).load():
        o, s = norm(x["subs"][0]), norm(x["subs"][1])
        if o in pos and s in pos and o != s and o not in s and s not in o: KB[pos[o], pos[s]] += 1
KCAP = 30
kbtop = [[c for c in np.argsort(-KB[o])[:KCAP] if KB[o, c] > 0] for o in range(N)]
cand_top = [list(d["topk"][o]) for o in range(N)]
cand_hib = [list(dict.fromkeys(cand_top[o] + [int(c) for c in kbtop[o]])) for o in range(N)]
def feats(r, o, c, kb):
    X = F.montar(d, r, o, c)
    if kb: X = np.concatenate([X, np.log1p(KB[o, c])[:, None], np.log1p(KB[c, o])[:, None]], axis=1)
    return X.astype(np.float32)
subs = {"treino": collections.defaultdict(set), "validacao": collections.defaultdict(set)}
for o, s, r, sp in d["pares"]:
    if sp in subs and d["rec_ids"][r] > 0: subs[sp][o].add(s)      # SÓ os nossos pares (a Recipe1MSubs é só base de conhecimento)
proib_val = collections.defaultdict(set)
for sp in subs:
    for o, ss in subs[sp].items(): proib_val[o] |= ss
def dataset(split, cands, kb, rng, proib, so_rank):
    Rr, O, C, Y = [], [], [], []
    for o, s, r, sp in d["pares"]:
        if sp != split or d["rec_ids"][r] < 0: continue
        if so_rank and s not in cands[o]: continue
        negs = [c for c in cands[o] if c not in proib[o]]
        negs = rng.choice(negs, size=min(5, len(negs)), replace=False)
        for c, y in [(s, 1)] + [(c, 0) for c in negs]: Rr.append(r); O.append(o); C.append(c); Y.append(y)
    Rr, O, C, Y = map(np.array, (Rr, O, C, Y))
    return feats(Rr, O, C, kb), Y.astype(np.float32)
def metr(ordem, s):
    if s not in ordem: return np.zeros(4)
    p = ordem.index(s) + 1
    return np.array([1 / p, 1 / np.log2(p + 1) if p <= 10 else 0, p <= 5, p <= 10], float)
por_par = collections.defaultdict(list)
for o, s, r, sp in d["pares"]:
    if sp == "validacao": por_par[(o, s)].append(r)
def avaliar(modelo, cands, kb, regra=None):
    acum = []
    for (o, s), recs in por_par.items():
        cs = np.array(cands[o]); ms = []
        for r in recs:
            if regra: ordem = regra(o, cs)
            else:
                sc = modelo.predict(feats(np.full(len(cs), r), np.full(len(cs), o), cs, kb), verbose=0, batch_size=512)[:, 0]
                ordem = list(cs[np.argsort(-sc, kind="stable")])
            ms.append(metr(ordem, s))
            if regra: break
        acum.append(np.mean(ms, axis=0))
    return np.mean(acum, axis=0)
pop = collections.Counter(s for o, s, r, sp in d["pares"] if sp == "treino" and d["rec_ids"][r] > 0)
def fmt(v): return f"MRR {v[0]:.3f} NDCG@10 {v[1]:.3f} R@5 {v[2]:.3f} R@10 {v[3]:.3f}"
print("== baselines (503 pares de validação, todos) ==", flush=True)
print("pgvector top-30      ", fmt(avaliar(None, cand_top, False, lambda o, cs: list(cs))), flush=True)
print("popularidade top-30  ", fmt(avaliar(None, cand_top, False, lambda o, cs: sorted(cs, key=lambda c: -pop[c]))), flush=True)
print("so Recipe1MSubs (hib)", fmt(avaliar(None, cand_hib, False, lambda o, cs: sorted(cs, key=lambda c: (-KB[o, c], d['dist'][o, c])))), flush=True)
print("KB + popularidade    ", fmt(avaliar(None, cand_hib, False, lambda o, cs: sorted(cs, key=lambda c: (-KB[o, c] - 0.05 * pop[c], d['dist'][o, c])))), flush=True)
var = [("A top30, sem KB (atual)", cand_top, False), ("B top30 + feature KB", cand_top, True),
       ("C hibrido, sem feature", cand_hib, False), ("D hibrido + feature KB", cand_hib, True)]
for nome, cands, kb in var:
    rs = []
    for seed in (1, 2, 3):
        keras.utils.set_random_seed(seed); rng = np.random.default_rng(seed)
        Xt, yt = dataset("treino", cands, kb, rng, subs["treino"], False)
        Xv, yv = dataset("validacao", cands, kb, rng, proib_val, True)
        mod = L.construir(Xt.shape[1], dict(camadas=[256, 128, 64], ativacao="relu", dropout=0.3, l2=0.0, otimizador="adam", taxa=1e-3))
        mod.fit(Xt, yt, validation_data=(Xv, yv), epochs=40, batch_size=256, verbose=0,
                callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])
        rs.append(avaliar(mod, cands, kb))
    mu, sd = np.mean(rs, 0), np.std(rs, 0)
    print(f"{nome:26}", " ".join(f"{k} {a:.3f}±{b:.3f}" for k, a, b in zip(("MRR", "NDCG@10", "R@5", "R@10"), mu, sd)), flush=True)
