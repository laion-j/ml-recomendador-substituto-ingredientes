"""Gera data/teste_manual_candidatos.csv: 150 pares do split TESTE para conferência humana.

ATENÇÃO: A REVISÃO HUMANA ESTÁ PENDENTE. A coluna `valido` fica vazia até uma pessoa preenchê-la
(1 = substituição real, 0 = ruído). `sugestao_agente` é só palpite do agente e NÃO conta como conferência.
Enquanto `valido` estiver vazia, o teste final não pode ser reportado como "conferido à mão".

Uso: .venv/bin/python scripts_ml/06_teste_manual.py
Entrada: data/pares_split.csv, data/pares_limpos.csv
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
N = 150
SEED = 42

pares = pd.read_csv(RAIZ / "data" / "pares_split.csv")
teste = pares[pares["split"] == "teste"].copy()

# Estratos por frequência: a maioria dos pares aparece 1 vez; sem estratificar, só veríamos a cauda.
teste["estrato"] = pd.cut(teste["ocorrencias"], [0, 1, 3, 10**6], labels=["raro(1)", "medio(2-3)", "frequente(4+)"])
print(teste["estrato"].value_counts().to_string())
# cota igual por estrato (50 cada); se um estrato tiver menos, o resto é completado com os demais
cota = N // 3
partes = [g.sample(min(len(g), cota), random_state=SEED) for _, g in teste.groupby("estrato", observed=True)]
sel = pd.concat(partes)
falta = N - len(sel)
if falta > 0:
    sel = pd.concat([sel, teste.drop(sel.index).sample(falta, random_state=SEED)])

# um exemplo de review por par (o mais curto, mais fácil de conferir)
rev = pd.read_csv(RAIZ / "data" / "pares_limpos.csv")
rev["tam"] = rev["review"].fillna("").str.len()
rev = rev.sort_values("tam").drop_duplicates(["original", "substituto"])[["original", "substituto", "review"]]
sel = sel.merge(rev, on=["original", "substituto"]).rename(columns={"review": "exemplo_review"})
sel["exemplo_review"] = sel["exemplo_review"].str.replace(r"\s+", " ", regex=True).str.slice(0, 400)
sel["valido"] = ""
sel["sugestao_agente"] = ""   # preenchida pelo agente; NÃO é conferência humana
sel = sel.sample(frac=1, random_state=SEED)
sel[["original", "substituto", "exemplo_review", "valido", "sugestao_agente", "estrato"]].to_csv(
    RAIZ / "data" / "teste_manual_candidatos.csv", index=False)
print(f"{len(sel)} candidatos salvos; revisão humana PENDENTE")
