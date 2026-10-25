"""Limpa os pares minerados e define o vocabulário (300 ingredientes mais frequentes).

Uso: .venv/bin/python scripts_ml/03_preparar.py
Entrada: data/pares.csv
Saída:   data/pares_limpos.csv e data/vocabulario.csv (ingrediente, ocorrencias)
Depois, a taxonomia (scripts_ml/taxonomia.json) é rotulada à mão para esse vocabulário.
"""
import re
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
GATILHO = re.compile(r"instead of|in place of|rather than|substitut|subbed|swapped|replaced", re.I)
MAX_POR_PAR = 20   # um par como sugar->splenda não pode dominar o treino
TOP_VOCAB = 300


def hipotetica(review: str) -> bool:
    """Pergunta antes do gatilho ("can I use X instead of Y?" vem depois, mas "Why? I used..." não conta).
    Olhamos só a janela de 100 caracteres antes do gatilho: perguntas longe dele não o afetam."""
    m = GATILHO.search(review)
    return bool(m) and "?" in review[max(0, m.start() - 100):m.start()]


def main() -> None:
    df = pd.read_csv(RAIZ / "data" / "pares.csv")
    antes = len(df)
    print(f"antes: {antes} linhas, {df[['original','substituto']].drop_duplicates().shape[0]} pares únicos")

    # 1) substituto curto demais (ruído de tokenização: "ma", "o")
    f1 = df["substituto"].str.len() >= 3
    # 2) um contém o outro = especificação, não substituição ("pumpkin" -> "pumpkin puree")
    f2 = ~df.apply(lambda r: r["original"] in r["substituto"] or r["substituto"] in r["original"], axis=1)
    # 3) review hipotética (pergunta)
    f3 = ~df["review"].fillna("").map(hipotetica)
    for nome, f in [("substituto < 3 chars", f1), ("contido um no outro", f2), ("pergunta hipotética", f3)]:
        print(f"  removeria por '{nome}': {(~f).sum()}")
    df = df[f1 & f2 & f3]
    print(f"após filtros: {len(df)} linhas")

    # Limite por par: mantém no máx. 20 ocorrências, sorteadas com seed fixa (reprodutível)
    df = df.sample(frac=1, random_state=42).groupby(["original", "substituto"]).head(MAX_POR_PAR)
    df = df.sort_values(["original", "substituto", "receita_id"]).reset_index(drop=True)
    df.to_csv(RAIZ / "data" / "pares_limpos.csv", index=False)
    print(f"depois (limite {MAX_POR_PAR}/par): {len(df)} linhas, "
          f"{df[['original','substituto']].drop_duplicates().shape[0]} pares únicos")

    # Vocabulário: soma de ocorrências como original OU substituto
    freq = pd.concat([df["original"], df["substituto"]]).value_counts().head(TOP_VOCAB)
    freq.rename_axis("ingrediente").reset_index(name="ocorrencias").to_csv(RAIZ / "data" / "vocabulario.csv", index=False)
    print(f"vocabulário: {len(freq)} ingredientes (menor frequência: {freq.iloc[-1]})")


if __name__ == "__main__":
    main()
