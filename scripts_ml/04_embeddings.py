"""Gera embeddings dos ingredientes e carrega ingredientes + pares no Postgres.

DECISÃO (medida com scripts_ml/08_experimentos_retrieval.py, validação, 237 pares únicos):
  variante           recall@30  recall@10
  baseline (nome+tax) 0,595      0,329
  taxonomia           0,650      0,371   <- melhor no papel, MAS descartada (ver abaixo)
  média nome+tax      0,646      0,312   <- ESCOLHIDA (default)
  O nome dominava o vetor (egg -> egg noodle), então misturamos nome e taxonomia.
  Por que não "só taxonomia": ingredientes com a mesma taxonomia viram vetores IDÊNTICOS
  (204 dos 300 ficaram empatados com outro, ex.: butter == unsalted butter, distância 0).
  O reranker não distingue candidatos empatados. O ganho em recall@30 (0,650 vs 0,646) está
  dentro da margem de ±3 pp, então não compensa perder a informação do nome.
  Candidatos híbridos (top-K UNIDO aos substitutos do original vistos no TREINO) NÃO ajudaram:
  recall idêntico e o conjunto cresceu (30 -> ~35). É estrutural: o split é por par único, então
  o substituto de validação nunca está entre os substitutos de treino do mesmo original.
  Por isso não existe candidatos.py. Diferenças pequenas (237 pares, ~±3 pp): ganho modesto.

Uso: .venv/bin/python scripts_ml/04_embeddings.py [--texto baseline|taxonomia|media]   (precisa do `docker compose up -d`)
Entrada: data/pares_limpos.csv, data/vocabulario.csv, scripts_ml/taxonomia.json
Saída:   tabelas `ingredientes` e `pares_substituicao` + data/pares_split.csv
Receitas e pares por ocorrência: 07_receitas.py (rodar depois deste, que zera as tabelas).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

RAIZ = Path(__file__).resolve().parent.parent
MODELO = "paraphrase-multilingual-MiniLM-L12-v2"   # 384 dims, aceita PT
SEED = 42


def textos_embedding(vocab: list[str], taxonomia: dict, modo: str, modelo: SentenceTransformer) -> np.ndarray:
    """texto -> vetor (384 floats, norma 1: cosseno = produto interno). `modo` escolhe o que entra no texto."""
    def tax(n):
        t = taxonomia[n]
        return f"{t['categoria']}; {', '.join(t['funcoes'])}; {', '.join(t['sabores'])}"
    cod = lambda textos: modelo.encode(textos, normalize_embeddings=True, show_progress_bar=False)
    if modo == "baseline":   # nome + taxonomia no mesmo texto (o nome domina)
        return cod([f"{n}; {tax(n)}" for n in vocab])
    if modo == "taxonomia":  # só categoria/funções/sabores: vizinhos por papel culinário
        return cod([tax(n) for n in vocab])
    # média: soma dos dois vetores unitários e renormaliza (volta à norma 1)
    soma = cod(vocab) + cod([tax(n) for n in vocab])
    return soma / np.linalg.norm(soma, axis=1, keepdims=True)


def conectar() -> psycopg.Connection:
    # Defaults do docker-compose.yml; sobrescreva por variável de ambiente se mudar.
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"), port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "ingredientes"), password=os.getenv("POSTGRES_PASSWORD", "ingredientes"),
        dbname=os.getenv("POSTGRES_DB", "ingredientes"),
    )


def atribuir_split(pares: pd.DataFrame) -> pd.DataFrame:
    """80/10/10 por PAR ÚNICO: o mesmo (original, substituto) nunca cai em dois splits (evita vazamento)."""
    u = pares.sort_values(["original", "substituto"]).reset_index(drop=True)  # ordem fixa antes de embaralhar
    ordem = np.random.default_rng(SEED).permutation(len(u))
    n_tr, n_va = int(len(u) * 0.8), int(len(u) * 0.1)
    split = np.empty(len(u), dtype=object)
    split[ordem[:n_tr]] = "treino"
    split[ordem[n_tr:n_tr + n_va]] = "validacao"
    split[ordem[n_tr + n_va:]] = "teste"
    u["split"] = split
    return u


def main(modo: str) -> None:
    taxonomia = json.load(open(RAIZ / "scripts_ml" / "taxonomia.json", encoding="utf-8"))
    vocab = pd.read_csv(RAIZ / "data" / "vocabulario.csv")["ingrediente"].tolist()

    modelo = SentenceTransformer(MODELO)
    emb = textos_embedding(vocab, taxonomia, modo, modelo)   # (300, 384) float32
    print("embeddings:", emb.shape, "| modo:", modo)

    # Só entram pares cujos DOIS lados estão no vocabulário (senão não há embedding para o lado de fora).
    pares = pd.read_csv(RAIZ / "data" / "pares_limpos.csv")
    n_antes = len(pares)
    pares = pares[pares["original"].isin(vocab) & pares["substituto"].isin(vocab)]
    print(f"pares com os dois lados no vocabulário: {len(pares)}/{n_antes} linhas")
    unicos = atribuir_split(pares[["original", "substituto"]].drop_duplicates())
    unicos["ocorrencias"] = unicos.merge(
        pares.groupby(["original", "substituto"]).size().rename("n").reset_index(), on=["original", "substituto"])["n"]
    unicos.to_csv(RAIZ / "data" / "pares_split.csv", index=False)

    with conectar() as con:
        register_vector(con)   # permite passar np.ndarray como tipo `vector`
        con.execute("TRUNCATE pares_substituicao, ingredientes RESTART IDENTITY CASCADE")
        ids = {}
        for nome, vec in zip(vocab, emb):
            t = taxonomia[nome]
            ids[nome] = con.execute(
                "INSERT INTO ingredientes (nome, nome_en, categoria, funcoes, sabores, embedding) "
                "VALUES (%s,%s,%s,%s,%s,%s) RETURNING id",
                (nome, nome, t["categoria"], t["funcoes"], t["sabores"], vec)).fetchone()[0]
        # Uma linha por par único. receita_id fica NULL (receitas ainda não carregadas);
        # a etapa seguinte recarrega uma linha por ocorrência, com a receita.
        con.cursor().executemany(
            "INSERT INTO pares_substituicao (original_id, substituto_id, fonte, split) VALUES (%s,%s,'review',%s)",
            [(ids[r.original], ids[r.substituto], r.split) for r in unicos.itertuples()])
    print(unicos["split"].value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--texto", choices=["baseline", "taxonomia", "media"], default="media")
    main(ap.parse_args().texto)
