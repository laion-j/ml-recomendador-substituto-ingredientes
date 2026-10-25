"""Carrega receitas, receita_ingrediente e recarrega pares_substituicao com UMA linha por ocorrência.

Uso: .venv/bin/python scripts_ml/07_receitas.py   (rodar DEPOIS de 04_embeddings.py; reexecutável)
Entrada: data/pares_limpos.csv, data/pares_split.csv, data/raw/RAW_recipes.csv, tabela `ingredientes`
Saída:   tabelas `receitas`, `receita_ingrediente`, `pares_substituicao` (receita_id preenchido)
O split é herdado do PAR ÚNICO, então ocorrências do mesmo par nunca caem em splits diferentes.
"""
import ast
import json
import importlib.util
from pathlib import Path

import pandas as pd
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

RAIZ = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("emb", RAIZ / "scripts_ml" / "04_embeddings.py")
emb = importlib.util.module_from_spec(spec); spec.loader.exec_module(emb)


def main() -> None:
    split = pd.read_csv(RAIZ / "data" / "pares_split.csv")[["original", "substituto", "split"]]
    # pares_split.csv já só tem pares com os dois lados no vocabulário; o merge herda o split do par único
    oc = pd.read_csv(RAIZ / "data" / "pares_limpos.csv").merge(split, on=["original", "substituto"])
    oc = oc.drop_duplicates(["receita_id", "original", "substituto"])   # mesma receita citando o par 2x = 1 linha
    print(f"ocorrências (par x receita): {len(oc)} em {oc.receita_id.nunique()} receitas")

    rec = pd.read_csv(RAIZ / "data" / "raw" / "RAW_recipes.csv", usecols=["id", "name", "steps", "ingredients"])
    rec = rec[rec["id"].isin(oc.receita_id.unique())].drop_duplicates("id")
    rec["ingredientes"] = rec["ingredients"].map(ast.literal_eval)    # string "['a','b']" -> lista
    rec["passos"] = rec["steps"].map(lambda s: "\n".join(f"{i+1}. {p}" for i, p in enumerate(ast.literal_eval(s))))
    oc = oc[oc.receita_id.isin(rec["id"])]                            # ocorrência sem receita no CSV é descartada
    print(f"receitas encontradas no RAW_recipes: {len(rec)}")

    # Tradução feita por 07a_traduzir_receitas.py. Casamento com o vocabulário usa os ingredientes EN
    # (rec["ingredientes"]); título, passos e lista exibidos/embedados vêm em PT.
    arq_pt = RAIZ / "data" / "receitas_pt.json"
    pt = {int(k): v for k, v in json.load(open(arq_pt, encoding="utf-8")).items()} if arq_pt.exists() else {}
    print("receitas traduzidas:", len(pt), "(sem tradução = fica em inglês)")
    rec["titulo_pt"] = [pt.get(i, {}).get("titulo", n) for i, n in zip(rec["id"], rec["name"])]
    rec["ingr_pt"] = [pt.get(i, {}).get("ingredientes", l) for i, l in zip(rec["id"], rec["ingredientes"])]
    rec["passos_pt"] = [("\n".join(f"{k+1}. {p}" for k, p in enumerate(pt[i]["passos"])) if i in pt else p)
                        for i, p in zip(rec["id"], rec["passos"])]

    # texto -> vetor: "titulo; ingredientes", mesmo modelo e normalização dos ingredientes
    modelo = SentenceTransformer(emb.MODELO)
    textos = [f"{n}; {', '.join(i)}" for n, i in zip(rec["titulo_pt"], rec["ingr_pt"])]
    vetores = modelo.encode(textos, normalize_embeddings=True, show_progress_bar=False, batch_size=64)

    with emb.conectar() as con:
        register_vector(con)
        ids = dict(con.execute("SELECT nome_en, id FROM ingredientes").fetchall())   # chave EN: casa com as receitas e os pares
        # reexecutável: zera só o que este script cria (ingredientes ficam como o 04 deixou)
        con.execute("TRUNCATE pares_substituicao, receita_ingrediente, receitas RESTART IDENTITY CASCADE")
        cur = con.cursor()
        cur.executemany(
            "INSERT INTO receitas (id, titulo, texto_preparo, ingredientes, embedding) VALUES (%s,%s,%s,%s::jsonb,%s)",
            [(int(r.id), r.titulo_pt, r.passos_pt, json.dumps(r.ingr_pt, ensure_ascii=False), v)
             for r, v in zip(rec.itertuples(), vetores)])
        # só ingredientes do vocabulário entram; set() evita repetir (PK composta)
        cur.executemany(
            "INSERT INTO receita_ingrediente (receita_id, ingrediente_id) VALUES (%s,%s)",
            [(int(r.id), ids[n]) for r in rec.itertuples() for n in set(r.ingredientes) if n in ids])
        cur.executemany(
            "INSERT INTO pares_substituicao (original_id, substituto_id, receita_id, fonte, split) VALUES (%s,%s,%s,'review',%s)",
            [(ids[r.original], ids[r.substituto], int(r.receita_id), r.split) for r in oc.itertuples()])

        for t in ["ingredientes", "receitas", "receita_ingrediente", "pares_substituicao"]:
            print(t, con.execute(f"SELECT count(*) FROM {t}").fetchone()[0])
        print("pares por split:", con.execute("SELECT split, count(*) FROM pares_substituicao GROUP BY 1 ORDER BY 1").fetchall())
        print("pares únicos por split:", con.execute(
            "SELECT split, count(*) FROM (SELECT DISTINCT original_id, substituto_id, split FROM pares_substituicao) x GROUP BY 1 ORDER BY 1").fetchall())
        print("pares únicos em >1 split:", con.execute(
            "SELECT count(*) FROM (SELECT original_id, substituto_id FROM pares_substituicao GROUP BY 1,2 HAVING count(DISTINCT split)>1) x").fetchone()[0])
        print("pares cujo original/substituto não está na receita:", con.execute(
            "SELECT count(*) FROM pares_substituicao p WHERE NOT EXISTS (SELECT 1 FROM receita_ingrediente r WHERE r.receita_id=p.receita_id AND r.ingrediente_id=p.original_id)").fetchone()[0], "(original)")


if __name__ == "__main__":
    main()
