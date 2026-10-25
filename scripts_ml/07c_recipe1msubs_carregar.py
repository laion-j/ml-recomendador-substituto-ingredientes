"""Carrega no banco as receitas e os pares da Recipe1MSubs preparados pelo 07b. Rode DEPOIS do 07_receitas.py (ele zera as tabelas).

Uso: .venv/bin/python scripts_ml/07c_recipe1msubs_carregar.py
Marca no banco: receitas com id NEGATIVO (o app só lista id > 0) e pares com fonte='recipe1msubs', sempre split='treino'.
Reexecutável: apaga só o que esta etapa criou antes de inserir.
"""
import importlib.util
import json
from pathlib import Path

from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

RAIZ = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("emb", RAIZ / "scripts_ml" / "04_embeddings.py")
emb = importlib.util.module_from_spec(spec); spec.loader.exec_module(emb)


def main() -> None:
    dados = json.loads((RAIZ / "data" / "recipe1msubs.json").read_text(encoding="utf-8"))
    ids_rec = sorted(int(i) for i in dados["receitas"])

    # Sem título nem passos: o texto é só a lista de ingredientes em PT. O das nossas receitas é "título; ingredientes".
    modelo = SentenceTransformer(emb.MODELO)
    textos = [", ".join(dados["receitas"][str(i)]["ingredientes_pt"]) for i in ids_rec]
    vetores = modelo.encode(textos, normalize_embeddings=True, show_progress_bar=False, batch_size=64)

    with emb.conectar() as con:
        register_vector(con)
        ing = dict(con.execute("SELECT nome_en, id FROM ingredientes").fetchall())
        con.execute("DELETE FROM pares_substituicao WHERE fonte = 'recipe1msubs'")
        con.execute("DELETE FROM receitas WHERE id < 0")
        cur = con.cursor()
        cur.executemany(
            "INSERT INTO receitas (id, titulo, texto_preparo, ingredientes, embedding) VALUES (%s,%s,NULL,%s::jsonb,%s)",
            [(i, f"[Recipe1MSubs {dados['receitas'][str(i)]['gismo_id']}]",
              json.dumps(dados["receitas"][str(i)]["ingredientes_pt"], ensure_ascii=False), v)
             for i, v in zip(ids_rec, vetores)])
        cur.executemany(
            "INSERT INTO pares_substituicao (original_id, substituto_id, receita_id, fonte, split) "
            "VALUES (%s,%s,%s,'recipe1msubs','treino')",
            [(ing[o], ing[s], r) for r, o, s in dados["pares"]])
        for fonte, split, n in con.execute(
                "SELECT fonte, split, count(*) FROM pares_substituicao GROUP BY 1,2 ORDER BY 1,2").fetchall():
            print(f"pares {fonte:13} {split:10} {n}")
        print("receitas (id>0 / id<0):", con.execute(
            "SELECT count(*) FILTER (WHERE id > 0), count(*) FILTER (WHERE id < 0) FROM receitas").fetchone())


if __name__ == "__main__":
    main()
