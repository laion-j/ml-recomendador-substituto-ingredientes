"""Gate da etapa 3: sanidade do banco, vizinhos de alguns ingredientes e recall@30 do retrieval puro.

Uso: .venv/bin/python scripts_ml/05_avaliar_retrieval.py
Baseline do projeto: só pgvector (cosseno), sem reranker.
"""
import importlib.util
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
# o módulo começa com número, então não dá para `import 04_embeddings` direto
spec = importlib.util.spec_from_file_location("emb", RAIZ / "scripts_ml" / "04_embeddings.py")
emb = importlib.util.module_from_spec(spec); spec.loader.exec_module(emb)

K = 30
VIZ = "SELECT nome, embedding <=> (SELECT embedding FROM ingredientes WHERE nome=%s) AS dist " \
      "FROM ingredientes WHERE nome <> %s ORDER BY embedding <=> (SELECT embedding FROM ingredientes WHERE nome=%s) LIMIT %s"

with emb.conectar() as con:
    n, nulos = con.execute("SELECT count(*), count(*) FILTER (WHERE embedding IS NULL) FROM ingredientes").fetchone()
    print(f"ingredientes: {n} | embedding nulo: {nulos}")
    print("splits:", con.execute("SELECT split, count(*) FROM pares_substituicao GROUP BY split ORDER BY 1").fetchall())

    for nome in ["butter", "sugar", "milk", "egg", "soy sauce", "buttermilk", "honey"]:
        # exclui o próprio ingrediente (distância 0) para listar só os candidatos
        rows = con.execute(VIZ, (nome, nome, nome, 5)).fetchall()
        print(f"\n{nome}: " + ", ".join(f"{r[0]} ({r[1]:.3f})" for r in rows))

    # recall@30 por PAR ÚNICO (DISTINCT: o banco tem uma linha por ocorrência): para cada par de validação, o substituto está entre os 30 vizinhos do original?
    # A consulta é feita no banco (índice HNSW), como será em produção.
    pares = con.execute("SELECT DISTINCT o.nome, s.nome FROM pares_substituicao p JOIN ingredientes o ON o.id=p.original_id "
                        "JOIN ingredientes s ON s.id=p.substituto_id WHERE p.split='validacao'").fetchall()
    cache, acertos = {}, 0
    for o, s in pares:
        if o not in cache:
            cache[o] = {r[0] for r in con.execute(VIZ, (o, o, o, K)).fetchall()}
        acertos += s in cache[o]
    print(f"\nrecall@{K} (validação, {len(pares)} pares, candidatos = {n - 1}): {acertos / len(pares):.3f}")
