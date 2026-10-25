"""Acesso ao banco. Só SQL aqui; nada de modelo de embedding."""
import numpy as np
from psycopg_pool import AsyncConnectionPool


async def listar_receitas(pool: AsyncConnectionPool, limite: int = 50) -> list[dict]:
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT id, titulo, ingredientes FROM receitas WHERE id > 0 ORDER BY id LIMIT %s", (limite,)
        )
        return [{"id": r[0], "titulo": r[1], "ingredientes": r[2]} for r in await cur.fetchall()]


async def embedding_do_ingrediente(pool: AsyncConnectionPool, nome: str) -> np.ndarray | None:
    """Embedding já gravado no banco, ou None se o nome não existe (aí o chamador embeda o texto)."""
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT embedding FROM ingredientes WHERE (lower(nome) = lower(%s) OR lower(nome_en) = lower(%s)) AND embedding IS NOT NULL",   # aceita PT ou EN
            (nome, nome),
        )
        row = await cur.fetchone()
        return row[0].to_numpy() if row else None  # Vector -> ndarray


async def buscar_candidatos(
    pool: AsyncConnectionPool, vetor_busca: np.ndarray, excluir_nome: str, k: int
) -> list[dict]:
    """Recall: top-k ingredientes mais próximos de `vetor_busca` por cosseno (<=>).

    NOTA: esta é a ÚNICA função que decide quem entra como candidato. Outro agente poderá
    trocá-la por uma versão híbrida: pgvector UNIÃO substitutos observados em
    pares_substituicao (split 'treino') para o ingrediente original. Manter a assinatura
    e o formato do retorno (lista de dicts) para o resto da API não mudar.
    """
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT nome, categoria, funcoes, sabores, embedding <=> %(v)s AS distancia, embedding
            FROM ingredientes
            WHERE embedding IS NOT NULL
              AND lower(nome) <> lower(%(excluir)s) AND lower(nome_en) <> lower(%(excluir)s)  -- o próprio ingrediente (PT ou EN) não é substituto de si
            ORDER BY embedding <=> %(v)s              -- mesmo operador do índice HNSW
            LIMIT %(k)s
            """,
            {"v": vetor_busca, "excluir": excluir_nome, "k": k},
        )
        return [
            {"nome": n, "categoria": c, "funcoes": f, "sabores": s,
             "distancia": float(d), "embedding": e.to_list()}
            for n, c, f, s, d, e in await cur.fetchall()
        ]
