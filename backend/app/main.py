"""API: recall com pgvector. O rerank (TFJS) acontece no front."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool
from sentence_transformers import SentenceTransformer

from . import repositorio
from .lab import router as lab_router
from .schemas import BuscaIn, BuscaOut, ReceitaResumo

MODELO = "paraphrase-multilingual-MiniLM-L12-v2"  # o mesmo do treino/ingestão
# Defaults = docker-compose.yml (só dev local); sobrescreva com DATABASE_URL.
DB_URL = os.getenv("DATABASE_URL", "postgresql://ingredientes:ingredientes@localhost:5432/ingredientes")


async def _registrar_vector(conn):
    # Ensina o psycopg a converter vector <-> numpy em cada conexão do pool.
    await register_vector_async(conn)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carrega o modelo UMA vez (segundos); recarregar por request seria inviável.
    app.state.modelo = SentenceTransformer(MODELO)
    app.state.pool = AsyncConnectionPool(DB_URL, configure=_registrar_vector, open=False)
    await app.state.pool.open(wait=True)
    yield
    await app.state.pool.close()


app = FastAPI(title="Recomendador de substitutos", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    # Dev server do front (Vite). Se a 5173 estiver ocupada, passe CORS_ORIGINS="http://localhost:5199,..."
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(lab_router)


async def embedar(request: Request, texto: str):
    # texto -> vetor 384 normalizado. encode é síncrono e pesado: threadpool para não travar o event loop.
    return await run_in_threadpool(
        request.app.state.modelo.encode, texto, normalize_embeddings=True
    )


@app.get("/api/receitas", response_model=list[ReceitaResumo])
async def receitas(request: Request):
    return await repositorio.listar_receitas(request.app.state.pool, 50)


@app.post("/api/buscar-candidatos", response_model=BuscaOut)
async def buscar_candidatos(corpo: BuscaIn, request: Request):
    pool = request.app.state.pool

    # Vetor de BUSCA = ingrediente faltante. Se existe no banco, reusa o embedding de lá
    # (mesmo texto "nome; categoria; funcoes; sabores" do treino); senão embeda o texto digitado.
    vetor_ing = await repositorio.embedding_do_ingrediente(pool, corpo.ingrediente_faltante)
    if vetor_ing is None:
        vetor_ing = await embedar(request, corpo.ingrediente_faltante)

    # A receita NÃO entra na busca: só vira vetor para o front usar como feature do reranker.
    vetor_rec = await embedar(request, corpo.receita_texto)

    candidatos = await repositorio.buscar_candidatos(
        pool, vetor_ing, corpo.ingrediente_faltante, corpo.k
    )
    return {
        "receita_embedding": vetor_rec.tolist(),
        "ingrediente_embedding": vetor_ing.tolist(),
        "candidatos": candidatos,
    }
