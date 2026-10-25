-- Schema do recomendador de substitutos. Roda uma única vez, na criação do volume.

-- Habilita o tipo `vector` e os operadores de distância (<=>, <->, <#>).
CREATE EXTENSION IF NOT EXISTS vector;

-- Dimensão 384 = saída do paraphrase-multilingual-MiniLM-L12-v2.
-- Trocar de modelo de embedding exige recriar as colunas.

CREATE TABLE receitas (
    id            INTEGER PRIMARY KEY,          -- mesmo id do Food.com, facilita o join com os reviews
    titulo        TEXT NOT NULL,
    texto_preparo TEXT,                         -- passos da receita (contexto para o embedding)
    ingredientes  JSONB NOT NULL DEFAULT '[]',  -- lista crua; a versão normalizada está em receita_ingrediente
    embedding     vector(384)                   -- NULL até o script 03 preencher
);

CREATE TABLE ingredientes (
    id        SERIAL PRIMARY KEY,
    nome      TEXT NOT NULL UNIQUE,             -- UNIQUE evita duplicatas e permite upsert por nome
    nome_en   TEXT,                             -- nome original em inglês (dados do Food.com)
    categoria TEXT,                             -- ex.: laticínio, gordura, açúcar
    funcoes   TEXT[] NOT NULL DEFAULT '{}',     -- ex.: {ligante, fermento}; taxonomia manual
    sabores   TEXT[] NOT NULL DEFAULT '{}',     -- ex.: {doce, ácido}; vira multi-hot no reranker
    embedding vector(384)
);

-- N:N entre receita e ingrediente: dá o contexto de cada ingrediente.
CREATE TABLE receita_ingrediente (
    receita_id     INTEGER NOT NULL REFERENCES receitas(id) ON DELETE CASCADE,
    ingrediente_id INTEGER NOT NULL REFERENCES ingredientes(id) ON DELETE CASCADE,
    PRIMARY KEY (receita_id, ingrediente_id)    -- PK composta: um ingrediente não repete na receita
);
-- A PK já indexa por receita_id; este cobre a busca inversa (receitas de um ingrediente).
CREATE INDEX idx_receita_ingrediente_ing ON receita_ingrediente (ingrediente_id);

-- Rótulos de treino/teste. Direcional: original -> substituto não implica o inverso.
CREATE TABLE pares_substituicao (
    id           SERIAL PRIMARY KEY,
    original_id  INTEGER NOT NULL REFERENCES ingredientes(id),
    substituto_id INTEGER NOT NULL REFERENCES ingredientes(id),
    receita_id   INTEGER REFERENCES receitas(id), -- nullable: pares fracos (coocorrência) não têm receita
    fonte        TEXT NOT NULL,                   -- ex.: 'review', 'coocorrencia', 'manual'
    split        TEXT NOT NULL DEFAULT 'treino'
                 CHECK (split IN ('treino', 'validacao', 'teste')), -- teste nunca entra no treino
    CHECK (original_id <> substituto_id)
);

-- Índices vetoriais: HNSW com cosseno.
-- Cosseno: os embeddings são normalizados e o que importa é a direção (semântica), não a norma.
--   Com vetores normalizados, ordenar por <=> equivale a ordenar por produto interno.
--   vector_cosine_ops precisa casar com o operador da consulta (<=>), senão o índice não é usado.
-- HNSW: busca aproximada (ANN) em tempo sublinear, boa recall/velocidade, e não exige
--   treino prévio com dados, ao contrário do IVFFlat (que precisa de dados na criação).
--   Custo: build mais lento e mais memória; aceitável neste volume de dados.
CREATE INDEX idx_ingredientes_embedding ON ingredientes
    USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_receitas_embedding ON receitas
    USING hnsw (embedding vector_cosine_ops);
