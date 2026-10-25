# AGENTS.md

Projeto de estudo: recomendador de substitutos de ingredientes por contexto de receita.
Objetivo real: aprender ML com TensorFlow.js + banco vetorial. O código será relido para estudo.

## Padrão de código (obrigatório)

- Breve e direto. Sem abstração que não se paga.
- Comentários explicativos em PT-BR, focados no "porquê" e em cada transformação de dado
  (texto -> vetor, vetor -> tensor, tensor -> score).
- Um arquivo = uma responsabilidade. Scripts do pipeline numerados (`01_`, `02_`...), rodáveis em ordem.
- Dependências com versão fixa. Sem Google Colab/Kaggle Notebooks: tudo roda local em Python.
- Toda decisão não óbvia (rótulos, features, métricas) é registrada neste arquivo.

## Arquitetura

```
Front (React) -> POST /api/buscar-candidatos -> FastAPI -> pgvector (recall, top 30-50)
                                                   |
Front recebe candidatos + embeddings -> TFJS model.predict() (rerank) -> lista com score
```

- pgvector: recall barato (cosseno, HNSW). TFJS: precisão, usando o contexto da receita.
- A busca parte do **ingrediente faltante** (nome + funções + sabores), não do texto da receita.
  A receita entra como feature do reranker.
- O backend devolve os embeddings (receita, original, candidatos) para o front montar o `tf.tensor2d`.
- Entrada do reranker: `[receita, original, candidato, original*candidato, |original-candidato|,
  distância_pgvector, multi-hot funções/sabores]`.
- Saída (sigmoid) é um **score de ranking**, não probabilidade calibrada.
- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (384 dims, aceita PT), com `normalize_embeddings=True`.
- Treino e inferência usam o mesmo modelo de embedding e o mesmo `feature_spec.json`.

## Decisão: de onde vêm os rótulos

Não existe dataset público pronto e confirmado de substituições. A literatura (Fatemi 2023,
Loesch 2022) constrói os pares a partir de comentários de receitas. Fazemos o mesmo com o Food.com.

1. **Positivos:** minerar reviews (`RAW_interactions.csv`) com padrões como "X instead of Y",
   "substituted X for Y". Casar Y com a lista de ingredientes da receita (isso dá o contexto).
2. **Positivos fracos:** coocorrência (ingredientes em contextos parecidos que nunca aparecem juntos).
3. **Negativos:** hard negatives vindos do top-K do próprio pgvector que não estão nos positivos.
   O reranker deve treinar na mesma distribuição que verá em produção.
4. **Teste:** 100-200 pares conferidos à mão, nunca usados no treino.

Limitações conhecidas: reviews são ruidosas ("used less sugar" não é substituição), tudo em inglês,
pares são direcionais (Y -> X não implica X -> Y).

## Avaliação

- Retrieval: recall@k.
- Reranker: MRR / NDCG.
- Baseline: só pgvector. Se o TFJS não superar o baseline, o reranker não se justifica.

## Dados

- Food.com Recipes and Reviews, slug Kaggle: `shuyangli94/food-com-recipes-and-user-interactions`.
- Download via Python (`kaggle` ou `kagglehub`), token em `~/.kaggle/kaggle.json`
  (aceitar os termos do dataset no site). Dados em `data/` ficam fora do git.
- Mapeamento de `funcoes` (ligante, fermento, gordura...) e `sabores` não existe pronto:
  taxonomia manual para 100-300 ingredientes comuns.

## Estrutura planejada

```
docker-compose.yml     # PostgreSQL + pgvector
db/init.sql            # DDL: receitas, ingredientes, receita_ingrediente, HNSW
scripts_ml/            # 01_baixar -> 02_minerar_pares -> 03_embeddings -> 04_treinar -> 05_exportar
backend/               # FastAPI (psycopg3 async), modelo de embedding carregado no lifespan
frontend/              # React + Vite + tfjs 4.22.0 (inferência em Web Worker, usar tf.tidy)
```

Referência da aula (padrão a seguir no front): `engenharia-de-software-com-ia-aplicada/modulo01.../parte05`
(encode -> concat -> MLP -> sigmoid -> `predict` em batch com `tensor2d` -> ordenar).

## Armadilhas

- `tensorflowjs_converter` costuma ter atrito com TF/Keras 3. Venv com versões fixas e testar
  a exportação cedo com um modelo de brinquedo.
- Validar paridade Keras vs TFJS com os mesmos inputs (teste automatizado).
- Liberar tensores no front (`tf.tidy` / `dispose`).

## Status

- [x] Crítica do prompt e decisão sobre rótulos
- [x] Token do Kaggle configurado (`API_TOKEN_KAGGLE` no `.env`, ignorado pelo git)
- [x] Docker + DDL (pgvector 0.8.7, HNSW em ingredientes e receitas; `docker compose up -d`)
- [x] Download e mineração de pares (27.639 pares, 11.348 únicos; precisão ~90% numa amostra de 50 conferida pelo agente, não por humano)
- [x] Ingredientes e embeddings: 300 ingredientes; embedding = média nome+taxonomia (a variante só-taxonomia deixou 204/300 vetores idênticos); recall@30 = 0,646 na validação (baseline 0,595). Candidatos híbridos com o grafo de treino não ajudam (split por par único). Revisão humana pendente: `taxonomia.json`, `data/teste_manual_candidatos.csv`
- [x] Receitas (7.906) e pares por ocorrência (9.302) no banco; splits por par único sem vazamento
- [x] Backend FastAPI (`backend/`, 8 testes verdes). Porta 8000 ocupada na máquina: usar 8765
- [x] Treino, avaliação, exportação (validação, 237 pares, 153 rankeáveis): o reranker supera pgvector e popularidade; R@10 nos rankeáveis 0,604 vs 0,484 (pgvector) vs 0,353 (popularidade). Uma semente, margem de vários pontos. Early stopping na época 3 (overfit forte). Paridade Keras x TFJS 1,2e-7
- [x] Frontend React + Vite + TFJS (build ok, 4 testes verdes com paridade real)
- [x] Fluxo de dados ponta a ponta verificado em Node (API real -> `features.js` -> `model.json` -> ranking)
- [ ] Teste no navegador (UI + Web Worker): a extensão do Chrome não estava conectada; nunca foi exercitado
- [ ] Revisão humana do teste manual e só então avaliar o split `teste` (uma única vez)

## Como rodar

```bash
docker compose up -d                                   # Postgres + pgvector
cd backend  && ../.venv/bin/uvicorn app.main:app --port 8765
cd frontend && npm run dev                              # 5173; se ocupada: -- --port 5199
# CORS do backend: CORS_ORIGINS="http://localhost:5199" se mudar a porta do front
```

Pipeline de dados/modelo (ordem): `scripts_ml/01_baixar` -> `02_minerar_pares` -> `03_preparar` -> `04_embeddings` ->
`07_receitas` -> `09_dataset` -> `10_treinar` -> `11_avaliar` -> `12_exportar` -> `13_paridade`.
`04` e `07` truncam e recarregam o banco. Treino/exportação usam `.venv-tf` (tf_keras, não Keras 3).

## Limitações conhecidas (leia antes de confiar nos resultados)

- Os rótulos vêm de reviews: ruído, direção (Y -> X não implica X -> Y) e viés de popularidade
  (ex.: para "milk" o modelo põe butter/cream acima de evaporated milk).
- Scores saturam perto de 1 (0,98+) para vários candidatos: é score de ranking, não probabilidade.
- Ingrediente fora dos 300 do vocabulário (texto livre) devolve vizinhos ruins (ex.: "manteiga de garrafa").
- Só 2.370 pares únicos de treino com os dois lados no vocabulário: ganho do reranker é modesto.
- Node 24 + tfjs-node 4.22: precisa do shim `util.isNullOrUndefined` (em `frontend/vitest.setup.js`
  e `scripts_ml/node_check/shim.cjs`).
- A porta 5173 desta máquina é de outro projeto; a 8000 também.
