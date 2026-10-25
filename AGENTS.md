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
- [x] Migração para PT-BR (24/10/2026): `ingredientes.nome` em PT (dicionário manual `scripts_ml/traducao_ingredientes.json`, `nome_en` guarda o original e é a chave de pares/receitas), receitas traduzidas por `07a_traduzir_receitas.py` (opus-mt-tc-big-en-pt via CTranslate2, `.venv-trad`, GPU precisa de `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` no LD_LIBRARY_PATH). Backup da versão EN em `../backup-en-2026-10-24/`. RESULTADO PIOR que em EN (mesmo split, 237 pares): rankeáveis 129 (recall@30 0,544; era 153/0,646); R@10 nos rankeáveis reranker 0,611 vs pgvector 0,558 (era 0,604 vs 0,484); MRR reranker 0,221 vs pgvector 0,227 (empate). Causa provável: MiniLM é mais fraco em PT para nomes curtos; não investigado.
- [x] Laboratório de treino (25/10/2026), aba "Laboratório de treino" do front: mostra a separação dos dados, deixa mudar camadas/neurônios/dropout/L2/otimizador/taxa/lote/épocas/paciência/negativos/semente, treina com curva ao vivo, valida (só split VALIDAÇÃO, contra pgvector e popularidade), salva (exporta tfjs + paridade no Node) e ativa no app (o anterior vai para `data/lab/_backup_ativo/`, há "desfazer"). O backend NÃO treina: `backend/app/lab.py` lança `scripts_ml/lab.py` no `.venv-tf` e lê `data/lab/<id>/progresso.json` (o front consulta a cada 1 s). Um treino por vez. `09_dataset.py` virou função (`montar_dataset`) e `features.feature_spec` saiu do `12_exportar.py`. Testado no Chromium (Playwright): treino real de 12 épocas, salvar com paridade OK.
- [ ] Teste no navegador da tela Substituir (UI + Web Worker): nunca foi exercitado (o laboratório foi)
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

## Vocabulário de 1000 ingredientes (2026-10-25)

- `03_preparar.py`: `TOP_VOCAB = 1000`, com lista `RUIDO` (palavras que a mineração confunde com ingrediente: hand, garden, dish...). Resultado: 13.896 ocorrências, 5.030 pares únicos (treino 4.024 / validação 503 / teste 503), 11.243 receitas.
- Rótulos dos 700 novos feitos à mão pelo agente (REVISAR): `scripts_ml/rotulos_extra_*.txt` (en|pt|categoria|funções|sabores), fundidos em `taxonomia.json` e `traducao_ingredientes.json` por `14_rotulos.py`.
- Embedding do ingrediente agora é `en_tax2` (nome EN + 2x taxonomia; `04_embeddings.py`, padrão). Nome PT sozinho dava recall@30 = 0,153; EN + 2x taxonomia, 0,423 (a média antiga nome PT + taxonomia: 0,314). O nome PT é só de exibição. O texto da receita continua em PT.
- Métricas NÃO são comparáveis com as dos 300 ingredientes (são 999 candidatos, não 299; o split é outro). Validação, só rankeáveis (213 de 503): R@10 reranker 0,566 x pgvector 0,526; MRR 0,192 x 0,186 (popularidade 0,191). Melhor época do treino padrão: 2 (overfit rápido).
- Backup do estado de 300 ingredientes em `../backup-pt-300-2026-10-25/` (dump do banco, data/, modelo exportado, taxonomia, tradução).
- `05_avaliar_retrieval.py` foi corrigido para consultar por `nome_en` (estava quebrado desde a migração PT).

## Recipe1MSubs (GISMo, Meta): investigação

- Arquivos públicos, sem cadastro: `https://dl.fbaipublicfiles.com/gismo/{train,val,test}_comments_subs.pkl` (+ `vocab_ingrs.pkl`, `merge_dict.pkl`). Código do GISMo é CC-BY-NC; a licença dos dados não está declarada (os dados vêm do Recipe1M, de uso acadêmico/não comercial).
- Formato: lista de dicts `{id, ingredients: [[variantes de nome]...], subs: (original, substituto)}`. 70.520 itens, 30.832 pares únicos, 5.156 ingredientes (nomes como `olive_oil`). Os 3 `.pkl` de substituição não têm imports (conferido com `pickletools`); `vocab_ingrs.pkl` referencia uma classe `Vocabulary` e NÃO deve ser carregado. Para ler: `pickle.Unpickler` com `find_class` que sempre recusa.
- Sem título, passos nem texto de review, só a lista de ingredientes da receita. O id é hash do Recipe1M (a receita completa exigiria cadastro lá). Para o reranker, o contexto da receita teria que ser só "ingredientes".
- Cobertura sobre nosso vocabulário: com 300 ingredientes, 25.546 itens / 4.293 pares únicos (nossa base: 9.302 / 2.370); com 1000, 38.279 itens / 10.348 pares únicos.

## Recipe1MSubs integrada: o que funcionou e o que não (2026-10-25)

Pipeline: `07b_recipe1msubs_preparar.py` (.venv-trad: lê os .pkl com Unpickler que recusa imports, normaliza pelo normalizador do 02, aplica os filtros do 03, tira pares que estão na NOSSA validação/teste, teto de 20/par somando as fontes, traduz os ingredientes) → `07c_recipe1msubs_carregar.py` (.venv: embedding e insert; receitas de id NEGATIVO, `fonte='recipe1msubs'`, sempre `split='treino'`). Rodar DEPOIS do `07_receitas.py` (ele zera as tabelas). O app só lista receitas `id > 0`. Resultado: 25.834 ocorrências, 11.624 pares únicos (8.910 que a nossa mineração não tinha), 19.820 receitas só com ingredientes.

**Como linhas de treino: NÃO ajudou.** `F.carregar(externos=False)` é o padrão; no laboratório é a opção "Treinar também com a Recipe1MSubs" (`usar_externos`, desligada). 3 sementes, rankeáveis (213 de 503):

| Configuração | MRR | R@10 |
|---|---|---|
| só nossos (padrão) | 0,187 | 0,570 |
| nossos + externos | 0,186 | 0,531 |
| nossos + externos sem contexto de receita | 0,186 | 0,560 |
| externos com peso 0,3 | 0,177 | 0,548 |
| rede 64-32 + externos | 0,181 | 0,533 |
| pgvector (referência) | 0,186 | 0,526 |

Com externos o melhor ponto do treino cai para a época 1-2 (decora). Tirar o contexto de receita não piora (val_loss 0,382 sem, 0,396 com): o vetor da receita ajuda a decorar, não a generalizar.

**Como base de conhecimento: GANHO GRANDE (`15_experimento_kb.py`).** A Recipe1MSubs conhece 67% dos nossos pares de validação (fonte independente). Candidatos = top-30 do pgvector + substitutos que ela conhece: recall dos candidatos 0,423 -> 0,753 (média de 65 candidatos). Nos 503 pares (par fora dos candidatos vale 0):

| Sistema | MRR | NDCG@10 | R@5 | R@10 |
|---|---|---|---|---|
| pgvector top-30 | 0,079 | 0,103 | 0,123 | 0,223 |
| reranker atual | 0,077 | 0,106 | 0,118 | 0,240 |
| top-30 + feature KB | 0,148 | 0,189 | 0,250 | 0,338 |
| candidatos híbridos, sem feature | 0,105 | 0,128 | 0,142 | 0,275 |
| híbridos + feature KB (MLP 1958) | 0,194 | 0,236 | 0,300 | 0,422 |
| ranker compacto (poucas features) | 0,166 | 0,206 | 0,253 | 0,388 |
| **só a contagem da KB (candidatos híbridos, desempate por distância)** | **0,209** | **0,257** | **0,310** | **0,457** |

O MLP e o ranker compacto NÃO passam da consulta simples: pares que a KB conhece e a nossa mineração não achou entram no treino como negativos (ruído de rótulo). Ainda não está no app: o app continua com top-30 + reranker TFJS (modelo exportado só dos nossos dados, 1000 ingredientes).
