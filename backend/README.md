# Backend (FastAPI)

Recall com pgvector. Requer o Postgres do `docker-compose.yml` rodando e `scripts_ml/` já ter gravado os embeddings.

```bash
source ../.venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8765      # rode dentro de backend/ (8000 costuma estar ocupada nesta máquina)
pytest                                # idem; usa o banco e o modelo reais
```

`DATABASE_URL` (opcional) sobrescreve o default do docker-compose.

## Contrato

`GET /api/receitas` -> `[{id, titulo, ingredientes}]` (até 50).

`POST /api/buscar-candidatos`
```json
{ "receita_texto": "...", "ingrediente_faltante": "butter", "k": 30 }   // k: 1..100
```
Resposta:
```json
{ "receita_embedding": [384 floats],
  "ingrediente_embedding": [384 floats],
  "candidatos": [{ "nome", "categoria", "funcoes", "sabores", "distancia", "embedding": [384] }] }
```
- A busca parte do ingrediente faltante (embedding do banco se o nome existir, senão embeda o texto); a receita só é vetorizada para o reranker.
- `distancia` = cosseno (`<=>`), ordem crescente; o próprio ingrediente é excluído.
- Texto vazio ou `k` fora de 1..100 -> 422.
- A consulta fica isolada em `app/repositorio.py::buscar_candidatos` (pode virar híbrida).
