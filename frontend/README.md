# Frontend (React + Vite + TFJS 4.22.0)

```bash
# backend em outro terminal: cd backend && uvicorn app.main:app --port 8765
npm install
npm run dev      # http://localhost:5173  (VITE_API_URL muda a URL da API; default http://localhost:8765)
npm test         # features.test.js + paridade (skip se o modelo ainda não foi exportado)
npm run build
```

Precisa de `public/models/` com `model.json`, `.bin`, `feature_spec.json` (gerados por `scripts_ml/`). Sem eles o app mostra "modelo não encontrado".

## Arquitetura

```
App.jsx -> lib/api.js -> POST /api/buscar-candidatos (recall pgvector)
        -> workers/rerankWorker.js -> lib/features.js -> tf.tensor2d -> model.predict -> scores
        -> ordena por score e mostra a posição original do pgvector
```

- `src/lib/features.js`: texto/lista -> vetor (multi-hot, o*c, |o-c|, distância); função pura, valida `dim_entrada`.
- `src/workers/rerankWorker.js`: vetores -> `tensor2d [n, dim]` -> `predict` -> `dataSync` (tudo em `tf.tidy`).
- `src/App.jsx`: UI, ordenação por score e cálculo da posição original (ordem do pgvector).
- O score é de ranking (sigmoid), não probabilidade calibrada.
