// Cliente HTTP do backend FastAPI. Base configurável por VITE_API_URL.
const BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8765'

async function chamar(caminho, opcoes) {
  const r = await fetch(BASE + caminho, opcoes)
  if (!r.ok) throw new Error(`API ${r.status}: ${await r.text()}`)
  return r.json()
}

export const listarReceitas = () => chamar('/api/receitas')

export const buscarCandidatos = (receita_texto, ingrediente_faltante, k = 30) =>
  chamar('/api/buscar-candidatos', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ receita_texto, ingrediente_faltante, k }),
  })
