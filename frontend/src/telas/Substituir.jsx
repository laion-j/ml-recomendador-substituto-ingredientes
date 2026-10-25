import { useEffect, useRef, useState } from 'react'
import { buscarCandidatos, listarReceitas } from '../lib/api.js'

export default function Substituir() {
  const [aba, setAba] = useState('pronta')
  const [receitas, setReceitas] = useState([])
  const [receitaId, setReceitaId] = useState('')
  const [textoProprio, setTextoProprio] = useState('')
  const [faltante, setFaltante] = useState('')
  const [carregando, setCarregando] = useState(false)
  const [erro, setErro] = useState(null)
  const [ranking, setRanking] = useState(null)
  const [erroModelo, setErroModelo] = useState(null)
  const [modeloPronto, setModeloPronto] = useState(false)

  const worker = useRef(null)
  const pendentes = useRef(new Map()) // id da requisição -> { resolve, reject }

  // Sobe o worker uma vez e pede o carregamento do modelo
  useEffect(() => {
    const w = new Worker(new URL('../workers/rerankWorker.js', import.meta.url), { type: 'module' })
    w.onmessage = ({ data }) => {
      if (data.tipo === 'pronto') setModeloPronto(true)
      else if (data.tipo === 'erro-modelo') setErroModelo(data.mensagem)
      else if (data.tipo === 'resultado') pendentes.current.get(data.id)?.resolve(data.scores)
      else if (data.tipo === 'erro') pendentes.current.get(data.id)?.reject(new Error(data.mensagem))
    }
    w.onerror = (e) => setErroModelo(e.message || 'falha no worker')
    w.postMessage({ tipo: 'carregar' })
    worker.current = w
    return () => w.terminate()
  }, [])

  useEffect(() => {
    listarReceitas().then(setReceitas).catch((e) => setErro(`Não foi possível listar receitas: ${e.message}`))
  }, [])

  const reranquear = (resposta) =>
    new Promise((resolve, reject) => {
      const id = crypto.randomUUID()
      pendentes.current.set(id, { resolve, reject })
      worker.current.postMessage({ tipo: 'rerank', id, resposta })
    })

  const receitaTexto = aba === 'pronta'
    ? (() => { const r = receitas.find((x) => String(x.id) === receitaId); return r ? `${r.titulo}. ${[].concat(r.ingredientes).join(', ')}` : '' })()
    : textoProprio

  const receitaSelecionada = aba === 'pronta' ? receitas.find((x) => String(x.id) === receitaId) : null
  const ingredientesDaReceita = receitaSelecionada ? [...new Set([].concat(receitaSelecionada.ingredientes))] : []

  // Clicar num ingrediente da receita = defini-lo como faltante e já buscar (sem esperar o state atualizar)
  function escolherIngrediente(ing) {
    setFaltante(ing)
    if (modeloPronto && !carregando) buscar(ing)
  }

  async function buscar(ingrediente = faltante) {
    setErro(null); setRanking(null); setCarregando(true)
    try {
      const resposta = await buscarCandidatos(receitaTexto, ingrediente.trim())
      // posição original = ordem em que o pgvector devolveu (distância crescente)
      const scores = await reranquear(resposta)
      const lista = resposta.candidatos
        .map((c, i) => ({ ...c, posOriginal: i + 1, score: scores[i] }))
        .sort((a, b) => b.score - a.score) // maior score primeiro
      setRanking(lista)
    } catch (e) {
      setErro(e.message)
    } finally {
      setCarregando(false)
    }
  }

  const podeBuscar = receitaTexto.trim() && faltante.trim() && !carregando && modeloPronto

  return (
    <main className="mx-auto max-w-3xl p-6 space-y-4">
      <h1 className="text-2xl font-bold">Substituto de ingredientes</h1>

      {erroModelo && <p className="rounded bg-amber-100 p-3 text-amber-900">{erroModelo}</p>}
      {!modeloPronto && !erroModelo && <p className="text-sm text-stone-500">Carregando modelo...</p>}

      <div className="flex gap-2">
        {[['pronta', 'Receita Pronta'], ['propria', 'Receita Própria']].map(([k, t]) => (
          <button key={k} onClick={() => setAba(k)}
            className={`rounded px-3 py-1 ${aba === k ? 'bg-stone-900 text-white' : 'bg-stone-200'}`}>{t}</button>
        ))}
      </div>

      {aba === 'pronta' ? (
        <>
          <select value={receitaId} onChange={(e) => { setReceitaId(e.target.value); setFaltante(''); setRanking(null) }}
            className="w-full rounded border p-2">
            <option value="">Escolha uma receita...</option>
            {receitas.map((r) => <option key={r.id} value={r.id}>{r.titulo}</option>)}
          </select>
          {ingredientesDaReceita.length > 0 && (
            <section className="rounded border bg-white p-3">
              <p className="mb-2 text-sm text-stone-500">Ingredientes da receita (clique no que quer substituir):</p>
              <ul className="flex flex-wrap gap-2">
                {ingredientesDaReceita.map((ing) => (
                  <li key={ing}>
                    <button onClick={() => escolherIngrediente(ing)} disabled={carregando}
                      className={`rounded-full border px-3 py-1 text-sm disabled:opacity-50 ${faltante === ing ? 'bg-emerald-600 text-white' : 'bg-stone-100 hover:bg-emerald-100'}`}>
                      {ing}
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </>
      ) : (
        <textarea value={textoProprio} onChange={(e) => setTextoProprio(e.target.value)} rows={5}
          placeholder="Cole ou escreva a receita (título e ingredientes)" className="w-full rounded border p-2" />
      )}

      <input value={faltante} onChange={(e) => setFaltante(e.target.value)} placeholder="Ingrediente faltante (ex.: manteiga)"
        className="w-full rounded border p-2" />

      <button onClick={() => buscar()} disabled={!podeBuscar}
        className="rounded bg-emerald-600 px-4 py-2 text-white disabled:opacity-40">
        {carregando ? 'Buscando...' : 'Buscar substitutos'}
      </button>

      {erro && <p className="rounded bg-red-100 p-3 text-red-800">{erro}</p>}

      {ranking && (
        <ol className="space-y-2">
          {ranking.map((c, i) => (
            <li key={c.nome} className="flex items-center justify-between rounded border bg-white p-3">
              <div>
                <span className="font-semibold">{i + 1}. {c.nome}</span>
                <span className="ml-2 text-sm text-stone-500">{c.categoria} | {c.funcoes?.join(', ')} | {c.sabores?.join(', ')}</span>
              </div>
              <div className="text-right text-sm">
                <div>score de compatibilidade: <b>{c.score.toFixed(3)}</b></div>
                <div className="text-stone-500">posição pgvector: #{c.posOriginal} ({c.posOriginal - (i + 1) >= 0 ? '+' : ''}{c.posOriginal - (i + 1)})</div>
              </div>
            </li>
          ))}
        </ol>
      )}
    </main>
  )
}
