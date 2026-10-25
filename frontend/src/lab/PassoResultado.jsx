import { useEffect, useState } from 'react'
import Passo from './Passo.jsx'
import { dec, sinal } from './formatos.js'

const SISTEMAS = [['pgvector', 'Só pgvector', 'o baseline: a busca vetorial sem rede'], ['popularidade', 'Popularidade', 'sempre sugere o mais usado'], ['reranker', 'Sua rede (reranker)', 'a rede deste experimento']]
const COLUNAS = [['mrr', 'MRR', 'Quão perto do topo fica a resposta certa (1 = sempre em 1º)'], ['ndcg10', 'NDCG@10', 'Nota que premia acertar cedo, dentro do top-10'], ['r5', 'R@5', 'Em quantos casos a resposta certa está no top-5'], ['r10', 'R@10', 'Em quantos casos a resposta certa está no top-10']]

// Texto do veredito: a regra do projeto é "se o reranker não superar o pgvector, ele não se justifica".
function veredito(m, ruido) {
  const r = m.sistemas.reranker.rankeaveis, p = m.sistemas.pgvector.rankeaveis
  const r10 = r.r10 - p.r10, mrr = r.mrr - p.mrr
  const resumo = `R@10 ${sinal(r10)} e MRR ${sinal(mrr)} em relação ao pgvector`
  if (r10 > 0 && mrr > 0) return { tom: 'ok', texto: `Superou o baseline: ${resumo}.` + ruido }
  if (r10 > 0 || mrr > 0) return { tom: 'meio', texto: `Resultado misto: ${resumo}.` + ruido }
  return { tom: 'ruim', texto: `Não superou o pgvector: ${resumo}. Pela regra do projeto, o reranker não se justificaria com essa configuração.` + ruido }
}

export default function PassoResultado({ treino, onSalvar, onAtivar, onDesfazer, ocupado }) {
  const m = treino?.metricas
  const [visao, setVisao] = useState('rankeaveis')
  const [nome, setNome] = useState('')
  const [confirmando, setConfirmando] = useState(false)
  useEffect(() => { setNome(treino ? `modelo-${treino.id.slice(9, 15)}` : ''); setConfirmando(false) }, [treino?.id])

  if (!m) {
    return <Passo n="4" titulo="Valide e salve" subtitulo="Quando o treino terminar, a rede é testada nos pares da validação e comparada com os baselines.">
      <div className="rounded-xl border border-dashed border-stone-300 p-8 text-center text-sm text-stone-400">Os resultados aparecem aqui</div>
    </Passo>
  }

  const n = visao === 'rankeaveis' ? m.n_rankeaveis : m.n_pares
  const melhores = Object.fromEntries(COLUNAS.map(([k]) => [k, Math.max(...SISTEMAS.map(([s]) => m.sistemas[s][visao][k]))]))
  const ruido = ` Atenção: é uma semente e ${m.n_rankeaveis} pares; diferenças de poucos pontos podem ser sorte.`
  const v = veredito(m, ruido)
  const salvo = treino.salvo

  return (
    <Passo n="4" titulo="Valide e salve"
      subtitulo="Nota da rede nos pares da validação (os que ela nunca estudou), comparada com os baselines. O teste continua trancado.">
      <div className="mb-3 flex gap-2">
        {[['rankeaveis', `Só rankeáveis (${m.n_rankeaveis})`], ['todos', `Todos os pares (${m.n_pares})`]].map(([k, t]) => (
          <button key={k} onClick={() => setVisao(k)}
            className={`rounded-lg px-3 py-1.5 text-sm ${visao === k ? 'bg-stone-900 text-white' : 'bg-stone-100 hover:bg-stone-200'}`}>{t}</button>
        ))}
      </div>
      <div className="overflow-x-auto rounded-xl border border-stone-200">
        <table className="w-full text-sm">
          <thead className="bg-stone-50 text-left text-xs uppercase tracking-wide text-stone-500">
            <tr><th className="px-4 py-2">Sistema</th>{COLUNAS.map(([k, t, dica]) => <th key={k} className="px-4 py-2 text-right" title={dica}>{t}</th>)}</tr>
          </thead>
          <tbody className="divide-y divide-stone-100">
            {SISTEMAS.map(([s, rotulo, dica]) => (
              <tr key={s} className={s === 'reranker' ? 'bg-emerald-50/50' : ''}>
                <td className="px-4 py-2.5"><div className="font-medium">{rotulo}</div><div className="text-xs text-stone-400">{dica}</div></td>
                {COLUNAS.map(([k]) => {
                  const x = m.sistemas[s][visao][k]
                  return <td key={k} className={`px-4 py-2.5 text-right tabular-nums ${x === melhores[k] ? 'font-bold text-emerald-700' : 'text-stone-600'}`}>{dec(x)}</td>
                })}
              </tr>
            ))}
            <tr className="bg-stone-50 text-xs">
              <td className="px-4 py-2 text-stone-500">Sua rede menos o pgvector</td>
              {COLUNAS.map(([k]) => {
                const d = m.sistemas.reranker[visao][k] - m.sistemas.pgvector[visao][k]
                return <td key={k} className={`px-4 py-2 text-right font-medium tabular-nums ${d > 0 ? 'text-emerald-700' : 'text-rose-600'}`}>{sinal(d)}</td>
              })}
            </tr>
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-stone-400">Passe o mouse sobre o nome da coluna para ver o que ela mede. Base: {n} pares. "Rankeáveis" = o substituto certo está entre os 30 candidatos da busca.</p>
      <p className={`mt-3 rounded-lg p-3 text-sm ${{ ok: 'bg-emerald-50 text-emerald-900', meio: 'bg-amber-50 text-amber-900', ruim: 'bg-rose-50 text-rose-900' }[v.tom]}`}>{v.texto}</p>

      <div className="mt-6 rounded-xl border border-stone-200 bg-stone-50 p-4">
        <h3 className="font-medium">Salvar este modelo</h3>
        {!salvo ? (
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <input value={nome} onChange={(e) => setNome(e.target.value)} maxLength={60} placeholder="nome do modelo"
              className="min-w-[14rem] flex-1 rounded-lg border border-stone-300 px-3 py-2 text-sm" />
            <button onClick={() => onSalvar(nome.trim())} disabled={ocupado || !nome.trim()}
              className="rounded-lg bg-stone-900 px-4 py-2 text-sm font-semibold text-white hover:bg-stone-700 disabled:opacity-40">
              {ocupado ? 'Exportando...' : 'Salvar modelo'}</button>
            <p className="w-full text-xs text-stone-500">Converte para o formato do navegador (TFJS) e confere que o resultado é igual ao do Python. Ainda não muda o que o app usa.</p>
          </div>
        ) : (
          <div className="mt-2 space-y-3">
            <p className="text-sm"><b>{salvo.nome}</b> salvo. {salvo.paridade_ok
              ? <span className="text-emerald-700">Paridade Keras x TFJS conferida.</span>
              : <span className="text-rose-700">A paridade falhou: não ative este modelo.</span>}</p>
            {treino.ativo ? (
              <div className="flex flex-wrap items-center gap-3">
                <span className="rounded-full bg-emerald-600 px-3 py-1 text-sm font-medium text-white">Em uso no app</span>
                <span className="text-xs text-stone-500">Recarregue a página (F5) para o app carregar este modelo.</span>
                <button onClick={onDesfazer} disabled={ocupado} className="rounded-lg border border-stone-300 px-3 py-1.5 text-sm hover:bg-white disabled:opacity-40">Voltar ao modelo anterior</button>
              </div>
            ) : !confirmando ? (
              <button onClick={() => setConfirmando(true)} disabled={!salvo.paridade_ok || ocupado}
                className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white hover:bg-emerald-700 disabled:opacity-40">Ativar no app</button>
            ) : (
              <div className="flex flex-wrap items-center gap-3 rounded-lg bg-amber-50 p-3">
                <span className="text-sm text-amber-900">Isso troca o modelo que a tela "Substituir ingrediente" usa. O atual fica guardado e dá para desfazer.</span>
                <button onClick={() => { setConfirmando(false); onAtivar() }} disabled={ocupado} className="rounded-lg bg-emerald-600 px-4 py-1.5 text-sm font-semibold text-white">Confirmar</button>
                <button onClick={() => setConfirmando(false)} className="rounded-lg border border-stone-300 px-3 py-1.5 text-sm">Cancelar</button>
              </div>
            )}
          </div>
        )}
      </div>
    </Passo>
  )
}
