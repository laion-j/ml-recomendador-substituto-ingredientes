import Passo from './Passo.jsx'
import { REFERENCIA } from './params.js'
import { dataHora, dec } from './formatos.js'

// Todos os experimentos já rodados, para comparar configurações lado a lado.
export default function Historico({ itens, focoId, onVer, onReusar }) {
  return (
    <Passo titulo="Experimentos anteriores"
      subtitulo="Cada treino fica guardado. Compare redes diferentes e reaproveite os parâmetros de uma que deu certo.">
      {itens.length === 0 ? <p className="text-sm text-stone-400">Nenhum experimento ainda. Rode o primeiro no passo 3.</p> : (
        <div className="overflow-x-auto rounded-xl border border-stone-200">
          <table className="w-full text-sm">
            <thead className="bg-stone-50 text-left text-xs uppercase tracking-wide text-stone-500">
              <tr>
                <th className="px-3 py-2">Quando</th><th className="px-3 py-2">Rede</th><th className="px-3 py-2 text-right">Épocas</th>
                <th className="px-3 py-2 text-right">Melhor perda val.</th><th className="px-3 py-2 text-right">R@10</th><th className="px-3 py-2 text-right">MRR</th>
                <th className="px-3 py-2">Situação</th><th />
              </tr>
            </thead>
            <tbody className="divide-y divide-stone-100">
              <tr className="bg-stone-50/60 text-xs text-stone-500">
                <td className="px-3 py-2" colSpan={4}>Referência: modelo do script 10_treinar (pgvector sozinho: R@10 {dec(REFERENCIA.r10_pgvector)}, MRR {dec(REFERENCIA.mrr_pgvector)})</td>
                <td className="px-3 py-2 text-right tabular-nums">{dec(REFERENCIA.r10)}</td><td className="px-3 py-2 text-right tabular-nums">{dec(REFERENCIA.mrr)}</td><td colSpan={2} />
              </tr>
              {itens.map((e) => {
                const r = e.metricas?.sistemas.reranker.rankeaveis
                return (
                  <tr key={e.id} className={e.id === focoId ? 'bg-emerald-50/60' : ''}>
                    <td className="whitespace-nowrap px-3 py-2 text-stone-500">{dataHora(e.criado)}</td>
                    <td className="px-3 py-2"><span className="font-medium">{e.params.camadas.join(' · ')}</span>
                      <span className="ml-2 text-xs text-stone-400">dropout {e.params.dropout} · {e.params.otimizador} · lr {e.params.taxa}</span></td>
                    <td className="px-3 py-2 text-right tabular-nums">{e.epocas_rodadas}/{e.params.epocas}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{dec(e.melhor_val_loss)}</td>
                    <td className="px-3 py-2 text-right font-medium tabular-nums">{dec(r?.r10)}</td>
                    <td className="px-3 py-2 text-right font-medium tabular-nums">{dec(r?.mrr)}</td>
                    <td className="px-3 py-2">
                      <span className="text-xs text-stone-500">{e.status}</span>
                      {e.salvo && <span className="ml-1.5 rounded-full bg-stone-200 px-2 py-0.5 text-xs">{e.salvo.nome}</span>}
                      {e.ativo && <span className="ml-1.5 rounded-full bg-emerald-600 px-2 py-0.5 text-xs text-white">em uso</span>}
                    </td>
                    <td className="whitespace-nowrap px-3 py-2 text-right">
                      <button onClick={() => onVer(e.id)} className="rounded-md px-2 py-1 text-xs hover:bg-stone-100">ver</button>
                      <button onClick={() => onReusar(e.params)} className="rounded-md px-2 py-1 text-xs hover:bg-stone-100">reusar parâmetros</button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </Passo>
  )
}
