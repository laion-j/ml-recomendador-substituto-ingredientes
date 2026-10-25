import Passo from './Passo.jsx'
import { num, pct } from './formatos.js'

// Barra horizontal dividida em fatias proporcionais; a legenda embaixo repete os números.
function BarraEmpilhada({ fatias }) {
  const total = fatias.reduce((s, f) => s + f.valor, 0)
  return (
    <div>
      <div className="flex h-9 overflow-hidden rounded-lg">
        {fatias.map((f) => (
          <div key={f.rotulo} style={{ width: `${(f.valor / total) * 100}%` }} title={`${f.rotulo}: ${num(f.valor)}`}
            className={`flex items-center justify-center text-xs font-medium text-white ${f.cor}`}>
            {f.valor / total > 0.07 ? pct(f.valor, total) : ''}
          </div>
        ))}
      </div>
      <ul className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-sm">
        {fatias.map((f) => (
          <li key={f.rotulo} className="flex items-center gap-1.5">
            <span className={`h-2.5 w-2.5 rounded-sm ${f.cor}`} />
            <span className="text-stone-600">{f.rotulo}</span>
            <b className="tabular-nums">{num(f.valor)}</b>
            {f.nota && <span className="text-xs text-stone-400">{f.nota}</span>}
          </li>
        ))}
      </ul>
    </div>
  )
}

const Esqueleto = () => <div className="h-40 animate-pulse rounded-xl bg-stone-100" />

export default function PassoDados({ dados, erro }) {
  const s = dados?.splits
  return (
    <Passo n="1" titulo="Os dados"
      subtitulo="Antes de treinar: de onde vêm os exemplos e como foram separados. Essa parte já está pronta no banco; aqui você só enxerga.">
      {erro && <p className="rounded-lg bg-red-50 p-3 text-sm text-red-800">{erro}</p>}
      {!dados && !erro && <Esqueleto />}
      {dados && (
        <div className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="space-y-2">
              <h3 className="font-medium">Separação dos pares de substituição</h3>
              <BarraEmpilhada fatias={[
                { rotulo: 'Treino', valor: s.treino.pares_unicos, cor: 'bg-emerald-500', nota: 'a rede estuda aqui' },
                { rotulo: 'Validação', valor: s.validacao.pares_unicos, cor: 'bg-amber-400', nota: 'confere se aprendeu' },
                { rotulo: 'Teste', valor: s.teste.pares_unicos, cor: 'bg-stone-400', nota: 'travado' },
              ]} />
              <p className="text-sm text-stone-500">
                A divisão é por <b>par único</b> (ex.: manteiga → margarina): o mesmo par nunca cai em dois grupos, senão
                a rede só decoraria. Os {num(s.treino.ocorrencias + s.validacao.ocorrencias + s.teste.ocorrencias)} usos
                reais desses pares (um por receita) acompanham o grupo do par.
                O <b>teste</b> fica trancado até o fim do projeto: usar antes contaminaria a nota final.
              </p>
              {s.treino.ocorrencias_externas > 0 && (
                <p className="rounded-lg bg-sky-50 p-3 text-sm text-sky-900">
                  <b>Duas fontes no treino.</b> {num(s.treino.ocorrencias - s.treino.ocorrencias_externas)} usos vêm das
                  reviews do Food.com (nossa mineração) e {num(s.treino.ocorrencias_externas)} da <b>Recipe1MSubs</b> (Meta),
                  que só traz a lista de ingredientes da receita. Ela é só treino: validação e teste são 100% nossos,
                  e pares que estão neles foram tirados dela.{' '}
                  {num(s.treino.pares_so_externos)} pares do treino só existem nessa fonte.
                </p>
              )}
            </div>

            <div className="space-y-2">
              <h3 className="font-medium">Exemplos que a rede enxerga</h3>
              {['treino', 'validacao'].map((k) => (
                <div key={k}>
                  <p className="mb-1 text-xs font-medium uppercase tracking-wide text-stone-400">{k === 'treino' ? 'Treino' : 'Validação'}</p>
                  <BarraEmpilhada fatias={[
                    { rotulo: 'Troca real', valor: dados.exemplos[k].positivos, cor: 'bg-emerald-600' },
                    { rotulo: 'Vizinho que ninguém usou', valor: dados.exemplos[k].negativos, cor: 'bg-rose-400' },
                  ]} />
                </div>
              ))}
              <p className="text-sm text-stone-500">
                Cada troca real vira 1 exemplo <b>positivo</b> e {dados.neg_por_pos} <b>negativos</b>: ingredientes que o
                pgvector acharia parecidos, mas que ninguém usou como troca. Assim a rede treina com os mesmos
                candidatos que verá no app. Na validação entram só os pares em que o substituto está no top-{dados.k}{' '}
                da busca ({dados.validacao_rankeaveis.pares} de {dados.validacao_rankeaveis.total}, {pct(dados.validacao_rankeaveis.pares, dados.validacao_rankeaveis.total)}):
                se a busca não o traz, a rede nem teria como acertar.
              </p>
            </div>
          </div>

          <div>
            <h3 className="mb-2 font-medium">Cartões de estudo (amostra do treino)</h3>
            <div className="overflow-x-auto rounded-xl border border-stone-200">
              <table className="w-full text-sm">
                <thead className="bg-stone-50 text-left text-xs uppercase tracking-wide text-stone-500">
                  <tr><th className="px-3 py-2">Receita</th><th className="px-3 py-2">Faltou</th><th className="px-3 py-2">Candidato</th><th className="px-3 py-2">Gabarito</th></tr>
                </thead>
                <tbody className="divide-y divide-stone-100">
                  {dados.amostra.map((a, i) => (
                    <tr key={i}>
                      <td className="max-w-[16rem] truncate px-3 py-2 text-stone-500" title={a.receita}>{a.receita}</td>
                      <td className="px-3 py-2 font-medium">{a.original}</td>
                      <td className="px-3 py-2 font-medium">{a.candidato}</td>
                      <td className="px-3 py-2">
                        <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${a.rotulo ? 'bg-emerald-100 text-emerald-800' : 'bg-rose-100 text-rose-800'}`}>
                          {a.rotulo ? '1 · troca real' : '0 · ninguém usou'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-xs text-stone-400">
              Cada linha vira um vetor de {num(dados.dim_entrada)} números (embeddings da receita, do original e do candidato, mais o produto e a diferença entre eles).
              A coluna "Gabarito" é a resposta que a rede tenta acertar.
            </p>
          </div>
        </div>
      )}
    </Passo>
  )
}
