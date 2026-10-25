import { contarParametros } from './params.js'
import { dec, num } from './formatos.js'

// Desenha a rede que os botões acima descrevem: uma coluna por camada, altura ~ log(nº de neurônios).
export default function Arquitetura({ camadas, dropout, dim, nExemplos }) {
  const colunas = [{ rotulo: 'entrada', n: dim }, ...camadas.map((n, i) => ({ rotulo: `oculta ${i + 1}`, n, dropout: dropout > 0 && i < camadas.length - 1 })), { rotulo: 'saída', n: 1 }]
  const W = 640, H = 210, base = 150
  const passo = W / colunas.length
  const altura = (n) => 18 + 100 * (Math.log(n) / Math.log(dim))
  const pesos = contarParametros(camadas, dim)
  const razao = nExemplos ? pesos / nExemplos : null

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Arquitetura da rede neural">
        {colunas.map((c, i) => {
          const cx = passo * i + passo / 2, h = altura(c.n), prox = colunas[i + 1]
          const cor = i === 0 ? 'fill-stone-300' : i === colunas.length - 1 ? 'fill-amber-400' : 'fill-emerald-500'
          return (
            <g key={i}>
              {prox && <line x1={cx + 16} y1={base - 55} x2={cx + passo - 16} y2={base - 55} className="stroke-stone-300" strokeWidth="2" strokeDasharray="2 4" />}
              <rect x={cx - 16} y={base - 55 - h / 2} width="32" height={h} rx="8" className={cor} />
              {c.dropout && <rect x={cx - 21} y={base - 60 - h / 2} width="42" height={h + 10} rx="11" className="fill-none stroke-stone-400" strokeDasharray="4 3" />}
              <text x={cx} y={base - 70 - h / 2} textAnchor="middle" className="fill-stone-900 text-[13px] font-semibold">{num(c.n)}</text>
              <text x={cx} y={H - 28} textAnchor="middle" className="fill-stone-500 text-[11px]">{c.rotulo}</text>
              {c.dropout && <text x={cx} y={H - 14} textAnchor="middle" className="fill-stone-400 text-[10px]">dropout {dec(dropout, 2)}</text>}
            </g>
          )
        })}
      </svg>
      <dl className="mt-1 grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-lg bg-stone-50 p-3">
          <dt className="text-xs text-stone-500">Pesos para aprender</dt>
          <dd className="text-lg font-semibold tabular-nums">{num(pesos)}</dd>
        </div>
        <div className={`rounded-lg p-3 ${razao > 5 ? 'bg-amber-50' : 'bg-stone-50'}`}>
          <dt className="text-xs text-stone-500">Pesos por exemplo de treino</dt>
          <dd className="text-lg font-semibold tabular-nums">{razao == null ? '-' : dec(razao, 1)}</dd>
        </div>
      </dl>
      {razao > 5 && (
        <p className="mt-2 rounded-lg bg-amber-50 p-2.5 text-xs text-amber-900">
          A rede tem {dec(razao, 0)}x mais pesos do que exemplos de treino. Quando sobra capacidade, ela tende a decorar os cartões (overfit): veja a curva de validação subir no passo 3.
        </p>
      )}
    </div>
  )
}
