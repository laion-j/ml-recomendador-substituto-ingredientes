import { useState } from 'react'
import { dec } from './formatos.js'

const W = 720, H = 300, L = 52, R = 16, T = 16, B = 36

// Curva de aprendizado em SVG puro: perda no treino (verde) x perda na validação (vermelho), por época.
// Quando as duas se separam (treino cai, validação sobe) a rede parou de aprender e começou a decorar.
export default function GraficoCurvas({ historico, epocasMax, melhorEpoca }) {
  const [hover, setHover] = useState(null)
  if (!historico.length) {
    return <div className="flex h-72 items-center justify-center rounded-xl border border-dashed border-stone-300 text-sm text-stone-400">
      A curva de aprendizado aparece aqui quando o treino começar
    </div>
  }
  const xMax = Math.max(historico.length, Math.min(epocasMax ?? 0, 30), 2)   // eixo não "pula" durante o treino
  const yMax = Math.max(...historico.flatMap((h) => [h.loss, h.val_loss])) * 1.08
  const x = (e) => L + ((e - 1) / (xMax - 1)) * (W - L - R)
  const y = (v) => T + (1 - v / yMax) * (H - T - B)
  const linha = (k) => historico.map((h) => `${x(h.epoca)},${y(h[k])}`).join(' ')
  const passoX = Math.ceil(xMax / 10)
  const ticksX = Array.from({ length: Math.floor((xMax - 1) / passoX) + 1 }, (_, i) => 1 + i * passoX)
  const h = hover != null ? historico[hover - 1] : null

  function mover(e) {
    const r = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - r.left) / r.width) * W                       // pixel da tela -> coordenada do SVG
    const ep = Math.round(1 + ((px - L) / (W - L - R)) * (xMax - 1))
    setHover(Math.min(Math.max(ep, 1), historico.length))
  }

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" onMouseMove={mover} onMouseLeave={() => setHover(null)} role="img" aria-label="Curva de perda por época">
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <line x1={L} x2={W - R} y1={y(f * yMax)} y2={y(f * yMax)} className="stroke-stone-100" />
            <text x={L - 8} y={y(f * yMax) + 4} textAnchor="end" className="fill-stone-400 text-[11px]">{dec(f * yMax, 2)}</text>
          </g>
        ))}
        {ticksX.map((e) => <text key={e} x={x(e)} y={H - 14} textAnchor="middle" className="fill-stone-400 text-[11px]">{e}</text>)}
        <text x={(L + W - R) / 2} y={H - 1} textAnchor="middle" className="fill-stone-400 text-[11px]">época</text>
        {melhorEpoca && (
          <g>
            <line x1={x(melhorEpoca)} x2={x(melhorEpoca)} y1={T} y2={H - B} className="stroke-emerald-500" strokeDasharray="4 4" />
            <text x={x(melhorEpoca) + 5} y={T + 11} className="fill-emerald-700 text-[11px] font-medium">melhor: época {melhorEpoca}</text>
          </g>
        )}
        <polyline points={linha('loss')} fill="none" className="stroke-emerald-600" strokeWidth="2.5" strokeLinejoin="round" />
        <polyline points={linha('val_loss')} fill="none" className="stroke-rose-500" strokeWidth="2.5" strokeLinejoin="round" />
        {historico.map((p) => (
          <g key={p.epoca}>
            <circle cx={x(p.epoca)} cy={y(p.loss)} r="3" className="fill-emerald-600" />
            <circle cx={x(p.epoca)} cy={y(p.val_loss)} r="3" className="fill-rose-500" />
          </g>
        ))}
        {h && <line x1={x(h.epoca)} x2={x(h.epoca)} y1={T} y2={H - B} className="stroke-stone-300" />}
      </svg>
      {h && (
        <div className="pointer-events-none absolute top-2 rounded-lg border border-stone-200 bg-white px-3 py-2 text-xs shadow-md"
          style={{ left: `${(x(h.epoca) / W) * 100}%`, transform: x(h.epoca) > W * 0.6 ? 'translateX(calc(-100% - 12px))' : 'translateX(12px)' }}>
          <b>Época {h.epoca}</b>
          <div className="text-emerald-700">treino {dec(h.loss)}</div>
          <div className="text-rose-600">validação {dec(h.val_loss)}</div>
        </div>
      )}
      <div className="mt-1 flex gap-5 pl-[52px] text-xs text-stone-600">
        <span className="flex items-center gap-1.5"><i className="h-0.5 w-5 bg-emerald-600" />perda no treino (cartões que a rede estuda)</span>
        <span className="flex items-center gap-1.5"><i className="h-0.5 w-5 bg-rose-500" />perda na validação (cartões que ela nunca viu)</span>
      </div>
    </div>
  )
}
