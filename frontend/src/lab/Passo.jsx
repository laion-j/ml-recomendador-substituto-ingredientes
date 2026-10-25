// Cartão numerado que envolve cada etapa do laboratório (dados -> configurar -> treinar -> validar).
export default function Passo({ n, titulo, subtitulo, direita, children }) {
  return (
    <section className="rounded-2xl border border-stone-200 bg-white shadow-sm">
      <header className="flex items-start justify-between gap-4 border-b border-stone-100 px-5 py-4">
        <div className="flex items-start gap-3">
          {n && <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-stone-900 text-sm font-semibold text-white">{n}</span>}
          <div>
            <h2 className="text-lg font-semibold text-stone-900">{titulo}</h2>
            {subtitulo && <p className="mt-0.5 max-w-2xl text-sm text-stone-500">{subtitulo}</p>}
          </div>
        </div>
        {direita}
      </header>
      <div className="p-5">{children}</div>
    </section>
  )
}
