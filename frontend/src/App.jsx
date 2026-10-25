import { useState } from 'react'
import Substituir from './telas/Substituir.jsx'
import Lab from './lab/Lab.jsx'

// Casca do app: duas telas, escolhidas por abas. Cada tela cuida do próprio layout/largura.
const TELAS = [['substituir', 'Substituir ingrediente'], ['lab', 'Laboratório de treino']]

export default function App() {
  const [tela, setTela] = useState('substituir')
  return (
    <div className="min-h-screen bg-stone-50 text-stone-900">
      <nav className="sticky top-0 z-10 border-b border-stone-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center gap-1 px-6 py-2">
          <span className="mr-4 font-semibold">Substitutos</span>
          {TELAS.map(([k, t]) => (
            <button key={k} onClick={() => setTela(k)}
              className={`rounded-lg px-3 py-1.5 text-sm font-medium transition ${tela === k ? 'bg-stone-900 text-white' : 'text-stone-600 hover:bg-stone-100'}`}>{t}</button>
          ))}
        </div>
      </nav>
      {/* hidden em vez de desmontar: o Substituir carrega o modelo TFJS num worker; remontar recarregaria tudo */}
      <div className={tela === 'substituir' ? '' : 'hidden'}><Substituir /></div>
      {tela === 'lab' && <Lab />}
    </div>
  )
}
