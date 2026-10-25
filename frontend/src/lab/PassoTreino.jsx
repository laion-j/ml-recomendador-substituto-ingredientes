import Passo from './Passo.jsx'
import GraficoCurvas from './GraficoCurvas.jsx'
import { dec, duracao, num } from './formatos.js'

const STATUS = {
  preparando: ['Preparando os dados...', 'bg-stone-100 text-stone-700'],
  treinando: ['Treinando', 'bg-emerald-100 text-emerald-800'],
  avaliando: ['Avaliando na validação...', 'bg-amber-100 text-amber-800'],
  concluido: ['Concluído', 'bg-emerald-600 text-white'],
  parado: ['Interrompido por você', 'bg-stone-200 text-stone-800'],
  falhou: ['Falhou', 'bg-rose-100 text-rose-800'],
}
export const EM_ANDAMENTO = ['preparando', 'treinando', 'avaliando']

// Lê a curva e diz em português o que ela mostra (overfit, ainda aprendendo...).
export function diagnostico(hist, melhor) {
  if (hist.length < 3 || !melhor) return null
  const ult = hist[hist.length - 1], best = hist[melhor - 1]
  if (ult.epoca - melhor >= 2 && ult.val_loss > best.val_loss + 0.01) {
    return { tom: 'alerta', texto: `Overfit: a perda de validação passou do melhor ponto (época ${melhor}, ${dec(best.val_loss)}) e já está em ${dec(ult.val_loss)}, enquanto a de treino segue caindo (${dec(ult.loss)}). A rede está decorando os cartões.` }
  }
  if (ult.epoca === melhor) return { tom: 'ok', texto: 'Ainda aprendendo: a perda de validação segue melhorando.' }
  return null
}

function Tile({ rotulo, valor, sub }) {
  return (
    <div className="rounded-xl bg-stone-50 p-3">
      <p className="text-xs text-stone-500">{rotulo}</p>
      <p className="text-xl font-semibold tabular-nums">{valor}</p>
      {sub && <p className="text-xs text-stone-400">{sub}</p>}
    </div>
  )
}

export default function PassoTreino({ treino, onTreinar, onParar, iniciando }) {
  const status = treino?.status
  const rodando = EM_ANDAMENTO.includes(status)
  const hist = treino?.historico ?? []
  const ult = hist[hist.length - 1]
  const [rotulo, cor] = STATUS[status] ?? ['Pronto para treinar', 'bg-stone-100 text-stone-600']
  const epocasMax = treino?.epocas_max ?? treino?.params?.epocas ?? 0
  const diag = diagnostico(hist, treino?.melhor_epoca)

  return (
    <Passo n="3" titulo="Treine e acompanhe"
      subtitulo="A rede lê os cartões, erra, ajusta os pesos e repete. A curva mostra o erro em cada volta (época)."
      direita={
        <div className="flex items-center gap-2">
          <span className={`rounded-full px-3 py-1 text-sm font-medium ${cor}`}>{rotulo}</span>
          {rodando
            ? <button onClick={onParar} disabled={status === 'avaliando'} className="rounded-lg border border-stone-300 px-4 py-2 text-sm font-medium hover:bg-stone-50 disabled:opacity-40">Parar</button>
            : <button onClick={onTreinar} disabled={iniciando} className="rounded-lg bg-emerald-600 px-5 py-2 text-sm font-semibold text-white shadow-sm hover:bg-emerald-700 disabled:opacity-50">
                {iniciando ? 'Iniciando...' : status ? 'Treinar de novo' : 'Treinar'}
              </button>}
        </div>
      }>
      {rodando && (
        <div className="mb-4 h-2 overflow-hidden rounded-full bg-stone-100" aria-label="progresso">
          <div className="h-full bg-emerald-500 transition-all duration-500"
            style={{ width: `${status === 'avaliando' ? 100 : Math.max(3, (hist.length / (epocasMax || 1)) * 100)}%` }} />
        </div>
      )}
      {status === 'falhou' && <pre className="mb-4 max-h-40 overflow-auto whitespace-pre-wrap rounded-lg bg-rose-50 p-3 text-xs text-rose-900">{treino.erro}</pre>}

      <GraficoCurvas historico={hist} epocasMax={epocasMax} melhorEpoca={treino?.melhor_epoca} />

      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <Tile rotulo="Época" valor={ult ? `${ult.epoca}` : '-'} sub={`de até ${epocasMax || '-'}`} />
        <Tile rotulo="Perda no treino" valor={dec(ult?.loss)} />
        <Tile rotulo="Perda na validação" valor={dec(ult?.val_loss)} />
        <Tile rotulo="Melhor época" valor={treino?.melhor_epoca ?? '-'} sub={treino?.melhor_epoca ? `validação ${dec(hist[treino.melhor_epoca - 1]?.val_loss)}` : null} />
        <Tile rotulo="Tempo" valor={duracao(ult?.seg)} />
        <Tile rotulo="Pesos" valor={num(treino?.params_totais)} />
      </div>

      {diag && (
        <p className={`mt-4 rounded-lg p-3 text-sm ${diag.tom === 'alerta' ? 'bg-amber-50 text-amber-900' : 'bg-emerald-50 text-emerald-900'}`}>{diag.texto}</p>
      )}
      {treino?.motivo_fim && !rodando && status !== 'falhou' && (
        <p className="mt-3 text-sm text-stone-500">
          Parou por: {{ epocas: 'chegou ao máximo de épocas', paciencia: 'a paciência acabou (a validação parou de melhorar)', usuario: 'pedido seu' }[treino.motivo_fim]}.
          {treino.params.restaurar_melhor ? ` Os pesos voltaram à época ${treino.melhor_epoca}.` : ' Vale a última época.'}
        </p>
      )}
    </Passo>
  )
}
