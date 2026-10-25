import { useCallback, useEffect, useRef, useState } from 'react'
import { ativarModelo, desfazerAtivacao, detalharTreino, iniciarTreino, listarTreinos, pararTreino, resumoDados, salvarModelo } from '../lib/apiLab.js'
import { PADRAO } from './params.js'
import PassoDados from './PassoDados.jsx'
import PassoConfig from './PassoConfig.jsx'
import PassoTreino, { EM_ANDAMENTO } from './PassoTreino.jsx'
import PassoResultado from './PassoResultado.jsx'
import Historico from './Historico.jsx'

// Tela do laboratório: dados -> configurar -> treinar -> validar/salvar. O treino roda no backend (Python/TensorFlow);
// esta tela só dispara, consulta o progresso a cada segundo e desenha.
export default function Lab() {
  const [params, setParams] = useState(PADRAO)
  const [dados, setDados] = useState(null)
  const [erroDados, setErroDados] = useState(null)
  const [itens, setItens] = useState([])
  const [treino, setTreino] = useState(null)       // o experimento em foco (rodando ou revisitado)
  const [iniciando, setIniciando] = useState(false)
  const [ocupado, setOcupado] = useState(false)    // salvando/ativando
  const [erro, setErro] = useState(null)
  const focoId = useRef(null)                       // evita que uma resposta atrasada sobrescreva outro experimento

  const carregarHistorico = useCallback(async () => {
    try { setItens((await listarTreinos()).experimentos) } catch (e) { setErro(e.message) }
  }, [])

  const focar = useCallback(async (id) => {
    focoId.current = id
    try {
      const e = await detalharTreino(id)
      if (focoId.current === id) setTreino(e)
    } catch (e) { setErro(e.message) }
  }, [])

  // Ao abrir: histórico e, se algum treino ainda roda no servidor, volta a acompanhá-lo.
  useEffect(() => {
    listarTreinos().then(({ experimentos }) => {
      setItens(experimentos)
      const rodando = experimentos.find((x) => EM_ANDAMENTO.includes(x.status))
      if (rodando) focar(rodando.id)
    }).catch((e) => setErro(e.message))
  }, [focar])

  // O resumo dos dados muda com "negativos por troca real" (com um respiro para o slider não disparar 10 buscas).
  useEffect(() => {
    const t = setTimeout(() => {
      resumoDados(params.neg_por_pos, params.usar_externos).then((d) => { setDados(d); setErroDados(null) })
        .catch((e) => setErroDados(`Não foi possível ler os dados: ${e.message}`))
    }, 350)
    return () => clearTimeout(t)
  }, [params.neg_por_pos, params.usar_externos])

  // Enquanto o experimento em foco está rodando, pergunta o progresso a cada segundo.
  const rodando = treino && EM_ANDAMENTO.includes(treino.status)
  useEffect(() => {
    if (!rodando) return
    const t = setInterval(() => focar(treino.id), 1000)
    return () => clearInterval(t)
  }, [rodando, treino?.id, focar])
  useEffect(() => { if (treino && !rodando) carregarHistorico() }, [treino?.status, rodando, carregarHistorico])

  async function treinar() {
    setErro(null); setIniciando(true)
    try {
      const { id } = await iniciarTreino(params)
      setTreino({ id, status: 'preparando', params, historico: [], epocas_max: params.epocas })
      await focar(id)
    } catch (e) { setErro(e.message) } finally { setIniciando(false) }
  }

  async function agir(fn) {
    setErro(null); setOcupado(true)
    try { await fn(); await focar(treino.id); await carregarHistorico() }
    catch (e) { setErro(e.message) } finally { setOcupado(false) }
  }

  return (
    <main className="mx-auto max-w-6xl space-y-6 px-6 py-8">
      <header>
        <h1 className="text-2xl font-bold">Laboratório de treino</h1>
        <p className="mt-1 max-w-3xl text-stone-600">
          Aqui você mexe nos botões da rede neural e vê o que acontece: como os dados foram separados, como o erro cai
          (ou não) a cada época, e se a rede realmente supera a busca simples. Quando gostar do resultado, salve o modelo e,
          se quiser, ative-o no app.
        </p>
      </header>

      {erro && <p className="rounded-xl bg-red-50 p-4 text-sm text-red-800">{erro}</p>}

      <PassoDados dados={dados} erro={erroDados} />
      <PassoConfig params={params} setParams={setParams} desabilitado={!!rodando}
        dim={dados?.dim_entrada ?? 1958}
        nExemplos={dados ? dados.exemplos.treino.positivos + dados.exemplos.treino.negativos : null} />
      <PassoTreino treino={treino} iniciando={iniciando} onTreinar={treinar}
        onParar={() => pararTreino(treino.id).catch((e) => setErro(e.message))} />
      <PassoResultado treino={treino} ocupado={ocupado}
        onSalvar={(nome) => agir(() => salvarModelo(treino.id, nome))}
        onAtivar={() => agir(() => ativarModelo(treino.id))}
        onDesfazer={() => agir(desfazerAtivacao)} />
      <Historico itens={itens} focoId={treino?.id} onVer={focar}
        onReusar={(p) => { setParams(p); window.scrollTo({ top: 0, behavior: 'smooth' }) }} />
    </main>
  )
}
