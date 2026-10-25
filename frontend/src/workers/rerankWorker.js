// Web Worker: carrega o modelo uma vez e faz o rerank fora da thread da UI.
// Mensagens (main -> worker): { tipo: 'carregar' } | { tipo: 'rerank', id, resposta }
// Mensagens (worker -> main): { tipo: 'pronto' } | { tipo: 'erro-modelo', mensagem } | { tipo: 'resultado', id, scores } | { tipo: 'erro', id, mensagem }
import * as tf from '@tensorflow/tfjs'
import { construirLote } from '../lib/features.js'

let model = null
let spec = null

async function carregar() {
  // o spec define vocabulários e dim_entrada: precisa ser o mesmo do treino
  const r = await fetch('/models/feature_spec.json')
  const ct = r.headers.get('content-type') ?? ''
  if (!r.ok || ct.includes('text/html')) throw new Error('modelo não encontrado em /models/model.json (feature_spec.json ausente)')
  spec = await r.json()
  try {
    model = await tf.loadLayersModel('/models/model.json')
  } catch (e) {
    throw new Error(`modelo não encontrado em /models/model.json (${e.message})`)
  }
}

function rerank(resposta) {
  // 1 linha de features por candidato -> matriz [n, dim_entrada]
  const entradas = construirLote(resposta, spec)
  // tidy libera o tensor de entrada e o de saída; só o array JS sai daqui
  return tf.tidy(() => {
    const x = tf.tensor2d(entradas, [entradas.length, spec.dim_entrada]) // JS -> tensor float32
    const saida = model.predict(x) // [n, 1], sigmoid: score de ranking
    return Array.from(saida.dataSync()) // tensor -> array JS (sai do tidy como valor, não tensor)
  })
}

self.onmessage = async ({ data }) => {
  try {
    if (data.tipo === 'carregar') {
      await carregar()
      self.postMessage({ tipo: 'pronto' })
    } else if (data.tipo === 'rerank') {
      if (!model) throw new Error('modelo não carregado')
      self.postMessage({ tipo: 'resultado', id: data.id, scores: rerank(data.resposta) })
    }
  } catch (e) {
    self.postMessage({ tipo: data.tipo === 'carregar' ? 'erro-modelo' : 'erro', id: data.id, mensagem: e.message })
  }
}
