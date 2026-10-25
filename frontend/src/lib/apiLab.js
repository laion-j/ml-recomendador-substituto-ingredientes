// Cliente do laboratório de treino (/api/lab). Reaproveita o chamar() do api.js.
import { chamar } from './api.js'

const json = (corpo) => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(corpo) })

export const resumoDados = (negPorPos, externos = false) => chamar(`/api/lab/dados?neg_por_pos=${negPorPos}&externos=${externos}`)
export const listarTreinos = () => chamar('/api/lab/treinos')
export const detalharTreino = (id) => chamar(`/api/lab/treinos/${id}`)
export const iniciarTreino = (params) => chamar('/api/lab/treinos', json(params))
export const pararTreino = (id) => chamar(`/api/lab/treinos/${id}/parar`, { method: 'POST' })
export const salvarModelo = (id, nome) => chamar(`/api/lab/treinos/${id}/salvar`, json({ nome }))
export const ativarModelo = (id) => chamar(`/api/lab/treinos/${id}/ativar`, { method: 'POST' })
export const desfazerAtivacao = () => chamar('/api/lab/ativo/desfazer', { method: 'POST' })
