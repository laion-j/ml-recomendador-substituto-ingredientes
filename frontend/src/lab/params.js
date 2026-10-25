// Parâmetros do laboratório: valores iniciais, atalhos (presets), textos de ajuda e a conta de pesos da rede.
// Os limites daqui espelham backend/app/schemas.py (ParamsTreino); lá é que valem de verdade.

export const PADRAO = {
  camadas: [256, 128, 64], ativacao: 'relu', dropout: 0.3, l2: 0, otimizador: 'adam', taxa: 0.001,
  lote: 256, epocas: 60, paciencia: 5, restaurar_melhor: true, neg_por_pos: 5, usar_externos: false, seed: 42,
}

// Cada preset conta uma história: o que acontece com uma rede pequena, uma sem freio, uma muito freada...
export const PRESETS = [
  { nome: 'Padrão do projeto', nota: 'o mesmo do script 10_treinar', params: PADRAO },
  { nome: 'Rede pequena', nota: 'poucos pesos, quase sem overfit', params: { ...PADRAO, camadas: [64, 32], dropout: 0.2 } },
  { nome: 'Rede grande', nota: 'mais capacidade, mais risco', params: { ...PADRAO, camadas: [512, 256, 128, 64] } },
  { nome: 'Sem freio', nota: 'veja o overfit acontecer', params: { ...PADRAO, camadas: [512, 256], dropout: 0, paciencia: 0, epocas: 40, restaurar_melhor: false } },
  { nome: 'Bem freada', nota: 'dropout alto + L2', params: { ...PADRAO, camadas: [128, 64], dropout: 0.5, l2: 0.001 } },
]

// Medido em 24/10/2026 com o script 10_treinar (PT-BR, validação, pares rankeáveis). Serve de régua na tabela de histórico.
export const REFERENCIA = { r10: 0.611, mrr: 0.221, r10_pgvector: 0.558, mrr_pgvector: 0.227 }

// Pesos de uma camada densa = entradas x saídas (a matriz W) + saídas (o vetor b).
export function contarParametros(camadas, dim) {
  const tamanhos = [dim, ...camadas, 1]
  return tamanhos.slice(1).reduce((soma, saida, i) => soma + tamanhos[i] * saida + saida, 0)
}

export const igual = (a, b) => JSON.stringify(a) === JSON.stringify(b)
