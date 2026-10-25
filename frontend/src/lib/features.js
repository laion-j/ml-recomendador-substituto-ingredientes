// Construtor de features do reranker. Função PURA (sem tf, sem rede): fácil de testar.
// O treino (scripts_ml) usa EXATAMENTE esta ordem; mudar aqui sem mudar lá quebra o modelo.

// Lista de nomes -> vetor multi-hot sobre o vocabulário (índice = posição no vocabulário).
// Nome fora do vocabulário é ignorado (fica tudo 0), igual ao treino.
const multiHot = (itens, vocab) => {
  const v = new Array(vocab.length).fill(0)
  for (const item of itens ?? []) {
    const i = vocab.indexOf(item)
    if (i >= 0) v[i] = 1
  }
  return v
}

// Monta o vetor de UM candidato:
// [r(D), o(D), c(D), o*c(D), |o-c|(D), distancia(1), cat multi-hot, funcoes multi-hot, sabores multi-hot]
export function construirFeatures(receitaEmb, ingredienteEmb, candidato, spec) {
  const r = receitaEmb
  const o = ingredienteEmb
  const c = candidato.embedding

  // produto elemento a elemento: dimensões em que original e candidato "concordam"
  const oc = o.map((x, i) => x * c[i])
  // diferença absoluta: o quanto divergem em cada dimensão
  const dif = o.map((x, i) => Math.abs(x - c[i]))

  const vetor = [
    ...r, ...o, ...c, ...oc, ...dif,
    candidato.distancia, // cosseno do pgvector (0 = idêntico), entra como feature numérica
    ...multiHot([candidato.categoria], spec.categorias), // categoria é única -> one-hot na prática
    ...multiHot(candidato.funcoes, spec.funcoes),
    ...multiHot(candidato.sabores, spec.sabores),
  ]

  // trava de segurança: se o tamanho não bate, o modelo receberia lixo silenciosamente
  if (vetor.length !== spec.dim_entrada) {
    throw new Error(`features com ${vetor.length} dims, o modelo espera ${spec.dim_entrada}`)
  }
  return vetor
}

// Um vetor por candidato, na mesma ordem da lista (a linha i do tensor = candidato i).
export const construirLote = (resp, spec) =>
  resp.candidatos.map((c) => construirFeatures(resp.receita_embedding, resp.ingrediente_embedding, c, spec))
