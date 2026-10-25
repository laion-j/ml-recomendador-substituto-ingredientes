import { describe, it, expect } from 'vitest'
import { construirFeatures, construirLote } from './features.js'

// Fixture sintética: embeddings de 2 dims e vocabulários minúsculos.
// dim_entrada = 5*2 + 1 + 2 (cat) + 3 (funcoes) + 2 (sabores) = 18
const spec = { dim_entrada: 18, categorias: ['laticinio', 'gordura'], funcoes: ['ligante', 'fermento', 'gordura'], sabores: ['doce', 'salgado'] }
const r = [0.1, 0.2]
const o = [1, 2]
const cand = { embedding: [3, 1], distancia: 0.25, categoria: 'gordura', funcoes: ['gordura', 'ligante'], sabores: ['salgado', 'desconhecido'] }

describe('construirFeatures', () => {
  it('respeita a ordem e o tamanho', () => {
    expect(construirFeatures(r, o, cand, spec)).toEqual([
      0.1, 0.2, // receita
      1, 2, // original
      3, 1, // candidato
      3, 2, // o*c
      2, 1, // |o-c|
      0.25, // distância
      0, 1, // categoria
      1, 0, 1, // funções (alfabético: ligante, fermento, gordura)
      0, 1, // sabores ('desconhecido' ignorado)
    ])
  })

  it('falha se o tamanho não bate com dim_entrada', () => {
    expect(() => construirFeatures(r, o, cand, { ...spec, dim_entrada: 99 })).toThrow(/99/)
  })

  it('construirLote gera uma linha por candidato', () => {
    const lote = construirLote({ receita_embedding: r, ingrediente_embedding: o, candidatos: [cand, cand] }, spec)
    expect(lote).toHaveLength(2)
    expect(lote[0]).toHaveLength(18)
  })
})
