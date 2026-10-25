import { describe, expect, it } from 'vitest'
import { contarParametros, PADRAO } from './params.js'

describe('contarParametros', () => {
  it('bate com o count_params do Keras para a rede padrão (1958 -> 256 -> 128 -> 64 -> 1)', () => {
    expect(contarParametros(PADRAO.camadas, 1958)).toBe(542721)
  })
  it('conta uma camada só: 10 entradas, 4 neurônios, 1 saída = (10*4+4) + (4*1+1)', () => {
    expect(contarParametros([4], 10)).toBe(49)
  })
})
