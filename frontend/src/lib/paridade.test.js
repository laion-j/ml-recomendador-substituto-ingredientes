// Paridade Keras vs TFJS: o mesmo input tem que dar o mesmo score (tolerância 1e-4).
// Roda só quando o modelo exportado e as fixtures existem em public/models/.
import { describe, it, expect } from 'vitest'
import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'

const dir = resolve(import.meta.dirname, '../../public/models')
const pronto = ['model.json', 'feature_spec.json', 'paridade_fixtures.json'].every((f) => existsSync(resolve(dir, f)))

describe.skipIf(!pronto)('paridade com o Keras', () => {
  it('scores do TFJS batem com os da fixture', async () => {
    const tf = await import('@tensorflow/tfjs-node')
    const fx = JSON.parse(readFileSync(resolve(dir, 'paridade_fixtures.json'), 'utf8'))
    const model = await tf.loadLayersModel(`file://${resolve(dir, 'model.json')}`)
    // formato esperado da fixture: { entradas: [[...dim]], scores: [...] }
    const scores = tf.tidy(() => model.predict(tf.tensor2d(fx.entradas)).dataSync())
    fx.scores.forEach((esperado, i) => expect(Math.abs(scores[i] - esperado)).toBeLessThan(1e-4))
  })
})
