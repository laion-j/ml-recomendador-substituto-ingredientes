// Carrega o model.json exportado, roda as entradas da fixture e compara com os scores do Keras.
import * as tf from "@tensorflow/tfjs-node";
import { readFileSync } from "node:fs";
const dir = process.argv[2];
const { entradas, scores } = JSON.parse(readFileSync(`${dir}/paridade_fixtures.json`, "utf8"));
const modelo = await tf.loadLayersModel(`file://${dir}/model.json`);
const saida = Array.from(modelo.predict(tf.tensor2d(entradas)).dataSync());   // (20,1) -> 20 floats
const dif = Math.max(...saida.map((v, i) => Math.abs(v - scores[i])));
console.log(`exemplos: ${saida.length} | diferença máxima Keras x TFJS: ${dif.toExponential(3)}`);
console.log(dif < 1e-4 ? "PARIDADE OK" : "PARIDADE FALHOU");
process.exit(dif < 1e-4 ? 0 : 1);
