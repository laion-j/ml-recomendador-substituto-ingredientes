// O tfjs-node 4.22 ainda chama util.isNullOrUndefined, removido no Node 24.
// Recolocamos a função antes de qualquer teste carregar o tfjs-node.
// (Só afeta os testes em Node; no navegador o tfjs puro não usa isso.)
import util from 'node:util'

util.isNullOrUndefined ??= (v) => v == null
