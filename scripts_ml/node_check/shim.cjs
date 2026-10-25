// Node 24 removeu util.isNullOrUndefined, que o tfjs-node 4.22 ainda usa: recolocamos antes de carregar o tfjs.
const util = require("util");
for (const f of ["isNullOrUndefined", "isArray", "isNumber", "isString", "isBoolean", "isFunction", "isObject", "isUndefined"]) {
  if (!util[f]) util[f] = f === "isNullOrUndefined" ? (v) => v == null : f === "isArray" ? Array.isArray : (v) => (f === "isUndefined" ? v === undefined : typeof v === f.slice(2).toLowerCase());
}
