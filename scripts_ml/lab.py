"""Laboratório de treino: motor usado pela API (/api/lab) para experimentar o reranker com parâmetros variáveis.

Uso (quem chama é o backend; roda no .venv-tf):
  lab.py dados   [--neg-por-pos 5]      -> JSON no stdout: como os dados foram separados e que exemplos saem deles
  lab.py treinar PASTA                  -> lê PASTA/params.json, treina, avalia na VALIDAÇÃO, grava PASTA/progresso.json
  lab.py exportar PASTA --nome NOME     -> converte PASTA/modelo.h5 para tfjs em PASTA/tfjs e confere a paridade no Node

Cada experimento é uma pasta em data/lab/<id>/. O 10_treinar/11_avaliar/12_exportar continuam sendo o caminho "oficial";
aqui reaproveitamos a mesma avaliação (11_avaliar.avaliar) e a mesma montagem de dataset (09_dataset.montar_dataset).
O split TESTE nunca é lido.
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"        # Keras 2 (tf_keras): é o que o conversor tfjs entende
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

import features as F


def script(nome: str):
    """Importa um script numerado (nome começa com dígito, então não dá para usar `import`)."""
    spec = importlib.util.spec_from_file_location(nome, F.RAIZ / "scripts_ml" / f"{nome}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def gravar(pasta: Path, **campos) -> dict:
    """Atualiza progresso.json de forma atômica (o front lê enquanto escrevemos: nunca pode ver arquivo pela metade)."""
    arq = pasta / "progresso.json"
    estado = json.loads(arq.read_text(encoding="utf-8")) if arq.exists() else {}
    estado.update(campos)
    tmp = pasta / "progresso.json.tmp"
    tmp.write_text(json.dumps(estado, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, arq)
    return estado


# ---------------------------------------------------------------- dados
def dados(neg_por_pos: int, externos: bool = False) -> None:
    """Resume o que o 04/07/09 fizeram: separação em treino/validação/teste e os exemplos que cada um gera."""
    d = F.carregar(externos)
    tr, va = script("09_dataset").montar_dataset(d, neg_por_pos)

    splits = {}
    for sp in ("treino", "validacao", "teste"):
        ocs = [(o, s) for o, s, r, x in d["pares"] if x == sp]
        # receita de id negativo = veio da Recipe1MSubs (07c); só existe no treino
        externas = sum(1 for o, s, r, x in d["pares"] if x == sp and d["rec_ids"][r] < 0)
        splits[sp] = {"pares_unicos": len(set(ocs)), "ocorrencias": len(ocs), "ocorrencias_externas": externas}
    proprios = {(o, s) for o, s, r, x in d["pares"] if x == "treino" and d["rec_ids"][r] > 0}
    splits["treino"]["pares_so_externos"] = len({(o, s) for o, s, r, x in d["pares"] if x == "treino"} - proprios)
    val_unicos = {(o, s) for o, s, r, x in d["pares"] if x == "validacao"}
    rankeaveis = sum(1 for o, s in val_unicos if s in d["topk"][o])   # substituto dentro do top-30 do pgvector

    # "cartões" de exemplo: 4 trocas reais e 4 vizinhos que ninguém usou como troca (hard negatives)
    rng = np.random.default_rng(7)
    escolhidos = [(i, 1) for i in rng.choice(np.flatnonzero(tr[3] == 1), 4, replace=False)] + \
                 [(i, 0) for i in rng.choice(np.flatnonzero(tr[3] == 0), 4, replace=False)]
    ids = [d["rec_ids"][tr[0][i]] for i, _ in escolhidos]
    with F.conectar() as con:
        titulos = dict(con.execute("SELECT id, titulo FROM receitas WHERE id = ANY(%s)", (ids,)).fetchall())
    amostra = [{"original": d["nomes"][tr[1][i]], "candidato": d["nomes"][tr[2][i]], "rotulo": int(y),
                "receita": titulos.get(d["rec_ids"][tr[0][i]], "?") if d["rec_ids"][tr[0][i]] > 0
                else "Recipe1MSubs (só a lista de ingredientes)"} for i, y in escolhidos]

    print(json.dumps({
        "neg_por_pos": neg_por_pos, "externos": externos, "k": F.K,
        "ingredientes": len(d["nomes"]), "receitas": len(d["rec_ids"]),
        "dim_entrada": 5 * 384 + 1 + d["tags"].shape[1],
        "splits": splits,
        "exemplos": {"treino": {"positivos": int(tr[3].sum()), "negativos": int((1 - tr[3]).sum())},
                     "validacao": {"positivos": int(va[3].sum()), "negativos": int((1 - va[3]).sum())}},
        "validacao_rankeaveis": {"pares": rankeaveis, "total": len(val_unicos)},
        "amostra": amostra,
    }, ensure_ascii=False))


# ---------------------------------------------------------------- treino
def construir(dim: int, p: dict):
    """MLP configurável. Mesmo desenho do 10_treinar quando p = padrão: dropout em todas as ocultas menos a última."""
    from tensorflow import keras
    reg = keras.regularizers.l2(p["l2"]) if p["l2"] > 0 else None     # L2 pune pesos grandes: freio contra overfit
    m = keras.Sequential()
    for i, n in enumerate(p["camadas"]):
        kw = {"input_shape": (dim,)} if i == 0 else {}
        m.add(keras.layers.Dense(n, activation=p["ativacao"], kernel_regularizer=reg, **kw))
        if p["dropout"] > 0 and i < len(p["camadas"]) - 1:
            m.add(keras.layers.Dropout(p["dropout"]))   # desliga neurônios ao acaso no treino: evita decorar
    m.add(keras.layers.Dense(1, activation="sigmoid"))   # 1 neurônio, 0..1: score de ranking
    otim = {"adam": keras.optimizers.Adam, "rmsprop": keras.optimizers.RMSprop,
            "sgd": lambda learning_rate: keras.optimizers.SGD(learning_rate, momentum=0.9)}[p["otimizador"]]
    m.compile(optimizer=otim(learning_rate=p["taxa"]), loss="binary_crossentropy")
    return m


def treinar(pasta: Path) -> None:
    from tensorflow import keras
    p = json.loads((pasta / "params.json").read_text(encoding="utf-8"))
    try:
        keras.utils.set_random_seed(p["seed"])
        gravar(pasta, status="preparando", inicio=time.time())
        d = F.carregar(p.get("usar_externos", False))
        tr, va = script("09_dataset").montar_dataset(d, p["neg_por_pos"], p["seed"])
        # índices -> matriz (n, dim_entrada): é aqui que embeddings, produtos e tags viram o tensor de entrada
        X_tr, y_tr = F.montar(d, tr[0], tr[1], tr[2]), tr[3].astype(np.float32)
        X_va, y_va = F.montar(d, va[0], va[1], va[2]), va[3].astype(np.float32)
        modelo = construir(X_tr.shape[1], p)
        gravar(pasta, status="treinando", dim_entrada=int(X_tr.shape[1]), params_totais=int(modelo.count_params()),
               n_treino=int(len(y_tr)), n_validacao=int(len(y_va)), historico=[], melhor_epoca=None, epocas_max=p["epocas"])

        class Acompanhar(keras.callbacks.Callback):
            """Grava a curva a cada época, guarda os melhores pesos, aplica paciência e atende o pedido de parada."""
            def __init__(self):
                super().__init__()
                self.hist, self.melhor, self.melhor_epoca, self.pesos, self.sem_melhora = [], np.inf, None, None, 0
                self.t0, self.motivo = time.time(), "epocas"

            def on_epoch_end(self, epoca, logs):
                self.hist.append({"epoca": epoca + 1, "loss": float(logs["loss"]), "val_loss": float(logs["val_loss"]),
                                  "seg": round(time.time() - self.t0, 1)})
                if logs["val_loss"] < self.melhor - 1e-6:
                    self.melhor, self.melhor_epoca, self.sem_melhora = float(logs["val_loss"]), epoca + 1, 0
                    self.pesos = self.model.get_weights()
                else:
                    self.sem_melhora += 1
                if (pasta / "parar").exists():
                    self.motivo, self.model.stop_training = "usuario", True
                elif p["paciencia"] > 0 and self.sem_melhora >= p["paciencia"]:
                    self.motivo, self.model.stop_training = "paciencia", True
                gravar(pasta, historico=self.hist, melhor_epoca=self.melhor_epoca)

        acomp = Acompanhar()
        modelo.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=p["epocas"], batch_size=p["lote"],
                   callbacks=[acomp], verbose=0)
        if p["restaurar_melhor"] and acomp.pesos is not None:
            modelo.set_weights(acomp.pesos)          # volta à época com menor val_loss (a rede "desfaz" o overfit)
        modelo.save(pasta / "modelo.h5")

        gravar(pasta, status="avaliando", motivo_fim=acomp.motivo)
        res, n, nr = script("11_avaliar").avaliar(modelo, d)
        nomes = ("mrr", "ndcg10", "r5", "r10")
        metricas = {"n_pares": n, "n_rankeaveis": nr,
                    "sistemas": {k: {"todos": dict(zip(nomes, map(float, v[0]))),
                                     "rankeaveis": dict(zip(nomes, map(float, v[1])))} for k, v in res.items()}}
        gravar(pasta, status="parado" if acomp.motivo == "usuario" else "concluido", metricas=metricas, fim=time.time())
    except Exception:
        gravar(pasta, status="falhou", erro=traceback.format_exc()[-1500:], fim=time.time())
        raise


# ---------------------------------------------------------------- exportar
def exportar(pasta: Path, nome: str) -> None:
    """h5 -> tfjs (model.json + .bin) + feature_spec + fixtures; o Node confere que o TFJS dá a mesma saída do Keras."""
    from tensorflow import keras
    p = json.loads((pasta / "params.json").read_text(encoding="utf-8"))
    pasta = pasta.resolve()                # o Node roda em outra pasta: caminho relativo quebraria
    saida = pasta / "tfjs"
    shutil.rmtree(saida, ignore_errors=True); saida.mkdir()
    conversor = Path(sys.executable).parent / "tensorflowjs_converter"
    subprocess.run([str(conversor), "--input_format=keras", "--output_format=tfjs_layers_model",
                    str(pasta / "modelo.h5"), str(saida)], check=True, capture_output=True)

    d = F.carregar(p.get("usar_externos", False))
    (saida / "feature_spec.json").write_text(json.dumps(F.feature_spec(d), ensure_ascii=False, indent=2), encoding="utf-8")
    _, va = script("09_dataset").montar_dataset(d, p["neg_por_pos"], p["seed"])
    idx = np.random.default_rng(F.SEED).choice(len(va[3]), 20, replace=False)
    X = F.montar(d, va[0][idx], va[1][idx], va[2][idx])
    y = keras.models.load_model(pasta / "modelo.h5").predict(X, verbose=0)[:, 0]
    (saida / "paridade_fixtures.json").write_text(json.dumps({"entradas": X.tolist(), "scores": y.tolist()}))

    r = subprocess.run(["node", "--require", "./shim.cjs", "paridade.mjs", str(saida)],
                       cwd=F.RAIZ / "scripts_ml" / "node_check", capture_output=True, text=True)
    gravar(pasta, salvo={"nome": nome, "em": datetime.now().isoformat(timespec="seconds"),
                         "paridade_ok": r.returncode == 0, "paridade": (r.stdout or r.stderr).strip()[-300:]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    dd = sub.add_parser("dados"); dd.add_argument("--neg-por-pos", type=int, default=5); dd.add_argument("--externos", action="store_true")
    sub.add_parser("treinar").add_argument("pasta", type=Path)
    e = sub.add_parser("exportar"); e.add_argument("pasta", type=Path); e.add_argument("--nome", required=True)
    a = ap.parse_args()
    if a.cmd == "dados": dados(a.neg_por_pos, a.externos)
    elif a.cmd == "treinar": treinar(a.pasta)
    else: exportar(a.pasta, a.nome)
