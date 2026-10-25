"""Paridade Keras x TFJS: 20 exemplos de VALIDAÇÃO, saída do Keras gravada em fixture e comparada no Node.

Uso: .venv-tf/bin/python scripts_ml/13_paridade.py
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
import subprocess

import numpy as np
from tensorflow import keras

import features as F

MODELOS = F.RAIZ / "frontend" / "public" / "models"
d = F.carregar()
z = np.load(F.RAIZ / "data" / "dataset_reranker.npz")
idx = np.random.default_rng(F.SEED).choice(len(z["va_y"]), 20, replace=False)
X = F.montar(d, z["va_r"][idx], z["va_o"][idx], z["va_c"][idx])                 # (20, dim_entrada)
y = keras.models.load_model(F.RAIZ / "data" / "reranker.h5").predict(X, verbose=0)[:, 0]
(MODELOS / "paridade_fixtures.json").write_text(json.dumps({"entradas": X.tolist(), "scores": y.tolist()}))

# o Node (tfjs-node) carrega o model.json EXPORTADO e compara com os scores do Keras
node = F.RAIZ / "scripts_ml" / "node_check"
subprocess.run(["node", "--require", "./shim.cjs", "paridade.mjs", str(MODELOS)], cwd=node, check=True)
