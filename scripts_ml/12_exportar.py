"""Exporta data/reranker.h5 para o formato tfjs (model.json + .bin) e grava o feature_spec.json.

Uso: .venv-tf/bin/python scripts_ml/12_exportar.py
Saída: frontend/public/models/{model.json, group1-shard1of1.bin, feature_spec.json}
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import features as F

SAIDA = F.RAIZ / "frontend" / "public" / "models"
SAIDA.mkdir(parents=True, exist_ok=True)
for f in SAIDA.glob("group1-shard*"): f.unlink()                 # evita .bin velho de uma exportação anterior

# --input_format=keras lê o HDF5 do Keras 2; o conversor fica no mesmo venv que o python atual
conversor = Path(sys.executable).parent / "tensorflowjs_converter"
subprocess.run([str(conversor), "--input_format=keras", "--output_format=tfjs_layers_model",
                str(F.RAIZ / "data" / "reranker.h5"), str(SAIDA)], check=True)

spec = F.feature_spec(F.carregar())
(SAIDA / "feature_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
print("dim_entrada =", spec["dim_entrada"])
for f in sorted(SAIDA.iterdir()): print(f"{f.name}: {f.stat().st_size} bytes")
