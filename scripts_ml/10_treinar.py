"""Treina o reranker (MLP) em Keras 2 (tf_keras), pois só ele exporta um model.json que o tfjs 4.22 carrega.

Uso: .venv-tf/bin/python scripts_ml/10_treinar.py
Entrada: data/dataset_reranker.npz (09). Saída: data/reranker.h5
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"        # tf.keras = Keras 2 (tf_keras); precisa vir ANTES de importar o tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow import keras

import features as F

keras.utils.set_random_seed(F.SEED)            # numpy + python + TF com a mesma semente
d = F.carregar()
z = np.load(F.RAIZ / "data" / "dataset_reranker.npz")

# índices -> matriz (n, dim_entrada): é aqui que texto/ids viram o tensor de entrada
X_tr, y_tr = F.montar(d, z["tr_r"], z["tr_o"], z["tr_c"]), z["tr_y"].astype(np.float32)
X_va, y_va = F.montar(d, z["va_r"], z["va_o"], z["va_c"]), z["va_y"].astype(np.float32)
dim = X_tr.shape[1]
print("X_treino", X_tr.shape, "X_val", X_va.shape)

# MLP 256-128-64: dim_entrada -> camadas densas com ReLU (+ dropout contra overfit) -> 1 neurônio sigmoid (score 0..1)
modelo = keras.Sequential([
    keras.layers.Dense(256, activation="relu", input_shape=(dim,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
modelo.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")

parar = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
modelo.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=60, batch_size=256, callbacks=[parar], verbose=2)
modelo.save(F.RAIZ / "data" / "reranker.h5")
print("salvo em data/reranker.h5")
