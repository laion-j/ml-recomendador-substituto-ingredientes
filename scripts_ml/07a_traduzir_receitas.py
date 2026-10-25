"""Traduz receitas (titulo, ingredientes, passos) de EN para PT-BR, localmente, com Helsinki-NLP/opus-mt-tc-big-en-pt.

Uso: .venv-trad/bin/python scripts_ml/07a_traduzir_receitas.py [--limite N]
Entrada: data/pares_limpos.csv, data/pares_split.csv, data/raw/RAW_recipes.csv  (mesmo filtro do 07_receitas.py)
Saída:   data/receitas_pt.json  {id: {"titulo", "ingredientes", "passos"}}; o 07_receitas.py usa se existir.
Modelo convertido para CTranslate2 (int8) em data/opus-en-pt-ct2:
  ct2-transformers-converter --model Helsinki-NLP/opus-mt-tc-big-en-pt --output_dir data/opus-en-pt-ct2 --quantization int8
"""
import argparse
import ast
import json
import re
import time
from pathlib import Path

import ctranslate2
import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import MarianTokenizer

RAIZ = Path(__file__).resolve().parent.parent
HF = "Helsinki-NLP/opus-mt-tc-big-en-pt"
LOTE = 64


def carregar():
    # O tc-big é multi-destino: o prefixo ">>por<<" escolhe português.
    tok = MarianTokenizer.from_pretrained(HF)
    tr = ctranslate2.Translator(str(RAIZ / "data" / "opus-en-pt-ct2"), device="auto", compute_type="int8",
                                inter_threads=2, intra_threads=8)
    return tok, tr


def traduzir(textos: list[str], tok, tr) -> dict[str, str]:
    """Traduz textos únicos. Ordena por tamanho para os lotes não terem padding enorme."""
    unicos = sorted({t for t in textos if t and t.strip()}, key=len)
    saida, t0 = {}, time.time()
    for i in range(0, len(unicos), LOTE):
        lote = unicos[i:i + LOTE]
        # texto -> tokens subword (strings), que é o que o CTranslate2 consome
        fontes = [tok.convert_ids_to_tokens(tok(">>por<< " + t, truncation=True, max_length=256)["input_ids"]) for t in lote]
        res = tr.translate_batch(fontes, beam_size=2, max_decoding_length=300)
        for t, r in zip(lote, res):
            saida[t] = tok.decode(tok.convert_tokens_to_ids(r.hypotheses[0]), skip_special_tokens=True)
        if (i // LOTE) % 20 == 0:
            print(f"  {i + len(lote)}/{len(unicos)} textos | {time.time() - t0:.0f}s", flush=True)
    return saida


def main(limite: int | None) -> None:
    split = pd.read_csv(RAIZ / "data" / "pares_split.csv")[["original", "substituto"]]
    oc = pd.read_csv(RAIZ / "data" / "pares_limpos.csv").merge(split, on=["original", "substituto"])
    rec = pd.read_csv(RAIZ / "data" / "raw" / "RAW_recipes.csv", usecols=["id", "name", "steps", "ingredients"])
    rec = rec[rec["id"].isin(oc.receita_id.unique())].drop_duplicates("id")
    if limite:
        rec = rec.head(limite)
    rec["ingredientes"] = rec["ingredients"].map(ast.literal_eval)
    rec["passos"] = rec["steps"].map(ast.literal_eval)
    print(f"{len(rec)} receitas")

    tok, tr = carregar()
    print("titulos"); t_tit = traduzir(rec["name"].tolist(), tok, tr)
    print("ingredientes"); t_ing = traduzir([i for l in rec["ingredientes"] for i in l], tok, tr)
    print("passos"); t_pas = traduzir([p for l in rec["passos"] for p in l], tok, tr)

    # Ingrediente do vocabulário usa o nosso dicionário (consistente com ingredientes.nome); o resto, a tradução automática.
    # O sufixo "(original)" só desambigua a coluna UNIQUE, não faz sentido na lista da receita.
    dic = json.load(open(RAIZ / "scripts_ml" / "traducao_ingredientes.json", encoding="utf-8"))
    dic = {en: re.sub(r" \(.*\)$", "", pt) for en, pt in dic.items() if en != "_meta"}
    t_ing = {**t_ing, **dic}

    out = {int(r.id): {"titulo": t_tit.get(r.name, r.name),
                       "ingredientes": [t_ing.get(i, i) for i in r.ingredientes],
                       "passos": [t_pas.get(p, p) for p in r.passos]}
           for r in rec.itertuples()}
    destino = RAIZ / "data" / ("receitas_pt_amostra.json" if limite else "receitas_pt.json")
    json.dump(out, open(destino, "w", encoding="utf-8"), ensure_ascii=False)
    print("salvo em", destino)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int)
    main(ap.parse_args().limite)
