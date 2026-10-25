"""Funde os rótulos manuais dos ingredientes novos (rotulos_extra_*.txt) em taxonomia.json e traducao_ingredientes.json.

Uso: .venv/bin/python scripts_ml/14_rotulos.py
Linha: en|pt|categoria|funcoes|sabores   (categoria = código de 3 letras; funcoes e sabores = letras coladas)
Confere contra data/vocabulario.csv: faltando rótulo, rótulo de ingrediente que não está no vocabulário e colisão de nome PT.
"""
import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
PASTA = RAIZ / "scripts_ml"

CAT = {"lat": "laticinio", "liq": "liquido", "ado": "adocante", "pro": "proteina", "gor": "gordura", "fru": "fruta",
       "cas": "castanha", "far": "farinha", "veg": "vegetal", "erv": "erva", "mol": "molho", "tem": "tempero",
       "aci": "acido", "gra": "graos", "leg": "leguminosa", "alc": "bebida_alcoolica", "mas": "massa", "out": "outro"}
FUN = {"G": "gordura", "S": "sabor", "U": "umidade", "A": "adocante", "E": "estrutura", "C": "acidez",
       "T": "espessante", "R": "cor", "F": "fermento", "L": "ligante"}
SAB = {"c": "cremoso", "g": "gorduroso", "d": "doce", "u": "umami", "a": "acido", "s": "salgado",
       "m": "amargo", "p": "picante", "h": "herbal"}


def main() -> None:
    vocab = pd.read_csv(RAIZ / "data" / "vocabulario.csv")["ingrediente"].tolist()
    tax = json.loads((PASTA / "taxonomia.json").read_text(encoding="utf-8"))
    trad = json.loads((PASTA / "traducao_ingredientes.json").read_text(encoding="utf-8"))

    novos = {}
    for arq in sorted(PASTA.glob("rotulos_extra_*.txt")):
        for n, linha in enumerate(arq.read_text(encoding="utf-8").splitlines(), 1):
            if not linha.strip() or linha.startswith("#"):
                continue
            partes = linha.split("|")
            assert len(partes) == 5, f"{arq.name}:{n} esperava 5 campos: {linha!r}"
            en, pt, cat, fun, sab = partes
            assert cat in CAT, f"{arq.name}:{n} categoria inválida {cat!r}"
            assert all(c in FUN for c in fun), f"{arq.name}:{n} função inválida {fun!r}"
            assert all(c in SAB for c in sab), f"{arq.name}:{n} sabor inválido {sab!r}"
            assert en not in novos, f"{arq.name}:{n} repetido: {en}"
            novos[en] = (pt.strip(), {"categoria": CAT[cat], "funcoes": [FUN[c] for c in fun], "sabores": [SAB[c] for c in sab]})

    faltam = [v for v in vocab if v not in tax and v not in novos]
    sobram = [e for e in novos if e not in set(vocab)]
    print(f"rótulos lidos: {len(novos)} | faltando: {len(faltam)} | fora do vocabulário: {len(sobram)}")
    if faltam: print("  FALTAM:", faltam)
    if sobram: print("  SOBRAM:", sobram)

    usados = {v.lower() for k, v in trad.items() if k != "_meta"}
    colisoes = 0
    for en, (pt, t) in novos.items():
        if en in tax:
            continue
        if pt.lower() in usados:                       # nome PT é UNIQUE no banco: desambigua com o original
            pt = f"{pt} ({en})"; colisoes += 1
        usados.add(pt.lower())
        tax[en] = t
        trad[en] = pt
    print(f"colisões de nome PT resolvidas com o original entre parênteses: {colisoes}")
    if faltam:
        raise SystemExit("complete os rótulos antes de gravar")

    # a ordem dos arquivos segue a do vocabulário (a meta fica no topo)
    for d in (tax, trad):
        meta = d["_meta"]
        ordenado = {"_meta": meta, **{v: d[v] for v in vocab}}
        d.clear(); d.update(ordenado)
    (PASTA / "taxonomia.json").write_text(json.dumps(tax, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (PASTA / "traducao_ingredientes.json").write_text(json.dumps(trad, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"taxonomia: {len(tax) - 1} ingredientes | tradução: {len(trad) - 1}")


if __name__ == "__main__":
    main()
