"""Prepara os pares da Recipe1MSubs (GISMo, Meta) como fonte EXTRA de treino.

Uso: .venv-trad/bin/python scripts_ml/07b_recipe1msubs_preparar.py     (precisa de ctranslate2; ver 07a)
Entrada: data/pares_limpos.csv, data/pares_split.csv, data/vocabulario.csv, traducao_ingredientes.json,
         data/raw/gismo/{train,val,test}_comments_subs.pkl (baixados aqui se faltarem)
Saída:   data/recipe1msubs.json  (receitas externas com ingredientes em PT + pares no vocabulário)
Depois:  .venv/bin/python scripts_ml/07c_recipe1msubs_carregar.py

Regras que protegem a avaliação:
  - os pares externos só vão para o TREINO; qualquer (original, substituto) que esteja na nossa validação ou no nosso
    teste é descartado daqui (senão o modelo treinaria com o gabarito da avaliação);
  - teto de MAX_POR_PAR ocorrências por par somando as duas fontes (mesmo teto do 03), para um par popular não dominar.
A fonte só traz a lista de ingredientes da receita (sem título, passos ou review), e o id é um hash do Recipe1M.
"""
import importlib.util
import json
import pickle
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
GISMO = RAIZ / "data" / "raw" / "gismo"
URL = "https://dl.fbaipublicfiles.com/gismo/{}_comments_subs.pkl"
MAX_POR_PAR = 20


def script(nome: str):
    spec = importlib.util.spec_from_file_location(nome, RAIZ / "scripts_ml" / f"{nome}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class SoDados(pickle.Unpickler):
    """Pickle de fonte externa pode executar código ao carregar. Estes arquivos só têm list/dict/str/tuple:
    recusamos qualquer import, então um arquivo adulterado falha em vez de rodar."""
    def find_class(self, modulo, nome):
        raise pickle.UnpicklingError(f"import bloqueado: {modulo}.{nome}")


def ler_gismo() -> list[dict]:
    GISMO.mkdir(parents=True, exist_ok=True)
    itens = []
    for sp in ("train", "val", "test"):   # os splits do GISMo não importam aqui: usamos o nosso, por par único
        arq = GISMO / f"{sp}_comments_subs.pkl"
        if not arq.exists():
            print("baixando", arq.name)
            urllib.request.urlretrieve(URL.format(sp), arq)
        with open(arq, "rb") as f:
            itens += SoDados(f).load()
    return itens


def main() -> None:
    mineracao = script("02_minerar_pares")
    token, normalizar = mineracao.TOKEN, mineracao.normalizar
    # "ground_beef" -> "beef", "olive_oil" -> "olive oil": mesma normalização usada para montar o nosso vocabulário
    norm = lambda nome: normalizar(token.findall(nome.replace("_", " ").lower()))

    vocab = pd.read_csv(RAIZ / "data" / "vocabulario.csv")["ingrediente"].tolist()
    no_vocab = set(vocab)
    split = pd.read_csv(RAIZ / "data" / "pares_split.csv")
    proibidos = {(r.original, r.substituto) for r in split[split.split != "treino"].itertuples()}   # validação e teste
    nossos = Counter(zip(*[pd.read_csv(RAIZ / "data" / "pares_limpos.csv")[c] for c in ("original", "substituto")]))

    itens = ler_gismo()
    print(f"itens do GISMo: {len(itens)}")

    # ---- 1) filtros (mesmos do 03) e casamento com o vocabulário
    perdas = Counter()
    cand = []   # (gismo_id, original, substituto, ingredientes [variante canônica de cada ingrediente])
    vistos = set()
    for x in itens:
        o, s = norm(x["subs"][0]), norm(x["subs"][1])
        if o not in no_vocab or s not in no_vocab:   perdas["fora do vocabulário"] += 1; continue
        if o == s or o in s or s in o:               perdas["igual ou contido um no outro"] += 1; continue
        if (o, s) in proibidos:                      perdas["par da validação/teste"] += 1; continue
        if (x["id"], o, s) in vistos:                perdas["repetido na mesma receita"] += 1; continue
        vistos.add((x["id"], o, s))
        cand.append((x["id"], o, s, [v[0] for v in x["ingredients"]]))
    print("descartados:", dict(perdas), "| ficam:", len(cand))

    # ---- 2) teto por par, somando as nossas ocorrências (sorteio com semente fixa)
    random.Random(42).shuffle(cand)
    usados = Counter()
    pares = []
    for gid, o, s, ings in cand:
        if nossos[(o, s)] + usados[(o, s)] >= MAX_POR_PAR:
            perdas["acima do teto por par"] += 1; continue
        usados[(o, s)] += 1
        pares.append((gid, o, s, ings))
    print(f"após o teto de {MAX_POR_PAR}/par: {len(pares)} ocorrências, {len(usados)} pares únicos "
          f"({sum(1 for p in usados if nossos[p] == 0)} pares que a nossa mineração não tem)")

    # ---- 3) receitas externas: só os ingredientes, traduzidos para PT como as nossas
    por_receita: dict[str, list[str]] = {}
    for gid, _, _, ings in pares:
        por_receita[gid] = ings
    nomes = sorted({i for ings in por_receita.values() for i in ings})
    print(f"receitas externas: {len(por_receita)} | nomes de ingrediente distintos: {len(nomes)}")

    trad_dic = json.loads((RAIZ / "scripts_ml" / "traducao_ingredientes.json").read_text(encoding="utf-8"))
    dic = {en: re.sub(r" \(.*\)$", "", pt) for en, pt in trad_dic.items() if en != "_meta"}   # sem o sufixo de desambiguação
    legivel = lambda n: n.replace("_", " ")
    ingl_para_trad = {n: legivel(n) for n in nomes if norm(n) not in dic}        # o que o dicionário manual não cobre
    trad07a = script("07a_traduzir_receitas")
    tok, tr = trad07a.carregar()
    auto = trad07a.traduzir(list(ingl_para_trad.values()), tok, tr)
    pt_de = {n: dic[norm(n)] if norm(n) in dic else auto.get(legivel(n), legivel(n)) for n in nomes}

    rec_ids = {gid: -(i + 1) for i, gid in enumerate(sorted(por_receita))}      # id negativo = receita externa
    saida = {
        "fonte": "Recipe1MSubs / GISMo (Meta)",
        "receitas": {str(rec_ids[g]): {"gismo_id": g, "ingredientes_en": [legivel(i) for i in ings],
                                       "ingredientes_pt": [pt_de[i] for i in ings]} for g, ings in por_receita.items()},
        "pares": [[rec_ids[g], o, s] for g, o, s, _ in pares],
        "estatisticas": {"itens_gismo": len(itens), "descartes": dict(perdas), "ocorrencias": len(pares),
                         "pares_unicos": len(usados), "receitas": len(por_receita)},
    }
    destino = RAIZ / "data" / "recipe1msubs.json"
    destino.write_text(json.dumps(saida, ensure_ascii=False), encoding="utf-8")
    print("salvo em", destino)
    print("exemplo:", next(iter(saida["receitas"].values())))


if __name__ == "__main__":
    main()
