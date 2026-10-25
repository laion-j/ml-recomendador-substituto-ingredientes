"""Minera pares (original -> substituto) das reviews do Food.com.

Uso: python scripts_ml/02_minerar_pares.py
Saída: data/pares.csv (receita_id, original, substituto, review) e
       data/amostra_conferencia.csv (50 pares aleatórios para conferência manual).

Ideia: a review diz "usei X no lugar de Y". Se Y está na lista de ingredientes da receita
e X não está, temos um positivo (Y -> X) com o contexto da receita.
"""
import ast
import html
import re
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"
SAIDA = RAIZ / "data"

# --- Vocabulários de limpeza -------------------------------------------------
# Palavras que delimitam o ingrediente quando andamos a partir do conector (verbos, pronomes, ligações).
PARADA = set("""i we you he she they it them this that these those my our your his her its the a an some any
and or but so then also just only really both either not no if when while as at by on in into to from of with
without for is was were are be been being am did do does done had has have having used use using add added adding
subbed sub substituted substitute substituting subs swapped swap swapping replaced replace replacing instead
rather than place made make tried try took take took put putting bought got get let cooked baked would could
should will can may might too very quite even still ended turned came went served love loved liked like
recipe recipes time times because since though although however which what who whom whose how why where
cup cups tablespoon tablespoons tbsp tsp teaspoon teaspoons oz ounce ounces lb lbs pound pounds can cans
package packages pinch dash clove cloves slice slices stick sticks bag bags box jar bottle handful
one two three four five six half quarter""".split())

# Padrões de quantidade: "menos açúcar" não é troca de ingrediente.
QUANTIDADE = {"less", "more", "extra", "half", "fewer", "additional", "double", "triple", "much", "little", "bit", "lot"}

# Adjetivos/estados que não mudam o ingrediente em si.
ADJETIVOS = set("""fresh large small medium big little whole ground chopped diced minced sliced shredded grated
finely coarsely freshly dried frozen canned raw cooked boiled melted softened packed lightly firmly
good quality organic extra virgin low fat nonfat non reduced free sodium lowfat light lite plain
regular real homemade store bought boneless skinless thinly thick thin fine hot cold warm ripe""".split())

# Resultados que são lixo (pronomes, genéricos).
LIXO = set("""it them that this those these one ones thing things something anything everything nothing
other another same different what which stuff ingredient ingredients substitute substitutes mine
her him me us you they he she we i recipe""".split())

TOKEN = re.compile(r"[a-z0-9']+|[,.;:!?()\-/]")


def singular(p: str) -> str:
    # Singular simples: só o fim da palavra; suficiente para ingredientes (tomatoes, berries, eggs).
    if len(p) > 4 and p.endswith("ies"):
        return p[:-3] + "y"
    if len(p) > 4 and p.endswith(("oes", "ches", "shes", "sses")):
        return p[:-2]
    if len(p) > 3 and p.endswith("s") and not p.endswith(("ss", "us")):
        return p[:-1]
    return p


def normalizar(tokens: list[str]) -> str:
    # minúsculas já vêm do tokenizador; aqui tira adjetivos e põe a última palavra no singular.
    t = [x.strip("'") for x in tokens if x.strip("'") and x not in ADJETIVOS]
    if t:
        t[-1] = singular(t[-1])
    return " ".join(t)


def andar(tokens: list[str], i: int, passo: int, maximo: int = 4) -> tuple[list[str], str | None]:
    """Coleta tokens a partir de i (para trás: passo=-1, para frente: +1) até achar parada.
    Devolve (tokens na ordem do texto, token que parou)."""
    saida, parou = [], None
    while 0 <= i < len(tokens) and len(saida) < maximo:
        t = tokens[i]
        if t in PARADA or not re.match(r"[a-z']", t) or t.isdigit():
            parou = t
            break
        saida.append(t)
        i += passo
    if passo == -1:
        saida.reverse()
    return saida, parou


# (conector, onde fica o substituto, verbo exigido antes do lado esquerdo ou None)
# "esq" = substituto à esquerda do conector; "dir" = substituto à direita.
CONECTORES = [
    (["instead", "of"], "esq", None),
    (["in", "place", "of"], "esq", None),
    (["rather", "than"], "esq", None),
    (["for"], "esq", ("substitut", "subbed", "sub")),    # substituted X for Y
    (["for"], "dir", ("swap",)),                          # swapped Y for X
    (["with"], "dir", ("replac", "swap", "substitut", "subbed")),  # replaced Y with X
]


def extrair(review: str) -> list[tuple[str, list[str], list[str]]]:
    """Devolve lista de (lado_original_tokens, ...). Cada item: (tipo, tokens_sub, tokens_cauda_orig_candidata)."""
    # Reviews vêm com entidades HTML (&eacute;, &#039;); sem isso 'purée' vira o token 'e'.
    tokens = TOKEN.findall(html.unescape(review).lower())
    achados = []
    for conector, lado, verbos in CONECTORES:
        n = len(conector)
        for i in range(len(tokens) - n + 1):
            if tokens[i : i + n] != conector:
                continue
            esq, parou = andar(tokens, i - 1, -1)
            dir_, _ = andar(tokens, i + n, +1)
            # Tira artigos/posse no começo do lado direito ("the butter" -> "butter").
            while dir_ and dir_[0] in {"the", "a", "an", "my", "some"}:
                dir_ = dir_[1:]
            if verbos:
                # O lado esquerdo precisa ter parado no verbo certo (substituted/swapped/replaced).
                if lado == "esq" and not (parou and parou.startswith(verbos)):
                    continue
                if lado == "dir" and not (parou and parou.startswith(verbos)):
                    continue
            # Janela de quantidade: "less sugar", "extra butter", "half the oil".
            janela = set(tokens[max(0, i - 6) : i]) | set(tokens[i + n : i + n + 4])
            if janela & QUANTIDADE:
                continue
            if lado == "esq":
                achados.append((esq, dir_))   # (substituto, candidatos a original)
            else:
                achados.append((dir_, esq))   # swapped/replaced: original à esquerda
    return achados


def casar_original(cauda: list[str], ingredientes: set[str], cabecas: set[str]) -> str | None:
    # O texto após o conector pode ter lixo ("butter was great"). Tentamos prefixos, do maior ao menor,
    # e aceitamos o que bate com um ingrediente da receita (nome completo ou última palavra dele).
    for k in range(min(3, len(cauda)), 0, -1):
        cand = normalizar(cauda[:k])
        if cand and (cand in ingredientes or (k == 1 and cand in cabecas)):
            return cand
    return None


def main() -> None:
    receitas = pd.read_csv(RAW / "RAW_recipes.csv", usecols=["id", "ingredients"])
    reviews = pd.read_csv(RAW / "RAW_interactions.csv", usecols=["recipe_id", "review"]).dropna()

    # Só reviews que contêm algum conector (filtro barato antes do regex pesado).
    gatilho = r"instead of|in place of|rather than|substitut|subbed|swapped|replaced"
    reviews = reviews[reviews["review"].str.contains(gatilho, case=False, regex=True)]
    print(f"reviews com gatilho: {len(reviews)}")

    # ingredients vem como string "['a', 'b']" -> conjunto normalizado por receita.
    ingr = {}
    for rid, s in zip(receitas["id"], receitas["ingredients"]):
        nomes = {normalizar(TOKEN.findall(x.lower())) for x in ast.literal_eval(s)}
        ingr[rid] = (nomes, {n.split()[-1] for n in nomes if n})

    linhas = []
    for rid, review in zip(reviews["recipe_id"], reviews["review"]):
        if rid not in ingr:
            continue
        nomes, cabecas = ingr[rid]
        for sub_tok, orig_tok in extrair(review):
            orig = casar_original(orig_tok, nomes, cabecas)
            sub = normalizar(sub_tok)
            if not orig or not sub or sub == orig:
                continue
            if len(sub.split()) > 3 or len(orig.split()) > 3:
                continue
            if set(sub.split()) & LIXO or set(orig.split()) & LIXO:
                continue
            # Substituto NÃO pode estar na receita (nem como última palavra de um ingrediente).
            if sub in nomes or sub in cabecas:
                continue
            linhas.append((rid, orig, sub, review.replace("\n", " ")[:400]))

    pares = pd.DataFrame(linhas, columns=["receita_id", "original", "substituto", "review"])
    pares = pares.drop_duplicates(["receita_id", "original", "substituto"])
    pares.to_csv(SAIDA / "pares.csv", index=False)

    unicos = pares.groupby(["original", "substituto"]).ngroups
    print(f"pares totais: {len(pares)} | pares únicos (original, substituto): {unicos}")
    print("\ntop 20 pares:")
    print(pares.groupby(["original", "substituto"]).size().nlargest(20).to_string())
    print("\ntop 20 originais:")
    print(pares["original"].value_counts().head(20).to_string())

    # Amostra fixa (seed) para conferência manual de precisão.
    pares.sample(min(50, len(pares)), random_state=42).to_csv(SAIDA / "amostra_conferencia.csv", index=False)


if __name__ == "__main__":
    main()
