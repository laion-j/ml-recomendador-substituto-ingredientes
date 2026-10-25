"""Baixa do Kaggle os CSVs do Food.com usados na mineração de pares.

Uso: python scripts_ml/01_baixar.py
Requer API_TOKEN_KAGGLE no .env (token gerado em kaggle.com/settings).
"""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DESTINO = RAIZ / "data" / "raw"
DATASET = "shuyangli94/food-com-recipes-and-user-interactions"

# Só estes dois: receitas (ingredientes + passos) e reviews (onde moram as substituições).
# Os PP_*.csv são versões pré-processadas do paper original e não precisamos delas.
ARQUIVOS = ["RAW_recipes.csv", "RAW_interactions.csv"]


def carregar_token() -> None:
    # Lê o .env na mão para não depender de python-dotenv (script de estudo, 1 variável).
    for linha in (RAIZ / ".env").read_text().splitlines():
        chave, _, valor = linha.partition("=")
        if chave.strip() == "API_TOKEN_KAGGLE":
            # O cliente do Kaggle espera o token nesta variável de ambiente.
            os.environ["KAGGLE_API_TOKEN"] = valor.strip()
            return
    raise SystemExit("API_TOKEN_KAGGLE não encontrado no .env")


def main() -> None:
    carregar_token()
    # Importa depois de definir o token: o cliente autentica ao ser usado.
    from kaggle import api

    DESTINO.mkdir(parents=True, exist_ok=True)
    for nome in ARQUIVOS:
        if (DESTINO / nome).exists():
            print(f"ok (já existe): {nome}")
            continue
        print(f"baixando {nome} ...")
        api.dataset_download_file(DATASET, nome, path=str(DESTINO), quiet=False)

    # Os arquivos podem vir zipados (.zip); extrai e remove o zip.
    for zip_ in DESTINO.glob("*.zip"):
        import zipfile

        with zipfile.ZipFile(zip_) as z:
            z.extractall(DESTINO)
        zip_.unlink()
        print(f"extraído: {zip_.name}")


if __name__ == "__main__":
    main()
