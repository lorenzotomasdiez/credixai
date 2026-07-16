"""Descarga el dataset Home Credit Default Risk (Kaggle) a data/raw/.

Requiere credenciales de la API de Kaggle (no se pueden commitear ni
generar automaticamente, hay que tenerlas antes de correr este script):
    1. Aceptar las reglas de la competencia en
       https://www.kaggle.com/c/home-credit-default-risk/rules
    2. Generar un token en https://www.kaggle.com/settings -> API -> Create New Token,
       que descarga kaggle.json.
    3. Colocar ese archivo en ~/.kaggle/kaggle.json (chmod 600), o exportar
       KAGGLE_USERNAME/KAGGLE_KEY como variables de entorno.

Si data/raw ya tiene los 9 CSV esperados, no vuelve a descargar (idempotente).

Uso:
    uv run python scripts/00_download_data.py
"""

import os
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "home-credit-default-risk"
OUT_DIR = "data/raw"
EXPECTED_FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
    "previous_application.csv",
    "installments_payments.csv",
]


def already_downloaded() -> bool:
    return all(os.path.exists(f"{OUT_DIR}/{name}") for name in EXPECTED_FILES)


def main() -> None:
    if already_downloaded():
        print(f"{OUT_DIR} ya tiene los {len(EXPECTED_FILES)} CSV esperados, no se descarga de nuevo.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Descargando competencia '{COMPETITION}' a {OUT_DIR}...")
    api.competition_download_files(COMPETITION, path=OUT_DIR, quiet=False)

    zip_path = f"{OUT_DIR}/{COMPETITION}.zip"
    print(f"Descomprimiendo {zip_path}...")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(OUT_DIR)
    os.remove(zip_path)

    # El zip de la competencia trae algunos CSV como .zip individuales.
    for entry in list(os.scandir(OUT_DIR)):
        if entry.name.endswith(".zip"):
            with zipfile.ZipFile(entry.path) as inner:
                inner.extractall(OUT_DIR)
            os.remove(entry.path)

    missing = [name for name in EXPECTED_FILES if not os.path.exists(f"{OUT_DIR}/{name}")]
    if missing:
        raise RuntimeError(f"Descarga incompleta, faltan: {missing}")

    print(f"Listo: {len(EXPECTED_FILES)} CSV en {OUT_DIR}.")


if __name__ == "__main__":
    main()
