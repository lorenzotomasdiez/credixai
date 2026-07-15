"""Tarea 2 (prd.md S5): construye la tabla de features de Home Credit Default Risk
y la persiste en data/processed/features.parquet.

Uso:
    uv run python scripts/02_features.py
"""

import os

from credixai.features import build_feature_table

DATA_DIR = "data/raw"
OUT_DIR = "data/processed"


def main() -> None:
    features = build_feature_table(DATA_DIR)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/features.parquet"
    features.to_parquet(out_path, index=False)
    print(f"Guardado en {out_path}: {features.shape}")


if __name__ == "__main__":
    main()
