"""Tarea 3: segmenta a los solicitantes en perfiles de riesgo
mediante K-Means y persiste los segmentos en data/processed/segments.parquet.

Requiere haber corrido antes scripts/02_features.py.

Uso:
    uv run python scripts/03_clustering.py
"""

import os

import pandas as pd

from credixai.clustering import build_segments

DATA_DIR = "data/processed"
OUT_DIR = "data/processed"


def main() -> None:
    features = pd.read_parquet(f"{DATA_DIR}/features.parquet")
    segments = build_segments(features)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/segments.parquet"
    segments.to_parquet(out_path, index=False)
    print(f"Guardado en {out_path}: {segments.shape}")


if __name__ == "__main__":
    main()
