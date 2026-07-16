"""API REST (RF-8, paso 2 de prd.md 9.1) para CrediXAI.

Entrypoint delgado sobre credixai.api.ScoringService, mismo patron de
separacion logica/entrypoint que app/dashboard.py. Expone /score y
/explain sobre el modelo final de la Tarea 4, reutilizando SHAP y reason
codes de la Tarea 5 sin reimplementar nada.

Requiere haber corrido antes scripts/02_features.py.

Uso:
    uv run fastapi run app/api.py
"""

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from credixai.api import ScoringService
from credixai.dashboard import load_features, train_full_model

app = FastAPI(
    title="CrediXAI API",
    description="Scoring crediticio explicable: probabilidad de default y explicacion SHAP por solicitud.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_service() -> ScoringService:
    features = load_features()
    bundle = train_full_model(features)
    return ScoringService(bundle)


class ScoreResponse(BaseModel):
    sk_id_curr: int
    probability: float
    threshold: float
    decision: str


class ExplainResponse(BaseModel):
    sk_id_curr: int
    probability: float
    threshold: float
    decision: str
    base_value: float
    shap_values: list[float]
    feature_names: list[str]
    reason_codes: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/score/{sk_id_curr}", response_model=ScoreResponse)
def score(sk_id_curr: int, service: ScoringService = Depends(get_service)) -> ScoreResponse:
    try:
        result = service.score(sk_id_curr)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"SK_ID_CURR {sk_id_curr} no encontrado")
    return ScoreResponse(**result.__dict__)


@app.get("/explain/{sk_id_curr}", response_model=ExplainResponse)
def explain(
    sk_id_curr: int, top_n: int = 4, service: ScoringService = Depends(get_service)
) -> ExplainResponse:
    try:
        result = service.explain(sk_id_curr, top_n=top_n)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"SK_ID_CURR {sk_id_curr} no encontrado")
    return ExplainResponse(**result.__dict__)
