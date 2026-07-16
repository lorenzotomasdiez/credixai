"""Serving (paso 2 de prd.md 9.1): logica reutilizable detras de app/api.py.

ScoringService envuelve un bundle ya entrenado (el mismo que produce
credixai.dashboard.train_full_model) y responde consultas de score/explain
por SK_ID_CURR, reutilizando sin duplicar compute_shap y reason_codes de
credixai.explainability.
"""

from dataclasses import dataclass, field

import shap

from credixai.explainability import reason_codes as _reason_codes


@dataclass
class ScoreResult:
    sk_id_curr: int
    probability: float
    threshold: float
    decision: str


@dataclass
class ExplainResult:
    sk_id_curr: int
    probability: float
    threshold: float
    decision: str
    base_value: float
    shap_values: list
    feature_names: list
    reason_codes: list = field(default_factory=list)


def _decision(probability: float, threshold: float) -> str:
    return "alto_riesgo" if probability >= threshold else "riesgo_aceptable"


class ScoringService:
    """Responde score/explain por SK_ID_CURR sobre un bundle ya entrenado.

    El bundle tiene la misma forma que credixai.dashboard.train_full_model():
    model, X, train_full (con columna SK_ID_CURR), proba_full, threshold.
    """

    def __init__(self, bundle: dict):
        self.bundle = bundle
        self._position_by_sk_id = {
            int(sk_id): pos for pos, sk_id in enumerate(bundle["train_full"]["SK_ID_CURR"])
        }
        self._explainer = shap.TreeExplainer(bundle["model"])

    def _position(self, sk_id_curr: int) -> int:
        if sk_id_curr not in self._position_by_sk_id:
            raise KeyError(f"SK_ID_CURR {sk_id_curr} no encontrado")
        return self._position_by_sk_id[sk_id_curr]

    def score(self, sk_id_curr: int) -> ScoreResult:
        pos = self._position(sk_id_curr)
        probability = float(self.bundle["proba_full"][pos])
        threshold = float(self.bundle["threshold"])
        return ScoreResult(
            sk_id_curr=sk_id_curr,
            probability=probability,
            threshold=threshold,
            decision=_decision(probability, threshold),
        )

    def explain(self, sk_id_curr: int, top_n: int = 4) -> ExplainResult:
        pos = self._position(sk_id_curr)
        threshold = float(self.bundle["threshold"])
        probability = float(self.bundle["proba_full"][pos])
        decision = _decision(probability, threshold)

        row = self.bundle["X"].iloc[[pos]]
        shap_row = self._explainer.shap_values(row)[0]
        feature_names = list(self.bundle["X"].columns)

        reasons = _reason_codes(shap_row, feature_names, top_n=top_n) if decision == "alto_riesgo" else []

        return ExplainResult(
            sk_id_curr=sk_id_curr,
            probability=probability,
            threshold=threshold,
            decision=decision,
            base_value=float(self._explainer.expected_value),
            shap_values=[float(v) for v in shap_row],
            feature_names=feature_names,
            reason_codes=reasons,
        )
