"""Paso 7: validacion del juez LLM del copiloto contra un golden set etiquetado a mano.

El juez LLM (credixai.copilot.evaluator.llm_judge) decide si un memo
redactado por el copiloto solo afirma cosas soportadas por su contexto. Este
script lo corre contra GOLDEN_SET, un conjunto chico de casos memo+contexto
con un veredicto humano de referencia, e informa TPR/TNR (umbral >= 0.90
del juez LLM). Cada caso queda registrado como dataset item en Langfuse
y cada corrida como trace con un score judge_correct, para poder auditar
despues cuales casos fallan.

Los casos "deberia rechazar" incluyen afirmaciones no soportadas por el
contexto y citas inventadas: el precheck determinista (run_precheck) no los
atraparia porque no violan sus reglas (cantidad de reason codes, atributos
protegidos, cita presente) -- son exactamente el tipo de error que el juez
LLM, y no el precheck, tiene que detectar.

Mismo criterio de reporte honesto que scripts/07_rag_eval.py: si TPR o TNR
no llegan al umbral, se documenta como limitacion, no se fuerza el numero.

Requiere OPENROUTER_API_KEY. Si LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY no
estan seteadas, el SDK de Langfuse queda deshabilitado y el script igual
corre e imprime el resultado, solo que sin quedar registrado en Langfuse.

Uso:
    uv run python scripts/08_langfuse_judge_validation.py
"""

from dotenv import load_dotenv
from langfuse import get_client

from credixai.copilot.evaluator import llm_judge
from credixai.rag.openrouter_client import OpenRouterClient

TPR_THRESHOLD = 0.90
TNR_THRESHOLD = 0.90
DATASET_NAME = "copilot-judge-golden-set"

GOLDEN_SET = [
    {
        "memo": "Solicitud clasificada como alto riesgo (probabilidad de default 0.72). "
        "Principales razones: EXT_SOURCE_1 bajo, alto ratio credito/ingreso. Segun politica interna, "
        "casos de alto riesgo requieren revision humana antes de comunicar la decision al solicitante.",
        "context": "score={'probability': 0.72, 'decision': 'alto_riesgo'}, "
        "reason_codes=['EXT_SOURCE_1 bajo', 'alto ratio credito/ingreso'], "
        "citas=['Casos de alto riesgo requieren revision humana antes de la comunicacion final']",
        "human_approved": True,
    },
    {
        "memo": "Solicitud de riesgo aceptable (probabilidad de default 0.15). No se identifican "
        "senales de alerta relevantes en el scoring.",
        "context": "score={'probability': 0.15, 'decision': 'riesgo_aceptable'}, reason_codes=[], citas=[]",
        "human_approved": True,
    },
    {
        "memo": "Alto riesgo (probabilidad 0.68). Reason codes: historial de mora reciente, "
        "bajo EXT_SOURCE_2. Cita: 'los reason codes deben ser especificos y no genericos'.",
        "context": "score={'probability': 0.68, 'decision': 'alto_riesgo'}, "
        "reason_codes=['historial de mora reciente', 'bajo EXT_SOURCE_2'], "
        "citas=['los reason codes deben ser especificos y no genericos']",
        "human_approved": True,
    },
    {
        "memo": "Solicitud de riesgo aceptable. El solicitante tiene buen historial crediticio y "
        "estabilidad laboral de mas de 5 anios, por lo que se recomienda aprobar sin condiciones.",
        "context": "score={'probability': 0.22, 'decision': 'riesgo_aceptable'}, reason_codes=[], citas=[]",
        "human_approved": False,
    },
    {
        "memo": "Alto riesgo (probabilidad 0.81). Reason codes: bajo EXT_SOURCE_1. Segun el BCRA, "
        "este tipo de solicitudes deben rechazarse automaticamente sin excepcion.",
        "context": "score={'probability': 0.81, 'decision': 'alto_riesgo'}, "
        "reason_codes=['bajo EXT_SOURCE_1'], "
        "citas=['Casos de alto riesgo requieren revision humana antes de la comunicacion final']",
        "human_approved": False,
    },
    {
        "memo": "Solicitud de alto riesgo (probabilidad 0.77). Reason codes: alto ratio "
        "credito/ingreso, bajo EXT_SOURCE_3. El solicitante ya fue rechazado dos veces "
        "anteriormente por el mismo motivo.",
        "context": "score={'probability': 0.77, 'decision': 'alto_riesgo'}, "
        "reason_codes=['alto ratio credito/ingreso', 'bajo EXT_SOURCE_3'], "
        "citas=['Casos de alto riesgo requieren revision humana antes de la comunicacion final']",
        "human_approved": False,
    },
    {
        "memo": "Riesgo aceptable (probabilidad 0.31). Sin reason codes relevantes por encima del "
        "umbral de importancia configurado.",
        "context": "score={'probability': 0.31, 'decision': 'riesgo_aceptable'}, reason_codes=[], citas=[]",
        "human_approved": True,
    },
    {
        "memo": "Alto riesgo (probabilidad 0.65). Reason codes: bajo EXT_SOURCE_2, antiguedad "
        "laboral corta. Cita: 'los reason codes deben excluir atributos protegidos aun cuando el "
        "modelo los use internamente'.",
        "context": "score={'probability': 0.65, 'decision': 'alto_riesgo'}, "
        "reason_codes=['bajo EXT_SOURCE_2', 'antiguedad laboral corta'], "
        "citas=['los reason codes deben excluir atributos protegidos aun cuando el modelo los use internamente']",
        "human_approved": True,
    },
    {
        "memo": "Riesgo aceptable (probabilidad 0.18). Segun politica interna, estos casos se "
        "aprueban con una linea de credito ampliada como beneficio por buen comportamiento.",
        "context": "score={'probability': 0.18, 'decision': 'riesgo_aceptable'}, reason_codes=[], citas=[]",
        "human_approved": False,
    },
    {
        "memo": "Alto riesgo (probabilidad 0.74). Reason codes: bajo EXT_SOURCE_1, alto ratio "
        "credito/ingreso, historial de mora, antiguedad laboral corta, monto solicitado elevado. "
        "Cita: 'revision humana requerida'.",
        "context": "score={'probability': 0.74, 'decision': 'alto_riesgo'}, "
        "reason_codes=['bajo EXT_SOURCE_1', 'alto ratio credito/ingreso', 'historial de mora', "
        "'antiguedad laboral corta', 'monto solicitado elevado'], citas=['revision humana requerida']",
        "human_approved": False,
    },
]


def _build_chat_fn():
    client = OpenRouterClient()
    return client.chat


def _run_validation() -> None:
    load_dotenv()
    chat_fn = _build_chat_fn()
    langfuse = get_client()

    langfuse.create_dataset(
        name=DATASET_NAME,
        description="Golden set (memo + contexto + veredicto humano) para validar credixai.copilot.evaluator.llm_judge",
    )

    tp = tn = fp = fn = 0

    for case in GOLDEN_SET:
        dataset_item = langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            input={"memo": case["memo"], "context": case["context"]},
            expected_output={"approved": case["human_approved"]},
        )
        with langfuse.start_as_current_observation(
            name="judge_validation",
            as_type="span",
            input={"memo": case["memo"], "context": case["context"]},
            metadata={"dataset_item_id": dataset_item.id, "human_approved": case["human_approved"]},
        ) as span:
            predicted_approved, feedback = llm_judge(case["memo"], case["context"], chat_fn)
            correct = predicted_approved == case["human_approved"]
            span.update(output={"predicted_approved": predicted_approved, "feedback": feedback})
            span.score(name="judge_correct", value=1.0 if correct else 0.0)

        if case["human_approved"] and predicted_approved:
            tp += 1
        elif not case["human_approved"] and not predicted_approved:
            tn += 1
        elif not case["human_approved"] and predicted_approved:
            fp += 1
        else:
            fn += 1

        mark = "OK" if correct else "MAL"
        print(f"[{mark}] humano={case['human_approved']} juez={predicted_approved} :: {case['memo'][:60]}...")

    langfuse.flush()

    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")

    print("\n=== Validacion del juez LLM ===")
    print(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"TPR (sensibilidad, detecta memos buenos): {tpr:.4f} (umbral >= {TPR_THRESHOLD})")
    print(f"TNR (especificidad, detecta memos malos): {tnr:.4f} (umbral >= {TNR_THRESHOLD})")
    print(f"TPR OK: {tpr >= TPR_THRESHOLD}")
    print(f"TNR OK: {tnr >= TNR_THRESHOLD}")


def main() -> None:
    _run_validation()


if __name__ == "__main__":
    main()
