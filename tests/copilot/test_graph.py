"""Tests para credixai.copilot.graph (paso 6).

TDD: se testea el grafo LangGraph completo con tool_fns y chat_fns
stubeados/scriptados (sin red, sin LangChain), cubriendo las tres ramas del
loop evaluator-optimizer: aprobado directo, rechazado+reintento+aprobado, y
rechazado dos veces -> needs_human_review.
"""

import asyncio

from credixai.copilot.graph import build_graph, initial_state
from credixai.rag.openrouter_client import ChatWithToolsResult, ToolCall

SK_ID_CURR = 100002


async def _fake_score_application(sk_id_curr):
    return {"sk_id_curr": sk_id_curr, "probability": 0.8, "threshold": 0.5, "decision": "alto_riesgo"}


async def _fake_explain_shap(sk_id_curr, top_n=4):
    return {"reason_codes": ["EXT_SOURCE_1 bajo", "AMT_CREDIT alto"]}


async def _fake_retrieve_policy(query):
    return {
        "answer": "maximo 4 reason codes",
        "citations": [{"doc_title": "Politica interna", "chunk_id": "doc::0", "snippet": "maximo 4 reason codes"}],
    }


def _tool_fns():
    return {
        "score_application": _fake_score_application,
        "explain_shap": _fake_explain_shap,
        "retrieve_policy": _fake_retrieve_policy,
    }


class _ScriptedOrchestrator:
    """Simula tool-calling real: score -> explain -> retrieve_policy -> listo."""

    def __init__(self):
        self._round = 0

    def __call__(self, messages, tools):
        self._round += 1
        if self._round == 1:
            return ChatWithToolsResult(
                content=None, tool_calls=[ToolCall(id="c1", name="score_application", arguments={"sk_id_curr": SK_ID_CURR})]
            )
        if self._round == 2:
            return ChatWithToolsResult(
                content=None,
                tool_calls=[ToolCall(id="c2", name="explain_shap", arguments={"sk_id_curr": SK_ID_CURR, "top_n": 4})],
            )
        if self._round == 3:
            return ChatWithToolsResult(
                content=None,
                tool_calls=[ToolCall(id="c3", name="retrieve_policy", arguments={"query": "adverse action reason codes"})],
            )
        return ChatWithToolsResult(content="listo", tool_calls=[])


def _is_judge_prompt(messages) -> bool:
    return "auditor de memos" in messages[0]["content"]


def test_graph_approves_memo_on_first_pass():
    def chat_fn(messages):
        if _is_judge_prompt(messages):
            return '{"approved": true, "feedback": "memo consistente"}'
        return "Memo: solicitud de alto riesgo por EXT_SOURCE_1 bajo y AMT_CREDIT alto. Ver Politica interna."

    graph = build_graph(_tool_fns(), _ScriptedOrchestrator(), chat_fn)
    result = asyncio.run(graph.ainvoke(initial_state(SK_ID_CURR)))

    assert result["status"] == "approved"
    assert result["score"]["decision"] == "alto_riesgo"
    assert result["explanation"]["reason_codes"] == ["EXT_SOURCE_1 bajo", "AMT_CREDIT alto"]
    assert result["iteration"] == 0
    assert "Politica interna" in result["memo"]


def test_graph_retries_once_then_approves():
    judge_calls = {"n": 0}

    def chat_fn(messages):
        if _is_judge_prompt(messages):
            judge_calls["n"] += 1
            if judge_calls["n"] == 1:
                return '{"approved": false, "feedback": "menciona la edad del solicitante"}'
            return '{"approved": true, "feedback": "corregido"}'
        return "Memo: alto riesgo por EXT_SOURCE_1 bajo y AMT_CREDIT alto. Ver Politica interna."

    graph = build_graph(_tool_fns(), _ScriptedOrchestrator(), chat_fn)
    result = asyncio.run(graph.ainvoke(initial_state(SK_ID_CURR)))

    assert result["status"] == "approved"
    assert result["iteration"] == 1
    assert judge_calls["n"] == 2


def test_graph_escalates_to_human_review_after_max_retries():
    def chat_fn(messages):
        if _is_judge_prompt(messages):
            return '{"approved": false, "feedback": "sigue mencionando atributos protegidos"}'
        return "Memo: alto riesgo. Ver Politica interna."

    graph = build_graph(_tool_fns(), _ScriptedOrchestrator(), chat_fn)
    result = asyncio.run(graph.ainvoke(initial_state(SK_ID_CURR)))

    assert result["status"] == "needs_human_review"
    assert result["iteration"] == 1
    assert "atributos protegidos" in result["evaluator_feedback"]


def test_graph_skips_explain_and_retrieve_policy_when_orchestrator_deems_them_unnecessary():
    def chat_fn(messages):
        if _is_judge_prompt(messages):
            return '{"approved": true, "feedback": "ok"}'
        return "Memo: riesgo aceptable, sin reason codes."

    class _RiesgoAceptableOrchestrator:
        def __init__(self):
            self._round = 0

        def __call__(self, messages, tools):
            self._round += 1
            if self._round == 1:
                return ChatWithToolsResult(
                    content=None,
                    tool_calls=[ToolCall(id="c1", name="score_application", arguments={"sk_id_curr": SK_ID_CURR})],
                )
            return ChatWithToolsResult(content="listo", tool_calls=[])

    async def fake_score_aceptable(sk_id_curr):
        return {"sk_id_curr": sk_id_curr, "probability": 0.1, "threshold": 0.5, "decision": "riesgo_aceptable"}

    tool_fns = _tool_fns()
    tool_fns["score_application"] = fake_score_aceptable

    graph = build_graph(tool_fns, _RiesgoAceptableOrchestrator(), chat_fn)
    result = asyncio.run(graph.ainvoke(initial_state(SK_ID_CURR)))

    assert result["status"] == "approved"
    assert result["explanation"] is None
    assert result["policy_result"] is None
