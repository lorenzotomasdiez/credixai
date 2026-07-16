"""Grafo LangGraph del copiloto agentico (RF-6, paso 6, prd.md 9.1).

Orchestrator-workers (Anthropic, "Building Effective Agents"): un LLM con
tool-calling real decide dinamicamente que tools invocar
(score_application, explain_shap, retrieve_policy) segun el caso, en vez de
una secuencia fija -- una solicitud riesgo_aceptable puede no necesitar
retrieve_policy. draft_memo + evaluator forman el loop evaluator-optimizer:
un reintento de redaccion con el feedback del evaluator antes de escalar a
revision humana (needs_human_review), nunca reintentos indefinidos.
"""

import json
import operator
from typing import Annotated, Callable, TypedDict

from langgraph.graph import END, StateGraph

from credixai.copilot.evaluator import llm_judge, run_precheck
from credixai.copilot.generation import draft_memo

MAX_RETRIES = 1
MAX_ORCHESTRATOR_ROUNDS = 6

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "score_application",
            "description": (
                "Obtiene la probabilidad de default y la decision "
                "(alto_riesgo/riesgo_aceptable) de una solicitud."
            ),
            "parameters": {
                "type": "object",
                "properties": {"sk_id_curr": {"type": "integer"}},
                "required": ["sk_id_curr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_shap",
            "description": "Obtiene los reason codes (top features SHAP) de una solicitud de alto riesgo.",
            "parameters": {
                "type": "object",
                "properties": {"sk_id_curr": {"type": "integer"}, "top_n": {"type": "integer"}},
                "required": ["sk_id_curr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_policy",
            "description": (
                "Busca politica/normativa relevante (BCRA, Basilea, adverse action) "
                "para citar en el memo."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

_ORCHESTRATOR_SYSTEM_PROMPT = (
    "Sos el orquestador del copiloto de un analista de credito en CrediXAI. Tu trabajo "
    "es reunir la evidencia necesaria para redactar un memo crediticio sobre una "
    "solicitud, llamando a las tools disponibles. Siempre empeza llamando "
    "score_application. Si la decision es alto_riesgo, llama tambien explain_shap "
    "(para los reason codes) y retrieve_policy (para citar la politica de adverse "
    "action correspondiente) antes de terminar. Si la decision es riesgo_aceptable, "
    "podes terminar sin llamar mas tools. Cuando ya tengas toda la evidencia que "
    "necesitas, respondes con texto (sin mas tool calls) para indicar que terminaste."
)


class CopilotState(TypedDict):
    sk_id_curr: int
    messages: Annotated[list[dict], operator.add]
    score: dict | None
    explanation: dict | None
    policy_result: dict | None
    memo: str | None
    evaluator_feedback: str | None
    iteration: int
    orchestrator_rounds: int
    status: str


def initial_state(sk_id_curr: int) -> CopilotState:
    return {
        "sk_id_curr": sk_id_curr,
        "messages": [
            {"role": "system", "content": _ORCHESTRATOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Analiza la solicitud sk_id_curr={sk_id_curr} y redacta un memo crediticio.",
            },
        ],
        "score": None,
        "explanation": None,
        "policy_result": None,
        "memo": None,
        "evaluator_feedback": None,
        "iteration": 0,
        "orchestrator_rounds": 0,
        "status": "in_progress",
    }


def build_graph(tool_fns: dict[str, Callable], chat_with_tools_fn: Callable, chat_fn: Callable):
    """tool_fns: {"score_application": async fn(sk_id_curr), "explain_shap": async fn(sk_id_curr, top_n),
    "retrieve_policy": async fn(query)}. chat_with_tools_fn(messages, tools) -> ChatWithToolsResult,
    usado por el orchestrator. chat_fn(messages) -> str, usado por draft_memo y el juez LLM del evaluator.
    """

    async def orchestrator(state: CopilotState) -> dict:
        result = chat_with_tools_fn(state["messages"], TOOLS_SCHEMA)
        assistant_message = {
            "role": "assistant",
            "content": result.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in result.tool_calls
            ],
        }
        return {"messages": [assistant_message], "orchestrator_rounds": state.get("orchestrator_rounds", 0) + 1}

    def route_after_orchestrator(state: CopilotState) -> str:
        last = state["messages"][-1]
        if state.get("orchestrator_rounds", 0) >= MAX_ORCHESTRATOR_ROUNDS:
            return "draft_memo"
        return "tool_executor" if last.get("tool_calls") else "draft_memo"

    async def tool_executor(state: CopilotState) -> dict:
        last = state["messages"][-1]
        updates: dict = {}
        tool_messages = []
        for call in last["tool_calls"]:
            call_id = call["id"]
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])
            result = await tool_fns[name](**args)
            tool_messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)})
            if name == "score_application":
                updates["score"] = result
            elif name == "explain_shap":
                updates["explanation"] = result
            elif name == "retrieve_policy":
                updates["policy_result"] = result
        updates["messages"] = tool_messages
        return updates

    def draft_memo_node(state: CopilotState) -> dict:
        reason_codes = (state.get("explanation") or {}).get("reason_codes", [])
        policy_result = state.get("policy_result") or {}
        citations = [f"[{c['doc_title']}] {c['snippet']}" for c in policy_result.get("citations", [])]
        memo = draft_memo(
            score=state["score"],
            reason_codes=reason_codes,
            policy_citations=citations,
            chat_fn=chat_fn,
            revision_feedback=state.get("evaluator_feedback") if state.get("iteration", 0) > 0 else None,
        )
        return {"memo": memo}

    def evaluator_node(state: CopilotState) -> dict:
        score = state["score"]
        reason_codes = (state.get("explanation") or {}).get("reason_codes", [])
        policy_result = state.get("policy_result") or {}
        citations = [c["snippet"] for c in policy_result.get("citations", [])]

        issues = run_precheck(state["memo"], score["decision"], reason_codes, citations)
        if issues:
            approved, feedback = False, "; ".join(issues)
        else:
            context = f"score={score}, reason_codes={reason_codes}, citas={citations}"
            approved, feedback = llm_judge(state["memo"], context, chat_fn)

        if approved:
            return {"status": "approved", "evaluator_feedback": feedback}
        if state.get("iteration", 0) < MAX_RETRIES:
            return {
                "status": "in_progress",
                "evaluator_feedback": feedback,
                "iteration": state.get("iteration", 0) + 1,
            }
        return {"status": "needs_human_review", "evaluator_feedback": feedback}

    def route_after_evaluator(state: CopilotState) -> str:
        return "draft_memo" if state["status"] == "in_progress" else END

    graph = StateGraph(CopilotState)
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("draft_memo", draft_memo_node)
    graph.add_node("evaluator", evaluator_node)

    graph.set_entry_point("orchestrator")
    graph.add_conditional_edges(
        "orchestrator", route_after_orchestrator, {"tool_executor": "tool_executor", "draft_memo": "draft_memo"}
    )
    graph.add_edge("tool_executor", "orchestrator")
    graph.add_edge("draft_memo", "evaluator")
    graph.add_conditional_edges("evaluator", route_after_evaluator, {"draft_memo": "draft_memo", END: END})

    return graph.compile()
