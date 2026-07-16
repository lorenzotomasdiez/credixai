"""Tools del copiloto agentico (RF-6, paso 6).

Cada tool llama al endpoint REST correspondiente (/score, /explain,
/rag/query) via httpx en vez de importar ScoringService/RagPipeline
directamente (decision de diseno: tools expuestas como endpoints FastAPI, no imports directos).
Son async porque httpx.ASGITransport (llamar la propia app FastAPI sin un
segundo proceso escuchando en red real) solo soporta clientes async; el
grafo LangGraph que las invoca tambien es async de punta a punta.
"""

import httpx


async def score_application(http_client: httpx.AsyncClient, sk_id_curr: int) -> dict:
    response = await http_client.get(f"/score/{sk_id_curr}")
    response.raise_for_status()
    return response.json()


async def explain_shap(http_client: httpx.AsyncClient, sk_id_curr: int, top_n: int = 4) -> dict:
    response = await http_client.get(f"/explain/{sk_id_curr}", params={"top_n": top_n})
    response.raise_for_status()
    return response.json()


async def retrieve_policy(http_client: httpx.AsyncClient, query: str) -> dict:
    response = await http_client.post("/rag/query", json={"question": query})
    response.raise_for_status()
    return response.json()
