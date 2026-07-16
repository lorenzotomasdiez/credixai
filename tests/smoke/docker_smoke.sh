#!/usr/bin/env bash
# Smoke test de contenedores (paso 3 de prd.md 9.1: Docker).
#
# Equivalente de TDD para infraestructura: define el contrato de exito
# ANTES de que existan los Dockerfiles, y falla ruidosamente si no se
# cumple. No reemplaza a pytest, es el "red/green" para artefactos que
# pytest no puede ejercitar (build de imagen, arranque de proceso real).
#
# Contrato:
#   1. docker build de Dockerfile.api y Dockerfile.dashboard exitoso.
#   2. Contenedor API responde 200 en /health.
#   3. Contenedor dashboard responde 200 en /_stcore/health (endpoint
#      nativo de Streamlit).
#   4. Ambos montan data/processed como volumen de solo lectura (no se
#      hornea en la imagen: son datos versionados con DVC, no con git).
#
# Uso: bash tests/smoke/docker_smoke.sh

set -euo pipefail

API_IMAGE="credixai-api:smoke"
DASH_IMAGE="credixai-dashboard:smoke"
API_CONTAINER="credixai-api-smoke"
DASH_CONTAINER="credixai-dashboard-smoke"
API_PORT=8001
DASH_PORT=8502

cleanup() {
    docker rm -f "$API_CONTAINER" "$DASH_CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

wait_for() {
    local url="$1"
    local name="$2"
    for _ in $(seq 1 30); do
        if curl -sf "$url" >/dev/null 2>&1; then
            echo "OK: $name respondio en $url"
            return 0
        fi
        sleep 1
    done
    echo "FALLO: $name nunca respondio en $url" >&2
    return 1
}

echo "== build API =="
docker build -f Dockerfile.api -t "$API_IMAGE" .

echo "== build dashboard =="
docker build -f Dockerfile.dashboard -t "$DASH_IMAGE" .

echo "== run API =="
docker run -d --name "$API_CONTAINER" \
    -p "${API_PORT}:8000" \
    -v "$(pwd)/data/processed:/app/data/processed:ro" \
    "$API_IMAGE"
wait_for "http://localhost:${API_PORT}/health" "API"

echo "== run dashboard =="
docker run -d --name "$DASH_CONTAINER" \
    -p "${DASH_PORT}:8501" \
    -v "$(pwd)/data/processed:/app/data/processed:ro" \
    "$DASH_IMAGE"
wait_for "http://localhost:${DASH_PORT}/_stcore/health" "Dashboard"

echo "== smoke test OK =="
