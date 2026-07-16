"""Lanza el dashboard de Streamlit (app/dashboard.py) forzando el thread pool
interno de PyArrow a 1 hilo antes de que se importe pandas.

Bug real encontrado y corregido (paso 9, prd.md 9.1): `streamlit run
app/dashboard.py` directo segfaultea de forma reproducible a partir de la
segunda vez que Streamlit re-ejecuta el script (cualquier interaccion con
un widget dispara una re-ejecucion completa), en `st.dataframe()` o en un
`.map()` sobre una columna de texto -- ambos pasan por
`pyarrow.pandas_compat`. Causa raiz: el CLI de Streamlit (`streamlit.web.cli`)
importa pandas como parte de su propio arranque, antes de ejecutar el
script del usuario, y pandas 3.0 inicializa el thread pool de PyArrow con
el default (multi-core) en ese momento. Llamar a `pyarrow.set_cpu_count(1)`
desde dentro de app/dashboard.py llega demasiado tarde: el pool ya existe.
Reproducido y verificado con `streamlit.testing.v1.AppTest` (headless, sin
navegador): 5 reruns consecutivos y el flujo completo de las pestanias RAG
y Copiloto sobreviven solo si `set_cpu_count(1)` corre antes de cualquier
import de streamlit/pandas, es decir, antes de que exista otra forma de
inyectarlo que no sea este launcher.

Uso:
    uv run python app/dashboard_launcher.py
"""

import sys

import pyarrow

pyarrow.set_cpu_count(1)
pyarrow.set_io_thread_count(1)

from streamlit.web import cli as stcli  # noqa: E402

if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        "app/dashboard.py",
        *sys.argv[1:],
    ]
    sys.exit(stcli.main())
