"""Tests para credixai.monitoring.logging (paso 8, prd.md 9.1).

TDD: se escribe antes que credixai/monitoring/logging.py.
format_log_entry es pura (recibe el timestamp en vez de generarlo) para
poder testearla sin depender del reloj real; el efecto de escribir a disco
se testea aparte, contra /score en app/api.py.
"""

import json

from credixai.api import ScoreResult
from credixai.monitoring.logging import append_log_entry, format_log_entry


def test_format_log_entry_includes_sk_id_curr_probability_threshold_and_decision():
    result = ScoreResult(sk_id_curr=100002, probability=0.72, threshold=0.5, decision="alto_riesgo")

    entry = format_log_entry(result, timestamp="2026-07-16T10:00:00Z")

    assert entry == {
        "timestamp": "2026-07-16T10:00:00Z",
        "sk_id_curr": 100002,
        "probability": 0.72,
        "threshold": 0.5,
        "decision": "alto_riesgo",
    }


def test_format_log_entry_is_json_serializable():
    result = ScoreResult(sk_id_curr=100003, probability=0.15, threshold=0.5, decision="riesgo_aceptable")

    entry = format_log_entry(result, timestamp="2026-07-16T10:00:01Z")

    assert json.loads(json.dumps(entry)) == entry


def test_append_log_entry_creates_file_and_parent_dir(tmp_path):
    log_path = tmp_path / "monitoring" / "prediction_log.jsonl"
    entry = {"timestamp": "2026-07-16T10:00:00Z", "sk_id_curr": 100002}

    append_log_entry(entry, str(log_path))

    assert log_path.exists()
    assert json.loads(log_path.read_text().strip()) == entry


def test_append_log_entry_appends_without_overwriting(tmp_path):
    log_path = tmp_path / "prediction_log.jsonl"
    first = {"sk_id_curr": 1}
    second = {"sk_id_curr": 2}

    append_log_entry(first, str(log_path))
    append_log_entry(second, str(log_path))

    lines = log_path.read_text().strip().splitlines()
    assert [json.loads(line) for line in lines] == [first, second]
