from credixai.copilot.evaluator import run_precheck


def test_passes_when_memo_is_well_formed():
    issues = run_precheck(
        memo="El solicitante presenta riesgo aceptable segun el modelo.",
        decision="riesgo_aceptable",
        reason_codes=[],
        citations=["cita 1"],
    )
    assert issues == []


def test_flags_empty_memo():
    issues = run_precheck(memo="", decision="riesgo_aceptable", reason_codes=[], citations=[])
    assert "memo vacio" in issues


def test_flags_more_than_four_reason_codes():
    issues = run_precheck(
        memo="texto",
        decision="alto_riesgo",
        reason_codes=["r1", "r2", "r3", "r4", "r5"],
        citations=["c1"],
    )
    assert "mas de 4 reason codes" in issues


def test_flags_alto_riesgo_without_policy_citation():
    issues = run_precheck(memo="texto", decision="alto_riesgo", reason_codes=["r1"], citations=[])
    assert "decision de alto riesgo sin cita de politica" in issues


def test_flags_protected_attribute_keywords_in_memo():
    issues = run_precheck(
        memo="Se rechaza la solicitud por el genero del solicitante.",
        decision="alto_riesgo",
        reason_codes=["r1"],
        citations=["c1"],
    )
    assert any("atributo protegido" in issue for issue in issues)


def test_multiple_issues_are_all_reported():
    issues = run_precheck(memo="", decision="alto_riesgo", reason_codes=["r1"], citations=[])
    assert "memo vacio" in issues
    assert "decision de alto riesgo sin cita de politica" in issues
