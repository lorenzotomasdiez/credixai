"""Tests para src/credixai/features.py (Tarea 2: feature engineering)."""

import numpy as np
import pandas as pd

from credixai.features import (
    add_business_ratios,
    add_no_record_flags,
    aggregate_bureau,
    aggregate_bureau_balance,
    build_feature_table,
    clean_days_employed,
    encode_categoricals,
)


def test_clean_days_employed_replaces_sentinel_and_flags_it():
    app = pd.DataFrame({"DAYS_EMPLOYED": [365243, -500, 365243, -100]})

    out = clean_days_employed(app)

    assert out["DAYS_EMPLOYED_ANOM"].tolist() == [True, False, True, False]
    assert out["DAYS_EMPLOYED"].isna().tolist() == [True, False, True, False]
    assert out.loc[1, "DAYS_EMPLOYED"] == -500


def test_add_business_ratios_computes_expected_values():
    app = pd.DataFrame({
        "AMT_CREDIT": [100_000.0, 50_000.0],
        "AMT_INCOME_TOTAL": [50_000.0, 0.0],
        "AMT_ANNUITY": [10_000.0, 5_000.0],
        "AMT_GOODS_PRICE": [80_000.0, 40_000.0],
    })

    out = add_business_ratios(app)

    assert out.loc[0, "credit_to_income"] == 2.0
    assert out.loc[0, "annuity_to_income"] == 0.2
    assert out.loc[0, "credit_to_goods"] == 1.25
    # division by zero income must produce NaN, not inf/crash
    assert np.isnan(out.loc[1, "credit_to_income"])
    assert np.isnan(out.loc[1, "annuity_to_income"])


def test_add_no_record_flags_true_when_reference_column_is_nan():
    features = pd.DataFrame({
        "bureau_count": [3, np.nan],
        "bb_credits_with_history": [np.nan, 2],
        "prev_count": [1, np.nan],
        "pos_credits_count": [np.nan, 1],
        "cc_cards_count": [1, np.nan],
        "inst_delay_mean": [0.5, np.nan],
    })

    out = add_no_record_flags(features)

    assert out["bureau_no_record"].tolist() == [False, True]
    assert out["bb_no_record"].tolist() == [True, False]
    assert out["inst_no_record"].tolist() == [False, True]


def test_encode_categoricals_onehot_and_prev_ever_refused_mapping():
    features = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3],
        "NAME_FAMILY_STATUS": ["Married", "Single", None],
        "prev_ever_refused": [True, False, True],
    })

    out = encode_categoricals(features)

    assert "prev_ever_refused" in out.columns
    assert out["prev_ever_refused"].tolist() == [1, 0, 1]
    assert "NAME_FAMILY_STATUS_Married" in out.columns
    assert "NAME_FAMILY_STATUS_Single" in out.columns
    # dummy_na=True must produce an explicit column for the missing category
    assert any(c.startswith("NAME_FAMILY_STATUS_") and "nan" in c.lower() for c in out.columns)


def test_aggregate_bureau_groups_by_sk_id_curr(tmp_path):
    pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "SK_ID_BUREAU": [10, 11, 12],
        "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
        "AMT_CREDIT_SUM": [1000.0, 2000.0, 500.0],
        "AMT_CREDIT_SUM_DEBT": [100.0, 0.0, 50.0],
        "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 10.0],
        "AMT_ANNUITY": [50.0, 60.0, 20.0],
    }).to_csv(tmp_path / "bureau.csv", index=False)

    out = aggregate_bureau(str(tmp_path))

    assert out.loc[1, "bureau_count"] == 2
    assert out.loc[1, "bureau_active_count"] == 1
    assert out.loc[2, "bureau_count"] == 1
    assert out.loc[1, "bureau_credit_sum_mean"] == 1500.0


def test_aggregate_bureau_balance_maps_dpd_status_and_joins_two_hops(tmp_path):
    bureau_ids = pd.DataFrame({"SK_ID_CURR": [1, 1], "SK_ID_BUREAU": [10, 11]})
    pd.DataFrame({
        "SK_ID_BUREAU": [10, 10, 11],
        "MONTHS_BALANCE": [0, -1, 0],
        "STATUS": ["0", "2", "C"],
    }).to_csv(tmp_path / "bureau_balance.csv", index=False)

    out = aggregate_bureau_balance(str(tmp_path), bureau_ids)

    assert out.loc[1, "bb_credits_with_history"] == 2
    # only credit 10 has a month with dpd_level > 0 (status "2")
    assert out.loc[1, "bb_months_dpd_sum"] == 1
    assert out.loc[1, "bb_worst_dpd_level_max"] == 2


def _write_minimal_home_credit_tables(data_dir):
    """Escribe versiones minimas (2 solicitantes) de las 8 tablas para probar
    build_feature_table de punta a punta sin depender del dataset real.
    """
    pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "TARGET": [0, 1],
        "AMT_CREDIT": [100_000.0, 50_000.0],
        "AMT_INCOME_TOTAL": [50_000.0, 40_000.0],
        "AMT_ANNUITY": [10_000.0, 5_000.0],
        "AMT_GOODS_PRICE": [80_000.0, 40_000.0],
        "DAYS_EMPLOYED": [-500, 365243],
        "DAYS_BIRTH": [-12000, -9000],
        "NAME_FAMILY_STATUS": ["Married", "Single"],
    }).to_csv(data_dir / "application_train.csv", index=False)

    pd.DataFrame({
        "SK_ID_CURR": [3],
        "AMT_CREDIT": [70_000.0],
        "AMT_INCOME_TOTAL": [60_000.0],
        "AMT_ANNUITY": [7_000.0],
        "AMT_GOODS_PRICE": [65_000.0],
        "DAYS_EMPLOYED": [-200],
        "DAYS_BIRTH": [-11000],
        "NAME_FAMILY_STATUS": ["Married"],
    }).to_csv(data_dir / "application_test.csv", index=False)

    pd.DataFrame({
        "SK_ID_CURR": [1],
        "SK_ID_BUREAU": [10],
        "CREDIT_ACTIVE": ["Active"],
        "AMT_CREDIT_SUM": [1000.0],
        "AMT_CREDIT_SUM_DEBT": [100.0],
        "AMT_CREDIT_SUM_OVERDUE": [0.0],
        "AMT_ANNUITY": [50.0],
    }).to_csv(data_dir / "bureau.csv", index=False)

    pd.DataFrame({
        "SK_ID_BUREAU": [10],
        "MONTHS_BALANCE": [0],
        "STATUS": ["0"],
    }).to_csv(data_dir / "bureau_balance.csv", index=False)

    pd.DataFrame({
        "SK_ID_CURR": [1],
        "SK_ID_PREV": [100],
        "NAME_CONTRACT_STATUS": ["Approved"],
        "AMT_CREDIT": [90_000.0],
        "AMT_ANNUITY": [9_000.0],
        "DAYS_DECISION": [-30],
    }).to_csv(data_dir / "previous_application.csv", index=False)

    pd.DataFrame({
        "SK_ID_CURR": [1],
        "SK_ID_PREV": [100],
        "SK_DPD": [0],
    }).to_csv(data_dir / "POS_CASH_balance.csv", index=False)

    pd.DataFrame({
        "SK_ID_CURR": [1],
        "SK_ID_PREV": [100],
        "AMT_BALANCE": [500.0],
        "AMT_CREDIT_LIMIT_ACTUAL": [1000.0],
        "SK_DPD": [0],
    }).to_csv(data_dir / "credit_card_balance.csv", index=False)

    pd.DataFrame({
        "SK_ID_CURR": [1],
        "DAYS_ENTRY_PAYMENT": [-5],
        "DAYS_INSTALMENT": [-10],
        "AMT_INSTALMENT": [1000.0],
        "AMT_PAYMENT": [1000.0],
    }).to_csv(data_dir / "installments_payments.csv", index=False)


def test_build_feature_table_end_to_end(tmp_path):
    _write_minimal_home_credit_tables(tmp_path)

    features = build_feature_table(str(tmp_path))

    assert features["SK_ID_CURR"].is_unique
    assert set(features["SK_ID_CURR"]) == {1, 2, 3}
    # IS_TRAIN derived from TARGET presence: rows 1,2 are train, row 3 is test
    assert features.set_index("SK_ID_CURR")["IS_TRAIN"].to_dict() == {1: 1, 2: 1, 3: 0}
    # applicant 1 has bureau history, applicant 2 does not
    assert not features.set_index("SK_ID_CURR").loc[1, "bureau_no_record"]
    assert features.set_index("SK_ID_CURR").loc[2, "bureau_no_record"]
    # DAYS_EMPLOYED sentinel for applicant 2 must be cleaned to NaN
    assert features.set_index("SK_ID_CURR").loc[2, "DAYS_EMPLOYED_ANOM"]
