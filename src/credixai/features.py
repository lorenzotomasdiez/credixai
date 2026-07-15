"""Feature engineering pipeline (Tarea 2) para Home Credit Default Risk.

Reproduce, en forma de funciones reutilizables, la logica ya validada
interactivamente en notebooks/02_features.ipynb. Ver docs/informe-final.md
seccion 2 para la justificacion e interpretacion de cada paso.
"""

import warnings

import numpy as np
import pandas as pd


def load_application(data_dir: str) -> pd.DataFrame:
    """Concatena application_train/application_test para evitar train/test skew."""
    app_train = pd.read_csv(f"{data_dir}/application_train.csv")
    app_test = pd.read_csv(f"{data_dir}/application_test.csv").assign(TARGET=np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        app = pd.concat([app_train, app_test], ignore_index=True).assign(
            IS_TRAIN=lambda d: d["TARGET"].notna().astype(int)
        )
    return app


def clean_days_employed(app: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza el valor centinela 365243 por NaN y preserva la senal con un flag."""
    app = app.copy()
    app["DAYS_EMPLOYED_ANOM"] = app["DAYS_EMPLOYED"] == 365243
    app["DAYS_EMPLOYED"] = app["DAYS_EMPLOYED"].replace(365243, np.nan)
    return app


def add_business_ratios(app: pd.DataFrame) -> pd.DataFrame:
    """Agrega credit-to-income, annuity-to-income y credit-to-goods (prd.md S7.2)."""
    app = app.copy()
    app["credit_to_income"] = app["AMT_CREDIT"] / app["AMT_INCOME_TOTAL"].replace(0, np.nan)
    app["annuity_to_income"] = app["AMT_ANNUITY"] / app["AMT_INCOME_TOTAL"].replace(0, np.nan)
    app["credit_to_goods"] = app["AMT_CREDIT"] / app["AMT_GOODS_PRICE"].replace(0, np.nan)
    return app


def aggregate_bureau(data_dir: str) -> pd.DataFrame:
    bureau = pd.read_csv(f"{data_dir}/bureau.csv")
    return bureau.groupby("SK_ID_CURR").agg(
        bureau_count=("SK_ID_BUREAU", "count"),
        bureau_active_count=("CREDIT_ACTIVE", lambda s: (s == "Active").sum()),
        bureau_credit_sum_mean=("AMT_CREDIT_SUM", "mean"),
        bureau_credit_sum_max=("AMT_CREDIT_SUM", "max"),
        bureau_debt_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
        bureau_debt_mean=("AMT_CREDIT_SUM_DEBT", "mean"),
        bureau_overdue_sum=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        bureau_overdue_max=("AMT_CREDIT_SUM_OVERDUE", "max"),
        bureau_annuity_mean=("AMT_ANNUITY", "mean"),
    )


def aggregate_bureau_balance(data_dir: str, bureau_ids: pd.DataFrame) -> pd.DataFrame:
    """bureau_ids: columnas SK_ID_CURR/SK_ID_BUREAU de bureau.csv, para el join de dos saltos."""
    bureau_balance = pd.read_csv(f"{data_dir}/bureau_balance.csv")

    status_to_dpd = {"C": 0, "X": 0, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
    bureau_balance["dpd_level"] = bureau_balance["STATUS"].map(status_to_dpd)

    bb_per_credit = bureau_balance.groupby("SK_ID_BUREAU").agg(
        months_count=("MONTHS_BALANCE", "count"),
        months_dpd=("dpd_level", lambda s: (s > 0).sum()),
        worst_dpd_level=("dpd_level", "max"),
    )

    bb_with_curr = bureau_ids.merge(bb_per_credit, on="SK_ID_BUREAU", how="inner")

    return bb_with_curr.groupby("SK_ID_CURR").agg(
        bb_credits_with_history=("SK_ID_BUREAU", "count"),
        bb_months_dpd_sum=("months_dpd", "sum"),
        bb_worst_dpd_level_max=("worst_dpd_level", "max"),
        bb_months_count_mean=("months_count", "mean"),
    )


def aggregate_previous_application(data_dir: str) -> pd.DataFrame:
    previous_application = pd.read_csv(f"{data_dir}/previous_application.csv")
    return previous_application.groupby("SK_ID_CURR").agg(
        prev_count=("SK_ID_PREV", "count"),
        prev_ever_refused=("NAME_CONTRACT_STATUS", lambda s: (s == "Refused").any()),
        prev_approval_rate=("NAME_CONTRACT_STATUS", lambda s: (s == "Approved").mean()),
        prev_credit_mean=("AMT_CREDIT", "mean"),
        prev_annuity_mean=("AMT_ANNUITY", "mean"),
        prev_days_decision_mean=("DAYS_DECISION", "mean"),
    )


def aggregate_pos_cash(data_dir: str) -> pd.DataFrame:
    pos_cash = pd.read_csv(f"{data_dir}/POS_CASH_balance.csv")
    pos_cash["dpd_flag"] = pos_cash["SK_DPD"] > 0
    return pos_cash.groupby("SK_ID_CURR").agg(
        pos_credits_count=("SK_ID_PREV", "nunique"),
        pos_months_count=("SK_ID_PREV", "count"),
        pos_dpd_rate=("dpd_flag", "mean"),
        pos_dpd_max=("SK_DPD", "max"),
    )


def aggregate_credit_card(data_dir: str) -> pd.DataFrame:
    credit_card = pd.read_csv(f"{data_dir}/credit_card_balance.csv")
    credit_card["utilization_ratio"] = (
        credit_card["AMT_BALANCE"] / credit_card["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    )
    credit_card["dpd_flag"] = credit_card["SK_DPD"] > 0
    return credit_card.groupby("SK_ID_CURR").agg(
        cc_cards_count=("SK_ID_PREV", "nunique"),
        cc_utilization_mean=("utilization_ratio", "mean"),
        cc_utilization_max=("utilization_ratio", "max"),
        cc_dpd_rate=("dpd_flag", "mean"),
    )


def aggregate_installments(data_dir: str) -> pd.DataFrame:
    installments = pd.read_csv(f"{data_dir}/installments_payments.csv")
    installments["payment_delay"] = installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
    installments["payment_shortfall"] = installments["AMT_INSTALMENT"] - installments["AMT_PAYMENT"]
    installments["late_flag"] = installments["payment_delay"] > 5
    return installments.groupby("SK_ID_CURR").agg(
        inst_delay_mean=("payment_delay", "mean"),
        inst_delay_max=("payment_delay", "max"),
        inst_late_rate=("late_flag", "mean"),
        inst_shortfall_mean=("payment_shortfall", "mean"),
    )


NO_RECORD_SPECS = {
    "bureau_no_record": "bureau_count",
    "bb_no_record": "bb_credits_with_history",
    "prev_no_record": "prev_count",
    "pos_no_record": "pos_credits_count",
    "cc_no_record": "cc_cards_count",
    "inst_no_record": "inst_delay_mean",
}


def add_no_record_flags(features: pd.DataFrame) -> pd.DataFrame:
    """Preserva la cobertura por tabla como feature explicita antes del one-hot."""
    features = features.copy()
    for flag_col, ref_col in NO_RECORD_SPECS.items():
        features[flag_col] = features[ref_col].isna()
    return features


def encode_categoricals(features: pd.DataFrame) -> pd.DataFrame:
    """One-hot de las categoricas reales; prev_ever_refused se trata aparte (flag de 3 estados)."""
    features = features.copy()
    cat_cols = features.select_dtypes(include=["object", "str"]).columns.tolist()
    onehot_cols = [c for c in cat_cols if c != "prev_ever_refused"]

    if "prev_ever_refused" in cat_cols:
        features["prev_ever_refused"] = features["prev_ever_refused"].map({True: 1, False: 0})

    return pd.get_dummies(features, columns=onehot_cols, dummy_na=True)


def build_feature_table(data_dir: str) -> pd.DataFrame:
    """Orquesta el pipeline completo de la Tarea 2 y devuelve la tabla final a nivel SK_ID_CURR."""
    app = load_application(data_dir)
    app = clean_days_employed(app)
    app = add_business_ratios(app)

    bureau_ids = pd.read_csv(f"{data_dir}/bureau.csv", usecols=["SK_ID_CURR", "SK_ID_BUREAU"])

    bureau_agg = aggregate_bureau(data_dir)
    bb_agg = aggregate_bureau_balance(data_dir, bureau_ids)
    prev_agg = aggregate_previous_application(data_dir)
    pos_agg = aggregate_pos_cash(data_dir)
    cc_agg = aggregate_credit_card(data_dir)
    inst_agg = aggregate_installments(data_dir)

    features = (
        app
        .merge(bureau_agg, on="SK_ID_CURR", how="left")
        .merge(bb_agg, on="SK_ID_CURR", how="left")
        .merge(prev_agg, on="SK_ID_CURR", how="left")
        .merge(pos_agg, on="SK_ID_CURR", how="left")
        .merge(cc_agg, on="SK_ID_CURR", how="left")
        .merge(inst_agg, on="SK_ID_CURR", how="left")
    )
    assert features["SK_ID_CURR"].is_unique, "SK_ID_CURR duplicado tras el merge"
    assert len(features) == len(app), "el merge cambio la cantidad de filas"

    features = add_no_record_flags(features)
    features = encode_categoricals(features)
    return features
