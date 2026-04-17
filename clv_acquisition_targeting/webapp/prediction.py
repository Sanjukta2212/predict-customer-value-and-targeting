"""
Train-once churn + CLV scoring for the CSR web app, with human-readable calculation traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]

NUMERIC_FEATURES = [
    "age",
    "tenure_months",
    "num_orders",
    "total_spend",
    "avg_order_value",
    "days_since_last_order",
    "email_opens_30d",
    "app_sessions_30d",
    "public_uninsured_rate_proxy",
]
CATEGORICAL_FEATURES = ["region"]
TARGET = "churned"
INTERNAL = ("_acquisition_prob", "_monthly_spend_intensity")


def load_config() -> Dict[str, Any]:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def make_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def _build_classifier(cfg: Dict[str, Any], seed: int) -> GradientBoostingClassifier:
    cm = cfg["churn_model"]
    return GradientBoostingClassifier(
        random_state=seed,
        n_estimators=int(cm["n_estimators"]),
        max_depth=int(cm["max_depth"]),
        learning_rate=float(cm["learning_rate"]),
        min_samples_leaf=int(cm["min_samples_leaf"]),
    )


def _training_csv_path() -> Path:
    cfg = load_config()
    rel = str(cfg.get("data", {}).get("customer_training_csv", "data/customers.csv"))
    p = Path(rel)
    return p if p.is_absolute() else (ROOT / p)


def build_training_frame() -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(_training_csv_path())
    drop_cols = list(INTERNAL) + [TARGET, "customer_id"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[TARGET].to_numpy()
    return X, y


_model: Any = None
_training_medians: Dict[str, float] = {}
_PROXY_BY_REGION: Dict[str, float] = {}


def _region_proxy_lookup() -> Dict[str, float]:
    global _PROXY_BY_REGION
    if not _PROXY_BY_REGION:
        t = pd.read_csv(_training_csv_path())
        if "public_uninsured_rate_proxy" in t.columns:
            _PROXY_BY_REGION = t.groupby("region")["public_uninsured_rate_proxy"].first().to_dict()
        else:
            _PROXY_BY_REGION = {r: 0.0 for r in ["NE", "SE", "MW", "W", "S"]}
    return _PROXY_BY_REGION


def get_model() -> Any:
    if _model is None:
        raise RuntimeError("Model not trained yet.")
    return _model


def train_model() -> None:
    global _model, _training_medians, _PROXY_BY_REGION
    _PROXY_BY_REGION = {}
    cfg = load_config()
    seed = int(cfg["project"]["random_seed"])
    X, y = build_training_frame()
    for c in NUMERIC_FEATURES:
        _training_medians[c] = float(X[c].median())
    base = Pipeline(
        steps=[
            ("prep", make_preprocessor()),
            ("clf", _build_classifier(cfg, seed)),
        ]
    )
    calibrated = CalibratedClassifierCV(
        estimator=base,
        method="isotonic",
        cv=min(3, max(2, len(y) // 500)),
    )
    calibrated.fit(X, y)
    _model = calibrated


def row_from_payload(payload: Dict[str, Any]) -> pd.DataFrame:
    region = str(payload["region"])
    proxy = float(_region_proxy_lookup().get(region, 0.0))
    row = {
        "age": int(payload["age"]),
        "tenure_months": int(payload["tenure_months"]),
        "num_orders": int(payload["num_orders"]),
        "total_spend": float(payload["total_spend"]),
        "avg_order_value": float(payload["avg_order_value"]),
        "days_since_last_order": int(payload["days_since_last_order"]),
        "email_opens_30d": int(payload["email_opens_30d"]),
        "app_sessions_30d": int(payload["app_sessions_30d"]),
        "public_uninsured_rate_proxy": proxy,
        "region": region,
    }
    return pd.DataFrame([row])


def predict_churn_proba(model: Any, X: pd.DataFrame) -> float:
    return float(model.predict_proba(X)[0, 1])


def _historical_monthly_margin(total_spend: float, tenure_months: int, margin_rate: float) -> float:
    tenure = max(float(tenure_months), 1.0)
    return (total_spend * margin_rate) / tenure


def _future_margin_sum(
    monthly_margin: float,
    churn_prob: float,
    discount_monthly: float,
    max_horizon_months: int,
    min_p: float,
    max_p: float,
) -> Tuple[float, List[Dict[str, Any]], float]:
    p = float(np.clip(churn_prob, min_p, max_p))
    d = discount_monthly
    m = monthly_margin
    t = np.arange(1, max_horizon_months + 1, dtype=float)
    survival = np.power(1.0 - p, t - 1.0)
    discount = np.power(1.0 + d, t)
    per_month = m * survival / discount
    future_total = float(per_month.sum())
    preview_rows: List[Dict[str, Any]] = []
    for i in range(min(6, len(per_month))):
        month = int(i + 1)
        preview_rows.append(
            {
                "month": month,
                "survival_to_start_of_month": float(np.power(1.0 - p, month - 1)),
                "discount_factor": float(np.power(1.0 + d, month)),
                "expected_margin_that_month": float(per_month[i]),
            }
        )
    return future_total, preview_rows, p


def build_calculation_trace(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Returns (result_summary, ordered_trace_steps) for UI / streaming.
    """
    cfg = load_config()
    clv_cfg = cfg["clv"]
    margin_rate = float(clv_cfg["margin_rate"])
    discount_monthly = float(clv_cfg["discount_monthly"])
    max_h = int(clv_cfg["max_horizon_months"])
    min_p = float(clv_cfg["min_churn_prob"])
    max_p = float(clv_cfg["max_churn_prob"])

    model = get_model()
    X = row_from_payload(payload)

    steps: List[Dict[str, Any]] = []

    steps.append(
        {
            "step": 1,
            "title": "Inputs received",
            "detail": "Mapped CSR fields to the rating factors the model was trained on.",
            "data": {
                "customer_age": int(payload["age"]),
                "months_as_customer": int(payload["tenure_months"]),
                "policies_or_service_events": int(payload["num_orders"]),
                "total_premiums_or_payments": float(payload["total_spend"]),
                "typical_payment_size": float(payload["avg_order_value"]),
                "days_since_last_contact": int(payload["days_since_last_order"]),
                "messages_opened_last_30_days": int(payload["email_opens_30d"]),
                "digital_sessions_last_30_days": int(payload["app_sessions_30d"]),
                "region": str(payload["region"]),
            },
        }
    )

    medians = {k: _training_medians.get(k) for k in NUMERIC_FEATURES}
    steps.append(
        {
            "step": 2,
            "title": "How this compares to typical customers",
            "detail": "Training-data medians are shown for context (the model uses scaled values internally).",
            "data": {"your_values": X.iloc[0].to_dict(), "typical_median_numeric_factors": medians},
        }
    )

    churn_p = predict_churn_proba(model, X)
    risk_label = "lower" if churn_p < 0.33 else "moderate" if churn_p < 0.55 else "higher"
    steps.append(
        {
            "step": 3,
            "title": "Churn / lapse risk (next period)",
            "detail": (
                "A gradient-boosting classifier with calibrated probabilities was fit on historical "
                "customers. It outputs the chance this account lapses or churns in the next decision window."
            ),
            "data": {
                "estimated_probability_of_churn_or_lapse": round(churn_p, 4),
                "plain_language_risk_band": risk_label,
            },
        }
    )

    ts = float(payload["total_spend"])
    ten = int(payload["tenure_months"])
    m = _historical_monthly_margin(ts, ten, margin_rate)
    historical_component = m * float(ten)

    steps.append(
        {
            "step": 4,
            "title": "Historical value (already realized)",
            "detail": (
                "We approximate average monthly margin from lifetime payments and tenure, "
                "then multiply by months on file."
            ),
            "data": {
                "margin_rate_used": margin_rate,
                "formula_monthly_margin": "(total_payments × margin_rate) ÷ max(months_as_customer, 1)",
                "monthly_margin_estimate": round(m, 2),
                "historical_value": round(historical_component, 2),
            },
        }
    )

    future_total, preview, p_used = _future_margin_sum(
        m, churn_p, discount_monthly, max_h, min_p, max_p
    )
    steps.append(
        {
            "step": 5,
            "title": "Future value (discounted, churn-adjusted)",
            "detail": (
                "We sum up to "
                f"{max_h} monthly slices. Each month survives with (1 − p) where p is churn risk; "
                "each slice is discounted by the monthly discount rate."
            ),
            "data": {
                "churn_probability_used_after_clipping": round(p_used, 4),
                "monthly_discount_rate": discount_monthly,
                "horizon_months": max_h,
                "first_months_breakdown": preview,
                "sum_all_months_future_margin": round(future_total, 2),
            },
        }
    )

    clv_total = historical_component + future_total
    steps.append(
        {
            "step": 6,
            "title": "Customer lifetime value (CLV) estimate",
            "detail": "CLV here = historical realized contribution proxy + expected discounted future margin.",
            "data": {
                "historical_part": round(historical_component, 2),
                "future_part": round(future_total, 2),
                "clv_total": round(clv_total, 2),
            },
        }
    )

    result = {
        "clv": round(clv_total, 2),
        "churn_probability": round(churn_p, 4),
        "risk_band": risk_label,
        "historical_value": round(historical_component, 2),
        "future_value": round(future_total, 2),
        "currency_note": "Values are in the same money units as total payments (e.g., USD).",
    }
    return result, steps
