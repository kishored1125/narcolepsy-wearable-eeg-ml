from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def candidate_models(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="linear",
                        class_weight="balanced",
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingClassifier(random_state=random_state)),
            ]
        ),
    }


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Pipeline]]:
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)
    groups = groups.reset_index(drop=True) if groups is not None else None
    cv = _make_cv(y, groups, n_splits=n_splits, random_state=random_state)

    rows = []
    predictions: dict[str, np.ndarray] = {}
    fitted: dict[str, Pipeline] = {}
    for name, model in candidate_models(random_state=random_state).items():
        estimator = clone(model)
        try:
            if groups is not None:
                y_pred = cross_val_predict(estimator, X, y, cv=cv, groups=groups, method="predict")
            else:
                y_pred = cross_val_predict(estimator, X, y, cv=cv, method="predict")
            predictions[name] = y_pred
            row = classification_metrics(y, y_pred)
            row["model"] = name
            rows.append(row)
            fitted[name] = clone(model).fit(X, y)
        except Exception as exc:
            warnings.warn(f"Model {name} failed: {exc}")

    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        metrics = metrics[["model"] + [c for c in metrics.columns if c != "model"]]
        metrics = metrics.sort_values("balanced_accuracy", ascending=False)
    return metrics, predictions, fitted


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    labels = sorted(pd.Series(y_true).dropna().unique())
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if len(labels) == 2:
        positive = labels[-1]
        out["sensitivity_positive"] = recall_score(y_true == positive, y_pred == positive, zero_division=0)
        out["specificity_positive"] = recall_score(y_true != positive, y_pred != positive, zero_division=0)
    return out


def best_model_name(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        raise ValueError("No model metrics available.")
    return str(metrics.iloc[0]["model"])


def feature_importance_table(model: Pipeline, feature_names: list[str], top_n: int = 30) -> pd.DataFrame:
    final = model.named_steps["model"]
    if hasattr(final, "feature_importances_"):
        values = final.feature_importances_
    elif hasattr(final, "coef_"):
        values = np.mean(np.abs(final.coef_), axis=0)
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    return (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def confusion_matrix_frame(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    labels = sorted(pd.Series(y_true).dropna().unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])


def _make_cv(y: pd.Series, groups: pd.Series | None, n_splits: int, random_state: int):
    min_class_count = int(y.value_counts().min())
    if groups is not None and groups.nunique() >= 2:
        splits = max(2, min(n_splits, min_class_count, int(groups.nunique())))
        return StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=random_state)
    splits = max(2, min(n_splits, min_class_count))
    return StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)


def binary_auc_if_available(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> float | None:
    if y.nunique() != 2 or not hasattr(model, "predict_proba"):
        return None
    try:
        proba = model.predict_proba(X)[:, 1]
        return float(roc_auc_score(y, proba))
    except Exception:
        return None
