from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


warnings.filterwarnings("ignore", category=FutureWarning)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def numeric_features(df: pd.DataFrame, blocked: set[str]) -> list[str]:
    return [c for c in df.columns if c not in blocked and pd.api.types.is_numeric_dtype(df[c])]


def clean_feature_matrix(df: pd.DataFrame, feature_cols: list[str], min_non_missing: float = 0.70) -> tuple[pd.DataFrame, list[str]]:
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    enough_data = [c for c in X.columns if X[c].notna().mean() >= min_non_missing]
    X = X[enough_data]
    variable = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    return X[variable], variable


def model_pipelines(k_features: int | str = "all", random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("select", SelectKBest(f_classif, k=k_features)),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("select", SelectKBest(f_classif, k=k_features)),
                ("model", SVC(kernel="linear", class_weight="balanced", probability=True, random_state=random_state)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("select", SelectKBest(f_classif, k=k_features)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        class_weight="balanced",
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("select", SelectKBest(f_classif, k=k_features)),
                ("model", GradientBoostingClassifier(random_state=random_state)),
            ]
        ),
    }


def classification_metrics(y_true: pd.Series, y_pred: np.ndarray, positive_label: str | None = None) -> dict[str, float]:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    if positive_label is not None:
        out[f"sensitivity_{positive_label}"] = recall_score(y_true == positive_label, y_pred == positive_label, zero_division=0)
        out["specificity_not_positive"] = recall_score(y_true != positive_label, y_pred != positive_label, zero_division=0)
    return out


def repeated_cv_model_comparison(
    X: pd.DataFrame,
    y: pd.Series,
    k_features: int | str,
    positive_label: str | None,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    rows = []
    for model_name, model in model_pipelines(k_features, random_state).items():
        for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = fitted.predict(X.iloc[test_idx])
            row = classification_metrics(y.iloc[test_idx], pred, positive_label=positive_label)
            row["model"] = model_name
            row["fold_id"] = fold_id
            if y.nunique() == 2:
                proba = fitted.predict_proba(X.iloc[test_idx])
                positive = positive_label or sorted(y.unique())[-1]
                pos_idx = list(fitted.classes_).index(positive)
                try:
                    row["roc_auc"] = roc_auc_score(y.iloc[test_idx] == positive, proba[:, pos_idx])
                except ValueError:
                    row["roc_auc"] = np.nan
            rows.append(row)
    fold_metrics = pd.DataFrame(rows)
    metric_cols = [c for c in fold_metrics.columns if c not in {"model", "fold_id"}]
    metrics = fold_metrics.groupby("model")[metric_cols].agg(["mean", "std"]).reset_index()
    metrics.columns = ["model"] + [f"{metric}_{stat}" for metric, stat in metrics.columns.to_flat_index()[1:]]
    sort_col = "balanced_accuracy_mean" if "balanced_accuracy_mean" in metrics.columns else "accuracy_mean"
    return metrics.sort_values(sort_col, ascending=False).reset_index(drop=True), fold_metrics


def single_cv_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    model: Pipeline,
    n_splits: int,
    random_state: int = 42,
) -> np.ndarray:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return cross_val_predict(clone(model), X, y, cv=cv, method="predict")


def confusion_frame(y_true: pd.Series, y_pred: np.ndarray, labels: list[str] | None = None) -> pd.DataFrame:
    labels = labels or sorted(pd.Series(y_true).dropna().unique())
    return pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )


def selected_feature_importance(model: Pipeline, X: pd.DataFrame, y: pd.Series, feature_cols: list[str]) -> pd.DataFrame:
    fitted = clone(model).fit(X, y)
    selected_mask = fitted.named_steps["select"].get_support()
    selected_features = list(np.asarray(feature_cols)[selected_mask])
    Xt = X.copy()
    for step_name in ["imputer", "scaler", "select"]:
        if step_name in fitted.named_steps:
            Xt = fitted.named_steps[step_name].transform(Xt)
    final = fitted.named_steps["model"]
    if hasattr(final, "feature_importances_"):
        intrinsic = np.asarray(final.feature_importances_, dtype=float)
    elif hasattr(final, "coef_"):
        intrinsic = np.mean(np.abs(final.coef_), axis=0)
    else:
        intrinsic = np.zeros(len(selected_features), dtype=float)
    perm = permutation_importance(final, Xt, y, scoring="balanced_accuracy", n_repeats=30, random_state=42, n_jobs=1)
    out = pd.DataFrame(
        {
            "feature": selected_features,
            "model_importance": intrinsic,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
        }
    )
    out["importance"] = np.where(out["permutation_importance_mean"].max() > 0, out["permutation_importance_mean"], out["model_importance"])
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def save_barplot(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x=x, y=y)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_confusion_plot(cm: pd.DataFrame, path: Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_importance_plot(df: pd.DataFrame, path: Path, title: str) -> None:
    if df.empty:
        return
    plot_df = df.sort_values("importance", ascending=True).tail(20)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=plot_df, x="importance", y="feature", orient="h")
    plt.title(title)
    plt.xlabel("Feature importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
