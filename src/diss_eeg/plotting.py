from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_barplot(df: pd.DataFrame, x: str, y: str, path, title: str, rotation: int = 30) -> None:
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x=x, y=y)
    plt.title(title)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_confusion_matrix(cm: pd.DataFrame, path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_feature_importance(df: pd.DataFrame, path, title: str) -> None:
    if df.empty:
        return
    plot_df = df.sort_values("importance", ascending=True).tail(20)
    plt.figure(figsize=(9, 7))
    sns.barplot(data=plot_df, x="importance", y="feature", orient="h")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
