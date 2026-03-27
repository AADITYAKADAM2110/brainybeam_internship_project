from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import BAR_BLUE, BAR_GREEN, BAR_ORANGE, HEATMAP_COLORS, MODEL_COLORS, REPORTS_DIR, SURVIVAL_COLORS
from src.data_work import make_features, missing_table


sns.set_theme(style="whitegrid")


def save_charts(raw_df: pd.DataFrame, processed_df: pd.DataFrame, scores: pd.DataFrame, importances: pd.DataFrame):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "survival_by_gender": REPORTS_DIR / "survival_by_gender.png",
        "survival_by_class": REPORTS_DIR / "survival_by_class.png",
        "survival_by_embarked": REPORTS_DIR / "survival_by_embarked.png",
        "age_distribution": REPORTS_DIR / "age_distribution.png",
        "fare_distribution": REPORTS_DIR / "fare_distribution.png",
        "family_size_survival": REPORTS_DIR / "family_size_survival.png",
        "missing_values": REPORTS_DIR / "missing_values.png",
        "correlation_heatmap": REPORTS_DIR / "correlation_heatmap.png",
        "model_comparison": REPORTS_DIR / "model_comparison.png",
        "feature_importance": REPORTS_DIR / "feature_importance.png",
    }
    df = make_features(raw_df)
    _count(raw_df, "Sex", "Survival by Gender", files["survival_by_gender"])
    _count(raw_df, "Pclass", "Survival by Class", files["survival_by_class"])
    _count(df, "Embarked", "Survival by Embarked", files["survival_by_embarked"])
    _hist(raw_df, "Age", "Age Distribution by Survival", files["age_distribution"])
    _hist(raw_df, "Fare", "Fare Distribution by Survival", files["fare_distribution"])
    _family(df, files["family_size_survival"])
    _missing(missing_table(raw_df), files["missing_values"])
    _heatmap(processed_df, files["correlation_heatmap"])
    _scores(scores, files["model_comparison"])
    _importance(importances, files["feature_importance"])
    return files


def _save(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _count(df, column, title, path):
    plt.figure(figsize=(9, 6))
    sns.countplot(data=df, x=column, hue="Survived", palette=SURVIVAL_COLORS)
    plt.title(title)
    _save(path)


def _hist(df, column, title, path):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column, hue="Survived", kde=True, multiple="stack", palette=SURVIVAL_COLORS)
    plt.title(title)
    _save(path)


def _family(df, path):
    plot_df = df.groupby("FamilySize", as_index=False)["Survived"].mean()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="FamilySize", y="Survived", color=BAR_BLUE)
    plt.title("Survival by Family Size")
    plt.ylim(0, 1)
    _save(path)


def _missing(df, path):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Column", y="Missing", color=BAR_ORANGE)
    plt.title("Missing Values")
    plt.xticks(rotation=45, ha="right")
    _save(path)


def _heatmap(df, path):
    plt.figure(figsize=(11, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap=HEATMAP_COLORS)
    plt.title("Correlation Heatmap")
    _save(path)


def _scores(df, path):
    plot_df = df.melt(id_vars="Model", value_vars=["Accuracy", "F1 Score"], var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="Score", y="Model", hue="Metric", palette=MODEL_COLORS)
    plt.title("Model Comparison")
    plt.xlim(0, 1)
    _save(path)


def _importance(df, path):
    if df.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df.head(10), x="Importance", y="Feature", color=BAR_GREEN)
    plt.title("Feature Importance")
    _save(path)
