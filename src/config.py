from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Titanic-Dataset.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SIMPLE_DATA_PATH = PROCESSED_DATA_DIR / "titanic_processed.csv"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "src" / "models"
SCORES_PATH = REPORTS_DIR / "model_metrics.csv"
EDA_PATH = REPORTS_DIR / "eda_summary.csv"
MANIFEST_PATH = MODELS_DIR / "model_manifest.pkl"

TARGET = "Survived"
RANDOM_STATE = 42

FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "FamilySize",
    "IsAlone",
    "FarePerPerson",
]

CHART_LABELS = {
    "survival_by_gender": "Survival by Gender",
    "survival_by_class": "Survival by Class",
    "survival_by_embarked": "Survival by Embarked",
    "age_distribution": "Age Distribution by Survival",
    "fare_distribution": "Fare Distribution by Survival",
    "family_size_survival": "Family Size vs Survival",
    "missing_values": "Missing Values Overview",
    "correlation_heatmap": "Correlation Heatmap",
    "model_comparison": "Model Comparison",
    "feature_importance": "Feature Importance",
}

SURVIVAL_COLORS = ["#4E79A7", "#E15759"]
BAR_BLUE = "#4E79A7"
BAR_ORANGE = "#F28E2B"
BAR_GREEN = "#59A14F"
HEATMAP_COLORS = "Blues"
MODEL_COLORS = ["#4E79A7", "#F28E2B"]


def ensure_folders() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
