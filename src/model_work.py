import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import FEATURES, MANIFEST_PATH, MODELS_DIR, RANDOM_STATE, TARGET
from src.data_work import prediction_frame


def model_list():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }


def train_models(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rows = []
    trained = {}
    for name, model in model_list().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append(
            {
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, pred), 4),
                "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
                "Recall": round(recall_score(y_test, pred, zero_division=0), 4),
                "F1 Score": round(f1_score(y_test, pred, zero_division=0), 4),
            }
        )
        trained[name] = model
        joblib.dump(model, MODELS_DIR / f"{name.lower().replace(' ', '_')}.pkl")

    scores = pd.DataFrame(rows).sort_values(["Accuracy", "F1 Score"], ascending=False).reset_index(drop=True)
    best_model = scores.loc[0, "Model"]
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump({"best_model": best_model, "features": FEATURES}, MANIFEST_PATH)

    importances = pd.DataFrame(columns=["Feature", "Importance"])
    rf = trained.get("Random Forest")
    if rf is not None and hasattr(rf, "feature_importances_"):
        importances = pd.DataFrame({"Feature": FEATURES, "Importance": rf.feature_importances_}).sort_values(
            "Importance", ascending=False
        )
    return scores, best_model, importances.reset_index(drop=True)


def predict(values: dict[str, float | int], model_name: str | None = None):
    info = joblib.load(MANIFEST_PATH)
    chosen = model_name or info["best_model"]
    model = joblib.load(MODELS_DIR / f"{chosen.lower().replace(' ', '_')}.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    row = prediction_frame(values)[info["features"]]
    prob = model.predict_proba(scaler.transform(row))[0]
    pred = int(model.predict(scaler.transform(row))[0])
    return pred, prob, chosen
