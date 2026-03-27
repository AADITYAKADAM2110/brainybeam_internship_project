from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import CHART_LABELS
from src.project import predict_survival


def show_overview(project) -> None:
    st.header("Overview")
    cols = st.columns(4)
    cols[0].metric("Passengers", project["summary"]["Passengers"])
    cols[1].metric("Survival Rate", f"{project['summary']['Survival Rate (%)']:.1f}%")
    cols[2].metric("Average Age", f"{project['summary']['Average Age']:.1f}")
    cols[3].metric("Average Fare", f"{project['summary']['Average Fare']:.1f}")
    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Summary")
        st.dataframe(pd.DataFrame({"Metric": project["summary"].keys(), "Value": project["summary"].values()}), width="stretch", hide_index=True)
    with right:
        st.subheader("Notes")
        st.dataframe(project["notes"], width="stretch", hide_index=True)


def show_data(project) -> None:
    st.header("Data")
    tabs = st.tabs(["Raw Data", "Processed Data", "Missing Values"])
    with tabs[0]:
        st.dataframe(project["raw_data"].head(20), width="stretch")
    with tabs[1]:
        st.dataframe(project["processed_data"].head(20), width="stretch")
    with tabs[2]:
        st.dataframe(project["missing_data"], width="stretch", hide_index=True)


def show_charts(project) -> None:
    st.header("Charts")
    chart = st.selectbox("Select chart", list(CHART_LABELS.keys()), format_func=CHART_LABELS.get)
    show_image(project["charts"][chart], CHART_LABELS[chart])


def show_model(project) -> None:
    st.header("Model")
    st.caption(f"Best model: {project['best_model']}")
    st.dataframe(project["scores"], width="stretch", hide_index=True)
    left, right = st.columns(2)
    with left:
        show_image(project["charts"]["model_comparison"], "Model Comparison")
    with right:
        show_image(project["charts"]["feature_importance"], "Feature Importance")


def show_predictor(project) -> None:
    st.header("Predictor")
    st.caption("Predict if a passenger survived or not.")
    left, right = st.columns(2)
    with left:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
        sex = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 1, 80, 29)
        sibsp = st.slider("Siblings / Spouses", 0, 8, 0)
    with right:
        parch = st.slider("Parents / Children", 0, 6, 0)
        fare = st.slider("Fare", 0.0, 550.0, 32.0, step=1.0)
        embarked = st.selectbox("Embarked", ["S", "C", "Q"])
    values = {
        "Pclass": pclass,
        "Sex": 1 if sex.lower() == "male" else 0,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": {"S": 0, "C": 1, "Q": 2}[embarked],
    }
    if st.button("Predict", type="primary"):
        pred, prob, name = predict_survival(values)
        if pred == 1:
            st.success(f"Prediction: Survived ({prob[1] * 100:.1f}%)")
        else:
            st.error(f"Prediction: Did not survive ({100 - prob[1] * 100:.1f}%)")
        st.info(f"Model used: {name}")


def show_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.warning(f"Missing file: {path.name}")
