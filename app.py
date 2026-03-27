import streamlit as st

from src.app_pages import (
    show_charts,
    show_data,
    show_model,
    show_overview,
    show_predictor,
)
from src.project import build_project


st.set_page_config(
    page_title="Titanic Data Analysis",
    page_icon="🛳️",
    layout="wide"
)


@st.cache_resource(show_spinner=False)
def get_project():
    return build_project()


project = get_project()

st.title("TitanicAnalytics")
st.caption("Titanic dataset analysis using EDA and machine learning.")
st.markdown("**Made by Aaditya Kadam (221250107012)**")
st.markdown("**Shree Swaminarayan Institute of Technology, Gandhinagar**")
st.markdown("**External Guide: Sagar Jasani**")
st.markdown("**Company: Brainybeam Info-Tech PVT LTD**")


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Data", "Charts", "Model", "Predictor"],
)
st.sidebar.markdown("---")
st.sidebar.markdown("### Abstract")
st.sidebar.write("This project focuses on predicting Titanic passenger survival by analyzing factors influencing the likelihood of survival. Utilizing a dataset containing passenger information such as age, gender, class, and embarked port, the project aims to develop a predictive model for determining the probability of survival. The analysis involves exploratory data analysis, feature engineering, and model training to create a reliable tool for understanding and predicting survival outcomes. The resulting model contributes to historical analysis and serves as a learning tool for exploring the impact of various factors on passenger survival during this significant maritime event.")
st.sidebar.markdown("**Aaditya Kadam**")

if page == "Overview":
    show_overview(project)
elif page == "Data":
    show_data(project)
elif page == "Charts":
    show_charts(project)
elif page == "Model":
    show_model(project)
else:
    show_predictor(project)
