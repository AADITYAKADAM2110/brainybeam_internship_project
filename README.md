# Titanic Data Analysis

This is my internship project based on the Titanic dataset. I first explored the dataset in notebooks and then built a simple Streamlit app for analysis, charts, model comparison, and prediction.

**Made by Aaditya Kadam**

## Project Overview

- Data cleaning and preprocessing
- Exploratory data analysis
- Feature engineering
- Model training and comparison
- Streamlit dashboard for prediction

## Project Structure

```text
.
├── app.py
├── main.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_model_testing.ipynb
│   └── 05_final_notes.ipynb
├── reports/
├── src/
│   ├── app_pages.py
│   ├── chart_work.py
│   ├── config.py
│   ├── data_work.py
│   ├── model_work.py
│   └── project.py
├── requirements.txt
└── pyproject.toml
```

## App Sections

- Overview
- Data
- Charts
- Model
- Predictor

## Charts Generated

- Survival by gender
- Survival by class
- Survival by embarked point
- Age distribution by survival
- Fare distribution by survival
- Survival by family size
- Missing values
- Correlation heatmap
- Model comparison
- Feature importance

## Models Used

- Logistic Regression
- Random Forest
- SVM
- KNN

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate processed data, charts, and model files:

```bash
python main.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Streamlit Deployment

1. Push this project to GitHub.
2. Open Streamlit Community Cloud.
3. Click `New app`.
4. Select your GitHub repository and branch.
5. Set the main file path to `app.py`.
6. Deploy the app.

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
