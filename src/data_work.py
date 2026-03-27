import pandas as pd

from src.config import FEATURES, TARGET


SEX_MAP = {"male": 1, "female": 0}
EMBARKED_MAP = {"S": 0, "C": 1, "Q": 2}


def load_data(path):
    return pd.read_csv(path)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode().iloc[0])
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["FarePerPerson"] = (df["Fare"] / df["FamilySize"]).round(2)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = make_features(df)
    df["Sex"] = df["Sex"].str.lower().map(SEX_MAP)
    df["Embarked"] = df["Embarked"].map(EMBARKED_MAP)
    return df[FEATURES + [TARGET]].copy()


def summary_numbers(raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> dict[str, float | int]:
    return {
        "Passengers": int(len(raw_df)),
        "Survival Rate (%)": round(raw_df[TARGET].mean() * 100, 2),
        "Average Age": round(raw_df["Age"].mean(), 2),
        "Average Fare": round(raw_df["Fare"].mean(), 2),
        "Missing Values": int(raw_df.isna().sum().sum()),
        "Features Used": len(processed_df.columns) - 1,
    }


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.isna().sum().reset_index()
    out.columns = ["Column", "Missing"]
    out["Percent"] = (out["Missing"] / len(df) * 100).round(2)
    return out.sort_values(["Missing", "Column"], ascending=[False, True]).reset_index(drop=True)


def notes_table(df: pd.DataFrame) -> pd.DataFrame:
    df = make_features(df)
    return pd.DataFrame(
        {
            "Topic": ["Gender", "Class", "Embarked", "Alone", "Family"],
            "Note": [
                "Female passengers had a higher survival rate.",
                "Passengers in higher class survived more often.",
                "Embarked point also shows some difference.",
                f"{df.loc[df['IsAlone'] == 1, TARGET].mean() * 100:.1f}% survival rate",
                f"{df.loc[df['IsAlone'] == 0, TARGET].mean() * 100:.1f}% survival rate",
            ],
        }
    )


def prediction_frame(values: dict[str, float | int]) -> pd.DataFrame:
    row = pd.DataFrame([values])
    row["FamilySize"] = row["SibSp"] + row["Parch"] + 1
    row["IsAlone"] = (row["FamilySize"] == 1).astype(int)
    row["FarePerPerson"] = (row["Fare"] / row["FamilySize"]).round(2)
    return row[FEATURES]
