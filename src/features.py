"""Feature engineering for the Titanic passenger list.

Every transform here is a plain function of a single DataFrame so the same
code runs in the notebook, in training, and at prediction time.
"""
import numpy as np
import pandas as pd

TITLE_MAP = {
    "Mr": "Mr", "Mrs": "Mrs", "Mme": "Mrs", "Ms": "Miss", "Miss": "Miss",
    "Mlle": "Miss", "Master": "Master",
}

NUMERIC = ["Age", "FarePerPerson", "FamilySize"]
CATEGORICAL = ["Pclass", "Sex", "Embarked", "Title", "Deck", "TravelGroup"]


def extract_title(name: str) -> str:
    """'Braund, Mr. Owen Harris' -> 'Mr'. Rare titles collapse into 'Rare'."""
    raw = name.split(",")[1].split(".")[0].strip()
    return TITLE_MAP.get(raw, "Rare")


def travel_group(size: int) -> str:
    if size == 1:
        return "alone"
    if size <= 4:
        return "small"
    return "large"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["Pclass"] = df["Pclass"].astype(str)
    out["Sex"] = df["Sex"]
    out["Embarked"] = df["Embarked"].fillna("S")
    out["Title"] = df["Name"].map(extract_title)
    out["Deck"] = df["Cabin"].fillna("U").str[0].replace({"T": "U"})
    out["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    out["TravelGroup"] = out["FamilySize"].map(travel_group)

    # Passengers on one ticket paid one fare between them, so split it.
    ticket_count = df.groupby("Ticket")["Ticket"].transform("count")
    out["FarePerPerson"] = np.log1p(df["Fare"].fillna(df["Fare"].median()) / ticket_count)

    # Age is left with gaps on purpose: TitleAgeImputer fills it inside the
    # model pipeline, so test rows never see medians learned from themselves.
    out["Age"] = df["Age"]
    return out[NUMERIC + CATEGORICAL]


class TitleAgeImputer:
    """Fill missing Age with the median age of passengers sharing the same Title."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby("Title")["Age"].median()
        self.overall_ = X["Age"].median()
        return self

    def transform(self, X):
        X = X.copy()
        fill = X["Title"].map(self.medians_).fillna(self.overall_)
        X["Age"] = X["Age"].fillna(fill)
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self
