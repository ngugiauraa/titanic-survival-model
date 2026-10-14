"""Score passengers from a CSV with the saved model.

Usage:
    python src/predict.py data/titanic.csv predictions.csv

The input needs the original manifest columns:
Name, Sex, Pclass, Age, SibSp, Parch, Fare, Ticket, Cabin, Embarked.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).parent))
from features import build_features  # noqa: E402

MODEL_PATH = Path(__file__).parent.parent / "models" / "survival_model.joblib"


def main(src: str, dest: str) -> None:
    passengers = pd.read_csv(src)
    model = joblib.load(MODEL_PATH)
    passengers["survival_probability"] = model.predict_proba(build_features(passengers))[:, 1].round(3)
    passengers["predicted_survived"] = (passengers["survival_probability"] >= 0.5).astype(int)
    passengers.to_csv(dest, index=False)
    print(f"Scored {len(passengers)} passengers -> {dest}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
