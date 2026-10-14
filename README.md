# Titanic Survival Model

Predicting which passengers survived the Titanic from their manifest details, with most of the effort going into feature engineering rather than model hunting.

## Results

| Stage | Model | ROC-AUC |
|---|---|---|
| 5-fold CV (tuned) | Random forest | 0.897 |
| 5-fold CV (tuned) | Histogram gradient boosting | 0.893 |
| 5-fold CV | Logistic regression | 0.867 |
| **Held-out test set (179 passengers)** | **Random forest** | **0.841** (84% accuracy) |

## What's different about this approach

Instead of feeding raw columns to a model, `src/features.py` rebuilds the manifest into features that carry meaning:

- **Title** from the name (Mr, Mrs, Miss, Master, Rare). It captures sex, age group and status in one field.
- **Deck** from the cabin's first letter, with `U` when no cabin was recorded. Missing cabins are informative, not noise.
- **Fare per person.** Several passengers often shared one ticket and one fare, so the fare is split by ticket group, then log-scaled.
- **Family size and travel group** (alone / small / large).
- **Age imputed by title** inside the model pipeline, so held-out passengers never leak into the medians.

Once these were in place, three very different model families scored within a few points of each other. Most of the signal comes from the features.

## Project layout

```
data/titanic.csv               passenger manifest (891 passengers)
notebook/titanic_survival.ipynb full analysis: EDA, features, CV, tuning, test, explanation
src/features.py                feature engineering, shared by notebook and prediction
src/predict.py                 score a CSV of passengers with the saved model
models/survival_model.joblib   final model, refit on all labelled passengers
```

## Run it

```bash
pip install -r requirements.txt
jupyter notebook notebook/titanic_survival.ipynb
python src/predict.py data/titanic.csv predictions.csv
```

## Data

The standard public Titanic passenger dataset (891 labelled passengers), from the [Data Science Dojo datasets repo](https://github.com/datasciencedojo/datasets).

## Author

Jesse Ngugi · [github.com/ngugiauraa](https://github.com/ngugiauraa)
