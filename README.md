# Titanic Survival Prediction

Simple machine learning project to predict Titanic passenger survival with main focus on feature engineering and ensemble-style model comparison between Decision Tree, Random Forest, and XGBoost.

## Workflow in `prediction.ipynb`
1. Install and import required libraries (`pandas`, `numpy`, `seaborn`, `scikit-learn`, `xgboost`, `scikit-optimize`).
2. Load train/test data.
3. Combine datasets for consistent preprocessing.
4. Feature engineering:
   - Extract `Title` from passenger names.
   - Normalize rare/similar titles.
   - Impute missing values (`Age`, `Fare`, `Embarked`).
   - Create binned features (`CatAge`, `CatFare`).
   - Create family-size feature (`Fam_Size`).
5. Drop low-value columns (`Cabin`, `Name`, `PassengerId`, `Ticket`) and one-hot encode categorical features.
6. Split processed data back into train/test sets.
7. Train and tune models:
   - `DecisionTreeClassifier` + `RandomizedSearchCV`
   - `RandomForestClassifier` + `RandomizedSearchCV`
   - `XGBClassifier` + `BayesSearchCV`
8. Print best CV scores for model comparison.

## Notes
- Current notebook paths use `/content/train.csv` and `/content/test.csv` (Google Colab style).
- For local VS Code runs, update to relative paths, for example:
  - `pd.read_csv('train.csv')`
  - `pd.read_csv('test.csv')`

## Quick Start
1. Open `prediction.ipynb`.
2. Run all cells in order.
3. Compare printed CV scores to choose the best model.
