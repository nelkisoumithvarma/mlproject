# Model Results

## Baseline Model (RandomForest – Numeric Only)
Single Split MAE: 17,024

---

## RandomForest (Engineered + Categorical Features)
Single Split MAE: 16,385  
5-Fold CV MAE: 17,126

Observation:
Single split was slightly optimistic.
Cross-validation gave more stable estimate.

---

## XGBoost (Engineered + Categorical Features)
Single Split MAE: 14,191  
5-Fold CV MAE: 14,882

Improvement over RandomForest:
~13% MAE reduction

---

## Final Model Choice
XGBoost selected due to significantly lower CV MAE.
