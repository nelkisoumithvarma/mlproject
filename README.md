# House Prices – End-to-End Machine Learning Project

**Dataset:** Kaggle House Prices – Advanced Regression Techniques

This project builds a complete regression pipeline to predict house sale prices using structured tabular data from the Kaggle House Prices dataset.

The focus is not just on model accuracy, but on proper ML workflow:
- Baseline modeling
- Feature engineering
- Pipeline-based preprocessing
- Cross-validation
- Model comparison (Bagging vs Boosting)

---

## Project Objective

Develop a robust regression model while following production-ready best practices:

- Avoid data leakage using `Pipeline` + `ColumnTransformer`
- Validate performance using 5-fold cross-validation
- Compare models fairly under identical preprocessing
- Improve performance through feature engineering and boosting

---


## 🛠 Technologies & Libraries

- **Pandas** – Feature engineering, missing value handling, DataFrame operations  
- **Scikit-learn** – Pipeline, ColumnTransformer, RandomForestRegressor, cross-validation  
- **XGBoost** – Gradient boosting model for performance improvement  
- **Matplotlib** – Exploratory data visualization  
- **NumPy** – Numerical computations  


----



## Modeling Workflow

### 1️ Baseline Model — RandomForest (Numeric Features Only)
- Median imputation
- Simple train/validation split
- MAE: **17,024**

This establishes a clean performance benchmark.

---

### 2️ Feature Engineering + Full Pipeline — RandomForest
Added:
- HouseAge
- RemodelAge
- TotalSF
- TotalBathrooms
- Garage & Basement indicators
- Proper categorical encoding

5-Fold Cross-Validation MAE: **17,126**

Observation:
Single split improvement was slightly optimistic — CV provided stable evaluation.

---

### 3️ Boosting Upgrade — XGBoost
Same preprocessing pipeline  
Tuned hyperparameters:
- 500 estimators
- Learning rate = 0.05
- max_depth = 4
- Subsample & column sampling

5-Fold Cross-Validation MAE: **14,882**

📉 ~13% performance improvement over RandomForest.

---

## Final Results Summary

| Model |                     | CV MAE |
|-------|--------|
| RandomForest (numeric only) | 17,024 |
| RandomForest (engineered)   | 17,126 |
| XGBoost                     | **14,882** |

XGBoost selected as final model due to significantly lower cross-validated error.

---

## 🧠 Key Learnings

- Cross-validation prevents misleading conclusions
- Proper preprocessing pipelines are critical for reproducibility
- Feature engineering alone may not outperform strong tree ensembles
- Boosting significantly reduces bias on structured tabular data
- Controlled experimentation (changing one variable at a time) matters

---

## 🗂 Project Structure

data/ # Local dataset (not tracked in Git)
ml/ # Python training scripts
notebooks/ # Data exploration
results.md # Experiment log
README.md
.gitignore


---

## 🚀 How to Run

From project root:

```bash
python ml/baseline_model.py
python ml/train_with_features.py
python ml/train_with_xgboost.py
