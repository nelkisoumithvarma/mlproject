# House Prices – Machine Learning Mini Project with feature engineering 

This project predicts house sale prices using tabular machine learning techniques.

## Project Goal
Build a strong regression model using:
- Baseline modeling
- Feature engineering
- Proper preprocessing (Pipeline + ColumnTransformer)
- Cross-validation
- Model comparison

## Models Used

### 1. Baseline – RandomForestRegrerssor (Numeric features only)
- Median imputation
- MAE: 17,024

### 2. RandomForestRegressor (Engineered + Categorical features)
- Full preprocessing pipeline
- 5-Fold CV MAE: 17,126

### 3. XGBoost (Engineered + Categorical features)
- Boosting with tuned parameters
- 5-Fold CV MAE: 14,882

## Key Learnings
- Proper preprocessing is critical
- Cross-validation prevents misleading results
- Boosting significantly improved performance (~13%)

## Project Structure

data/
ml/
notebooks/
results.md
README.md
.gitignore