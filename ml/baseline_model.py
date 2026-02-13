import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("data/raw/train.csv")

TARGET = "SalePrice"

y = df[TARGET]

X = df.drop(columns=[TARGET])

# Numeric columns only (base model with only numeric features)

X_num = X.select_dtypes(include=["number"])        

X_train, X_valid, y_train, y_valid = train_test_split(X_num, y, test_size=0.2, random_state=1)

model = RandomForestRegressor(n_estimators=100, random_state=1,n_jobs=-1)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", model),
])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_valid)

mae = mean_absolute_error(y_valid, preds)

print(f"Baseline MAE (with numeric features only): {mae:,.0f}")

