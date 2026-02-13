import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error




df = pd.read_csv("data/raw/train.csv")


TARGET = "SalePrice"

#Feature engineering 

# Calculate house age 
df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

# if Remodel age is same as house age then it is not remodeled
df["RemodelAge"] = df["YrSold"] - df["YearRemodAdd"]

# total house sqft
df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]


df["TotalBath"] = ( df["FullBath"] +  0.5 * df["HalfBath"] + df["BsmtFullBath"] +  0.5 * df["BsmtHalfBath"] )


df["HasGarage"] = (df["GarageArea"].fillna(0) > 0).astype(int)

df["HasBasement"] = (df["TotalBsmtSF"].fillna(0) > 0).astype(int)


y = df[TARGET]

X = df.drop(columns=[TARGET])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1)


numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

categorical_cols = X_train.select_dtypes(include=["object"]).columns


numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# replacing randomforestregressor with xgboostregressor
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)



pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])


pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_valid)

mae = mean_absolute_error(y_valid, preds)

print(f"MAE (with engineered + categorical features): {mae:,.0f}")


# let's comapre mae improvement with cross validation

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    pipeline,
    X,
    y,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

print("Cross validated MAE:", -cv_scores.mean())
