import argparse
import os 
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
from joblib import dump 

def preprocessor(num_cols, cat_cols):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols)
    ])
    return preprocess  

def main(args):
    script_dir = Path(__file__).parent
    path = script_dir / args.data
    if not path.exists():
        raise FileNotFoundError(f"Data not found at {path}")
    
    df = pd.read_csv(path)

    y = df['price']
    X = df.drop('price', axis=1)

    # infer feature groups for ColumnTransformer
    num_cols, cat_cols = X.select_dtypes(include=['number']).columns, X.select_dtypes(include=['object']).columns

    preprocess = preprocessor(num_cols, cat_cols)

    # Baseline: Linear Regression
    
    baseline = Pipeline([
        ("preprocess", preprocess), 
        ("model", LinearRegression())
    ])

    rmse_baseline = -cross_val_score(
        baseline, X, y, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1
    ).mean()

    print(f"RMSE for baseline model: {rmse_baseline:.3f}")

    # Lasso and Ridge Regressor


    # Random Forest Regressor

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])
     
    param_grid = {
        "model__n_estimators": [300,600],
        "model__max_depth": [None, 12, 24],
        "model__min_samples_leaf": [1,3,5]
    }
    grid = GridSearchCV(
        pipe, param_grid=param_grid, cv=5, 
        scoring="neg_root_mean_squared_error", n_jobs=-1, verbose=1
    )
    grid.fit(X,y)
    rmse_tuned = -grid.best_score_

    print(f"Baseline LinearRegression RMSE: {rmse_baseline:.3f}")
    print(f"Best RandomForest RMSE (CV): {rmse_tuned:.3f}")
    print(f"Best params: {grid.best_params_}")

    # save the best
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / "house_price_pipeline.pkl"
    dump(grid.best_estimator_, out_path)
    print(f"Saved trained pipeline to: {out_path}")

    # Results
    result_md = Path(args.result)
    result_md.write_text(
        "| Model | RMSE (CV) |\n|---|---:|\n"
        f"| LinearRegression (baseline) | {rmse_baseline:.3f} |\n"
        f"| RandomForest (tuned) | {rmse_tuned:.3f} |\n"
    )
    print(f"Wrote result table to: {result_md}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="path to dataset")
    parser.add_argument("--model_dir", type=str, required=True, help="where to save model artifact")
    parser.add_argument("--result", type=str, required=True, help="where to save result.md")
    args = parser.parse_args()
    main(args)

