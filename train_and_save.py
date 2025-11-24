# train_and_save.py
import os
import time
import joblib
import json
import numpy as np
import pandas as pd

# ML libs
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import shap

# ---------------- Config ----------------
DATA_PATH = "mainset.csv"        # <-- YOUR DATASET IN SAME FOLDER
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- Helpers ----------------
def parse_date(df):
    for c in df.columns:
        if "date" in c.lower():
            df[c] = pd.to_datetime(df[c], errors="coerce")
            return df.rename(columns={c: "date"})
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    return df.rename(columns={df.columns[0]: "date"})

def feature_engineer(df, power_col="generation_mw"):
    df = df.copy()
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)

    df['power_lag1'] = df[power_col].shift(1)
    df['power_lag7'] = df[power_col].shift(7)
    df['power_7d_mean'] = df[power_col].rolling(7, min_periods=1).mean()
    df['power_30d_mean'] = df[power_col].rolling(30, min_periods=1).mean()

    # rainfall / meteo mappings
    if 'IMERG_PRECTOT' in df.columns:
        df['precipitation_mm'] = df['IMERG_PRECTOT']
        df['rain_lag1'] = df['precipitation_mm'].shift(1)
        df['rain_7d_mean'] = df['precipitation_mm'].rolling(7, min_periods=1).mean()

    if 'T2M' in df.columns:
        df['temperature_c'] = df['T2M']

    if 'ALLSKY_SFC_SW_DWN' in df.columns:
        df['shortwave_radiation'] = df['ALLSKY_SFC_SW_DWN']

    if 'precipitation_mm' in df.columns and 'temperature_c' in df.columns:
        df['rain_temp_int'] = df['precipitation_mm'] * df['temperature_c']

    if 'precipitation_mm' in df.columns and 'shortwave_radiation' in df.columns:
        df['rain_rad_int'] = df['precipitation_mm'] * df['shortwave_radiation']

    df = df.dropna(subset=[power_col, "power_lag1"]).reset_index(drop=True)
    return df

def compute_metrics(y, p):
    y = np.array(y)
    p = np.array(p)
    if len(y) == 0:
        return {}

    return {
        "n": len(y),
        "r2": float(r2_score(y, p)) if len(y) > 1 else np.nan,
        "rmse": float(np.sqrt(((y - p) ** 2).mean())),
        "mae": float(np.abs(y - p).mean())
    }

# ---------------- Load & Prepare Data ----------------
print("Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

df = parse_date(df)

# Your dataset's power column
power_col = "Actual Day Peak (MW)"
df["generation_mw"] = df[power_col]

df_fe = feature_engineer(df)

# feature list
candidate = [c for c in df_fe.columns if c not in ["date", "generation_mw"]]
features = df_fe[candidate].select_dtypes(include=[np.number]).columns.tolist()

# splits
n = len(df_fe)
test_ratio = 0.17
val_ratio = 0.21

n_test = int(n * test_ratio)
n_val = int(n * val_ratio)
n_train = n - n_val - n_test

train = df_fe.iloc[:n_train]
val = df_fe.iloc[n_train:n_train+n_val]
test = df_fe.iloc[n_train+n_val:]

X_train = train[features].astype(float).fillna(0)
y_train = train["generation_mw"].astype(float)

X_val = val[features].astype(float).fillna(0)
y_val = val["generation_mw"].astype(float)

X_test = test[features].astype(float).fillna(0)
y_test = test["generation_mw"].astype(float)

# ---------------- Train Models ----------------

results = {}
preds = {}

# ---- LightGBM ----
print("Training LightGBM...")
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbosity": -1,
}

dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val)

bst = lgb.train(
    params,
    dtrain,
    num_boost_round=500,
    valid_sets=[dtrain, dval],
    callbacks=[lgb.early_stopping(50)]
)

joblib.dump(bst, os.path.join(OUT_DIR, "model_lightgbm.pkl"))
pred_lgb = bst.predict(X_test)
results["LightGBM"] = compute_metrics(y_test, pred_lgb)

preds["LightGBM"] = pd.DataFrame({
    "date": test["date"],
    "y_true": y_test,
    "y_pred": pred_lgb
})

# ---- XGBoost ----
print("Training XGBoost...")
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params_xgb = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.05,
    "max_depth": 6
}

bst_xgb = xgb.train(
    params_xgb,
    dtrain,
    num_boost_round=500,
    evals=[(dval, "valid")],
    early_stopping_rounds=50,
    verbose_eval=False
)

joblib.dump(bst_xgb, os.path.join(OUT_DIR, "model_xgboost.pkl"))
pred_xgb = bst_xgb.predict(xgb.DMatrix(X_test))
results["XGBoost"] = compute_metrics(y_test, pred_xgb)

preds["XGBoost"] = pd.DataFrame({
    "date": test["date"],
    "y_true": y_test,
    "y_pred": pred_xgb
})

# ---- RandomForest ----
print("Training RandomForest...")
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

joblib.dump(rf, os.path.join(OUT_DIR, "model_rf.pkl"))
pred_rf = rf.predict(X_test)
results["RandomForest"] = compute_metrics(y_test, pred_rf)

preds["RandomForest"] = pd.DataFrame({
    "date": test["date"],
    "y_true": y_test,
    "y_pred": pred_rf
})

# ---------------- Save All Outputs ----------------
print("Saving predictions...")
for model_name, dfp in preds.items():
    dfp.to_csv(os.path.join(OUT_DIR, f"preds_{model_name.lower()}.csv"), index=False)

print("Saving feature importances...")
pd.DataFrame({
    "feature": features,
    "importance": bst.feature_importance(importance_type="gain")
}).to_csv(os.path.join(OUT_DIR, "fi_lightgbm.csv"), index=False)

xgb_score = bst_xgb.get_score(importance_type="gain")
pd.DataFrame({
    "feature": list(xgb_score.keys()),
    "importance": list(xgb_score.values())
}).to_csv(os.path.join(OUT_DIR, "fi_xgboost.csv"), index=False)

pd.DataFrame({
    "feature": features,
    "importance": rf.feature_importances_
}).to_csv(os.path.join(OUT_DIR, "fi_rf.csv"), index=False)

print("Saving comparison summary...")
pd.DataFrame(results).T.to_csv(os.path.join(OUT_DIR, "comparison_results.csv"))

print("DONE — All saved inside /models")
