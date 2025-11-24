# app.py
"""
HydroCast — Final app: Multi-model + Dashboard + Date-based prediction
Auto-loads dataset from repo (mainset.csv) or local session path (/mnt/data/mainset.csv).
Includes dataset download button in sidebar.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import warnings
warnings.filterwarnings("ignore")

# ML libs
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error
import scipy.stats as stats

# explainability / plotting
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os

# ---------- Config ----------
# Primary (deploy) path - repo root file that you upload to GitHub
DATA_PATH = "mainset.csv"

# Session/local path that existed during development in this environment
SESSION_PATH = "/mnt/data/mainset.csv"  # <-- original session file path (kept as fallback for local runs)

st.set_page_config(page_title="HydroCast — Final", layout="wide")

# ---------- Helper functions ----------
def parse_date_col(df):
    for c in df.columns:
        if c.lower() in ["date", "day", "timestamp", "datetime"]:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df = df.sort_values(c).reset_index(drop=True)
            return df, c
    # fallback: first column
    df.iloc[:,0] = pd.to_datetime(df.iloc[:,0], errors="coerce")
    df = df.sort_values(df.columns[0]).reset_index(drop=True)
    return df, df.columns[0]

def feature_engineer(df, power_col="generation_mw"):
    df = df.copy()
    if "date" not in df.columns:
        raise ValueError("Missing 'date' column for feature engineering.")
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['power_lag1'] = df[power_col].shift(1)
    df['power_lag7'] = df[power_col].shift(7)
    df['power_7d_mean'] = df[power_col].rolling(7, min_periods=1).mean()
    df['power_30d_mean'] = df[power_col].rolling(30, min_periods=1).mean()
    if 'precipitation_mm' in df.columns:
        df['rain_lag1'] = df['precipitation_mm'].shift(1)
        df['rain_7d_mean'] = df['precipitation_mm'].rolling(7, min_periods=1).mean()
    if 'precipitation_mm' in df.columns and 'temperature_c' in df.columns:
        df['rain_temp_int'] = df['precipitation_mm'] * df['temperature_c']
    if 'precipitation_mm' in df.columns and 'shortwave_radiation' in df.columns:
        df['rain_rad_int'] = df['precipitation_mm'] * df['shortwave_radiation']
    df = df.dropna(subset=[power_col, 'power_lag1']).reset_index(drop=True)
    return df

# training wrappers
def train_lightgbm(X_train, y_train, X_val=None, y_val=None, quick_mode=True):
    params = {'objective':'regression','metric':'rmse','verbosity':-1,'boosting_type':'gbdt',
              'learning_rate':0.05,'num_leaves':31,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'seed':42}
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_sets = [train_set]
    callbacks = [lgb.log_evaluation(period=100)]
    if X_val is not None and y_val is not None:
        valid_sets.append(lgb.Dataset(X_val, label=y_val))
        callbacks.append(lgb.early_stopping(stopping_rounds=50))
    rounds = 300 if quick_mode else 1000
    model = lgb.train(params, train_set, num_boost_round=rounds, valid_sets=valid_sets, callbacks=callbacks)
    return model

def train_xgboost(X_train, y_train, X_val=None, y_val=None, quick_mode=True):
    params = {'objective':'reg:squarederror','eval_metric':'rmse','verbosity':0,'eta':0.05,'max_depth':6,'subsample':0.8,'colsample_bytree':0.8,'seed':42}
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.tolist())
    evals = [(dtrain,'train')]
    if X_val is not None and y_val is not None:
        deval = xgb.DMatrix(X_val, label=y_val, feature_names=X_val.columns.tolist())
        evals.append((deval,'valid'))
    rounds = 300 if quick_mode else 1000
    bst = xgb.train(params, dtrain, num_boost_round=rounds, evals=evals, early_stopping_rounds=50, verbose_eval=False)
    return bst

def train_rf(X_train, y_train, quick_mode=True):
    n_estimators = 100 if quick_mode else 300
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

# metrics
def mape(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    mask = (y_true != 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

def smape(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom > 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100.0

def pct_within_threshold(y_true, y_pred, pct=0.1):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    rel_err = np.abs(y_pred[mask] - y_true[mask]) / (np.abs(y_true[mask]) + 1e-9)
    return (rel_err <= pct).mean() * 100.0

def compute_all_metrics(y_true, y_pred):
    yt = np.array(y_true); yp = np.array(y_pred)
    n = len(yt)
    metrics = {}
    metrics['n'] = n
    metrics['r2'] = np.nan if n < 2 else (r2_score(yt, yp) if not np.isclose(np.var(yt),0.0) else (1.0 if np.allclose(yt,yp) else np.nan))
    metrics['rmse'] = mean_squared_error(yt, yp, squared=False) if n>0 else np.nan
    metrics['mae'] = mean_absolute_error(yt, yp) if n>0 else np.nan
    metrics['medae'] = median_absolute_error(yt, yp) if n>0 else np.nan
    metrics['mape'] = mape(yt, yp)
    metrics['smape'] = smape(yt, yp)
    metrics['bias'] = np.mean(yp - yt) if n>0 else np.nan
    try:
        metrics['pearson_r'] = float(stats.pearsonr(yt, yp)[0]) if n>1 else np.nan
    except Exception:
        metrics['pearson_r'] = np.nan
    metrics['pct_within_5pct'] = pct_within_threshold(yt, yp, pct=0.05)
    metrics['pct_within_10pct'] = pct_within_threshold(yt, yp, pct=0.10)
    return metrics

def get_feature_importances(model, model_name, feature_names):
    if model_name == "LightGBM":
        imp = model.feature_importance(importance_type='gain')
        df = pd.DataFrame({'feature': feature_names, 'importance': imp})
    elif model_name == "XGBoost":
        try:
            score = model.get_score(importance_type='gain')
            df = pd.DataFrame({'feature': list(score.keys()), 'importance': list(score.values())})
            for f in feature_names:
                if f not in df['feature'].values:
                    df = df.append({'feature': f, 'importance': 0}, ignore_index=True)
        except Exception:
            df = pd.DataFrame({'feature': feature_names, 'importance': [0]*len(feature_names)})
    else:
        try:
            imp = model.feature_importances_
            df = pd.DataFrame({'feature': feature_names, 'importance': imp})
        except Exception:
            df = pd.DataFrame({'feature': feature_names, 'importance': [0]*len(feature_names)})
    df = df.groupby('feature', as_index=False).sum().sort_values('importance', ascending=False).reset_index(drop=True)
    return df

def normalize_series(s):
    if s.max() == s.min():
        return s*0.0
    return (s - s.min()) / (s.max() - s.min())

# ---------- UI ----------
st.title("HydroCast — Final Multi-model Dashboard")
st.write("Paper (local): /mnt/data/paper hydropower forecasting (final camera ready).doc")
# --------------------------
# LOAD DATA (auto + uploader)
# --------------------------
df = None
loaded_file_path = None

st.sidebar.header("Dataset source")
use_auto = st.sidebar.checkbox("Load dataset automatically from repo/session (recommended)", value=True)
st.sidebar.write("Session dataset path (dev): `/mnt/data/mainset.csv`")
st.sidebar.write("Repo dataset path (deploy): `mainset.csv`")

# 1) If user checked auto-load, try session -> repo
if use_auto:
    # prefer session path (development environment)
    if os.path.exists(SESSION_PATH):
        try:
            df = pd.read_csv(SESSION_PATH)
            loaded_file_path = SESSION_PATH
            st.sidebar.success(f"Loaded dataset from session: {SESSION_PATH}")
        except Exception as e:
            st.sidebar.warning(f"Session dataset found but failed to load: {e}")

    # then try repo path
    if df is None and os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            loaded_file_path = DATA_PATH
            st.sidebar.success(f"Loaded dataset from repo: {DATA_PATH}")
        except Exception as e:
            st.sidebar.warning(f"Repo dataset found but failed to load: {e}")

# 2) Always show uploader so users can upload their own file if they want
uploaded_file = st.sidebar.file_uploader("Or upload your CSV (optional) — will override auto-load if provided", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        loaded_file_path = "uploaded"
        st.sidebar.success("Loaded dataset from uploaded file")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded file: {e}")

# 3) If still no df, show info and stop
if df is None:
    st.info("No dataset loaded. Either enable auto-load and include `mainset.csv` in repo root (or keep session file `/mnt/data/mainset.csv`), or upload a CSV using the sidebar uploader.")
    st.stop()

# 4) Provide download button for the dataset actually used
if loaded_file_path and loaded_file_path not in ["uploaded"]:
    try:
        fobj = open(loaded_file_path, "rb")
        st.sidebar.download_button("📥 Download dataset (current file)", fobj, file_name="mainset.csv", mime="text/csv")
        fobj.close()
    except Exception:
        pass
else:
    # if uploaded, let user download the in-memory df
    try:
        buf = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("📥 Download dataset (uploaded)", buf, file_name="mainset.csv", mime="text/csv")
    except Exception:
        pass

# 5) Show a small preview
st.subheader("Raw data preview")
st.write("Columns detected:", df.columns.tolist())
st.dataframe(df.head())


# ---------- map columns ----------
st.sidebar.header("Data source & mapping (leave defaults if your file matches)")

power_col = st.sidebar.text_input("Power (MW) column:", "Actual Day Peak (MW)")
precip_col = st.sidebar.text_input("Precipitation column:", "IMERG_PRECTOT")
temp_col = st.sidebar.text_input("Temperature column:", "T2M")
rad_col = st.sidebar.text_input("Shortwave radiation column:", "ALLSKY_SFC_SW_DWN")

st.sidebar.header("Model selection")
mode = st.sidebar.radio("Mode", ("Single model","Compare models"))
models_available = ["LightGBM","XGBoost","RandomForest"]
if mode == "Single model":
    model_choice = st.sidebar.selectbox("Select model", models_available)
    selected_models = [model_choice]
else:
    selected_models = st.sidebar.multiselect("Select models to compare", models_available, default=models_available)

st.sidebar.header("Training & display options")
quick_mode = st.sidebar.checkbox("Quick mode (faster training)", value=True)
shap_sample = st.sidebar.slider("SHAP sample size (max 200)", 50, 200, 200)
normalize_imp = st.sidebar.checkbox("Normalize feature importances (0-1)", value=True)
test_ratio = st.sidebar.slider("Test ratio", 0.05, 0.3, 0.17)
val_ratio = st.sidebar.slider("Validation ratio", 0.05, 0.3, 0.21)
save_models = st.sidebar.checkbox("Save trained models (.pkl)", value=True)
model_prefix = st.sidebar.text_input("Model filename prefix", "model")
sort_by = st.sidebar.selectbox("Comparison sort by", ("r2","rmse","mae","mape"))

# ---------- prepare dataframe ----------
# rename selected columns to internal names (non-destructive)
col_map = {}
if power_col in df.columns: col_map[power_col] = "generation_mw"
if precip_col in df.columns: col_map[precip_col] = "precipitation_mm"
if temp_col in df.columns: col_map[temp_col] = "temperature_c"
if rad_col in df.columns: col_map[rad_col] = "shortwave_radiation"

df_proc = df.rename(columns=col_map).copy()
df_proc, detected_date = parse_date_col(df_proc)
if detected_date != "date":
    df_proc = df_proc.rename(columns={detected_date: "date"})

st.info("Running feature engineering (internal copy)...")
try:
    df_fe = feature_engineer(df_proc, power_col="generation_mw")
except Exception as e:
    st.error(f"Feature engineering error: {e}")
    st.stop()

st.subheader("Prepared data (after FE)")
st.dataframe(df_fe.head())

# ---------- Features ----------
candidate = [c for c in df_fe.columns if c not in ["date","generation_mw"]]
numeric_cols = df_fe[candidate].select_dtypes(include=[np.number]).columns.tolist()
non_numeric = [c for c in candidate if c not in numeric_cols]
encoded = []
for c in non_numeric:
    nunique = df_fe[c].nunique(dropna=True)
    if nunique <= 20:
        vals = sorted(df_fe[c].dropna().unique())
        mapping = {v:i for i,v in enumerate(vals)}
        df_fe[c + "_enc"] = df_fe[c].map(mapping).astype(float)
        encoded.append(c + "_enc")
features = numeric_cols + encoded
features = [f for f in features if f in df_fe.columns]
st.write("Final features used:", features)

# ---------- Split ----------
n = len(df_fe)
n_test = int(n * test_ratio)
n_val = int(n * val_ratio)
n_train = n - n_val - n_test
st.write(f"Dataset length: {n} -> Train/Val/Test = {n_train}/{n_val}/{n_test}")
train_df = df_fe.iloc[:n_train].reset_index(drop=True)
val_df = df_fe.iloc[n_train:n_train+n_val].reset_index(drop=True)
test_df = df_fe.iloc[n_train+n_val:].reset_index(drop=True)

# ---------- Train & Compare ----------
if st.button("Train selected model(s)"):
    if len(features) == 0:
        st.error("No features available.")
        st.stop()

    all_results = []
    all_preds = {}
    trained_models = {}

    for m in selected_models:
        st.subheader(f"Training: {m}")
        X_train = train_df[features].copy()
        X_val = val_df[features].copy()
        X_test = test_df[features].copy()
        y_train = train_df["generation_mw"].copy()
        y_val = val_df["generation_mw"].copy()
        y_test = test_df["generation_mw"].copy()

        # numeric coercion and imputation
        X_train = X_train.apply(pd.to_numeric, errors='coerce')
        X_val = X_val.apply(pd.to_numeric, errors='coerce')
        X_test = X_test.apply(pd.to_numeric, errors='coerce')
        for col in X_train.columns:
            mv = X_train[col].mean()
            X_train[col].fillna(mv, inplace=True)
            if col in X_val.columns: X_val[col].fillna(mv, inplace=True)
            if col in X_test.columns: X_test[col].fillna(mv, inplace=True)

        # drop constant columns
        const_cols = [c for c in X_train.columns if X_train[c].nunique() <= 1]
        if const_cols:
            X_train.drop(columns=const_cols, inplace=True)
            X_val.drop(columns=[c for c in const_cols if c in X_val.columns], inplace=True)
            X_test.drop(columns=[c for c in const_cols if c in X_test.columns], inplace=True)

        # align columns
        common_cols = [c for c in X_train.columns if c in X_val.columns and c in X_test.columns]
        X_train = X_train[common_cols]; X_val = X_val[common_cols]; X_test = X_test[common_cols]

        start = time.time()
        try:
            if m == "LightGBM":
                model = train_lightgbm(X_train, y_train, X_val, y_val, quick_mode)
            elif m == "XGBoost":
                model = train_xgboost(X_train, y_train, X_val, y_val, quick_mode)
            elif m == "RandomForest":
                model = train_rf(X_train, y_train, quick_mode)
            else:
                st.error("Unknown model")
                continue
        except Exception as e:
            st.error(f"Training {m} failed: {e}")
            continue
        elapsed = time.time() - start

        if m == "XGBoost":
            preds = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns.tolist()), ntree_limit=getattr(model,'best_ntree_limit',None))
        elif m == "LightGBM":
            preds = model.predict(X_test, num_iteration=getattr(model,'best_iteration',None))
        else:
            preds = model.predict(X_test)

        metrics = compute_all_metrics(y_test.values, preds)
        st.write(f"{m} metrics: R2={metrics['r2']:.4f} RMSE={metrics['rmse']:.3f} MAE={metrics['mae']:.3f} MAPE={metrics['mape']:.2f}% Bias={metrics['bias']:.3f}")

        rec = {'model': m, **metrics, 'time_s': elapsed}
        all_results.append(rec)
        all_preds[m] = pd.DataFrame({'date': test_df['date'].values, 'y_true': y_test.values, 'y_pred': preds})
        trained_models[m] = {'model': model, 'cols': X_train.columns.tolist()}

        if save_models:
            fname = f"{m}_{model_prefix}_{int(time.time())}.pkl"
            try:
                joblib.dump(model, fname)
                trained_models[m]['saved_name'] = fname
                st.write(f"Saved {m} -> {fname}")
            except Exception as e:
                st.write(f"Failed to save {m}: {e}")

    # build comparison & dashboard
    if len(all_results) > 0:
        comp = pd.DataFrame(all_results).set_index('model')
        # ensure columns exist
        for col in ['r2','rmse','mae','mape','smape','bias','pearson_r','medae','pct_within_5pct','pct_within_10pct','time_s','n']:
            if col not in comp.columns:
                comp[col] = np.nan

        # relative improvements
        best_rmse = comp['rmse'].min()
        best_mae = comp['mae'].min()
        comp['rel_impr_rmse_pct'] = comp['rmse'].apply(lambda x: 100.0*(best_rmse - x)/best_rmse if best_rmse>0 else np.nan)
        comp['rel_impr_mae_pct'] = comp['mae'].apply(lambda x: 100.0*(best_mae - x)/best_mae if best_mae>0 else np.nan)

        # ranks
        comp['rank_r2'] = comp['r2'].rank(ascending=False, method='min')
        comp['rank_rmse'] = comp['rmse'].rank(ascending=True, method='min')
        comp['rank_mae'] = comp['mae'].rank(ascending=True, method='min')
        comp['rank_mape'] = comp['mape'].rank(ascending=True, method='min')

        # left: table+radar, right: overlay plot
        left_col, right_col = st.columns([1,2])
        with left_col:
            st.subheader("Comparison table")
            sort_asc = False if sort_by == 'r2' else True
            st.dataframe(comp.sort_values(by=sort_by, ascending=sort_asc).reset_index())

            st.subheader("Rank radar")
            rank_cols = ['rank_r2','rank_rmse','rank_mae','rank_mape']
            radar_df = comp[rank_cols].copy()
            radar_norm = radar_df.apply(lambda c: normalize_series(c), axis=0)
            radar_norm['model'] = radar_norm.index
            categories = list(radar_norm.columns[:-1])
            fig_radar = go.Figure()
            for idx, row in radar_norm.iterrows():
                fig_radar.add_trace(go.Scatterpolar(r=np.append(row[categories].values, row[categories].values[0]),
                                                   theta=categories + [categories[0]],
                                                   name=idx, fill='toself'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), height=400)
            st.plotly_chart(fig_radar, use_container_width=True)

            st.subheader("Relative improvements")
            st.write(comp[['rmse','rel_impr_rmse_pct','mae','rel_impr_mae_pct']].round(3))

        with right_col:
            st.subheader("Actual vs Predicted — overlay")
            ovfig = go.Figure()
            any_model = next(iter(all_preds))
            ovfig.add_trace(go.Scatter(x=all_preds[any_model]['date'], y=all_preds[any_model]['y_true'], mode='lines', name='True', line=dict(color='black', dash='dash')))
            for m in all_preds:
                dfm = all_preds[m]
                ovfig.add_trace(go.Scatter(x=dfm['date'], y=dfm['y_pred'], mode='lines', name=f"Pred {m}", opacity=0.8))
            ovfig.update_layout(title="Overlay: Actual vs Predictions", height=450)
            st.plotly_chart(ovfig, use_container_width=True)

        # compact grid of per-model plots
        st.subheader("Per-model Actual vs Pred (grid)")
        models = list(all_preds.keys())
        cols_per_row = 2
        rows = (len(models) + cols_per_row - 1) // cols_per_row
        for r in range(rows):
            cols = st.columns(cols_per_row)
            for cidx in range(cols_per_row):
                midx = r*cols_per_row + cidx
                if midx < len(models):
                    m = models[midx]
                    with cols[cidx]:
                        st.write(f"Model: {m}")
                        dfm = all_preds[m]
                        fig = px.line(dfm, x='date', y=['y_true','y_pred'], labels={'value':'Generation (MW)','date':'Date'})
                        fig.update_layout(title=f"{m}: Actual vs Pred")
                        st.plotly_chart(fig, use_container_width=True)

        # pairwise RMSE differences + correlation side-by-side
        st.subheader("Pairwise comparisons")
        hm1, hm2 = st.columns(2)
        with hm1:
            st.write("Pairwise RMSE differences (rows - cols)")
            rmse_map = comp['rmse'].to_dict()
            models_list = list(rmse_map.keys())
            pair_df = pd.DataFrame(index=models_list, columns=models_list, dtype=float)
            for a in models_list:
                for b in models_list:
                    pair_df.loc[a,b] = rmse_map[a] - rmse_map[b]
            fig_heat = px.imshow(pair_df, text_auto=True, color_continuous_scale='RdBu', title="RMSE differences")
            st.plotly_chart(fig_heat, use_container_width=True)
        with hm2:
            st.write("Pairwise prediction Pearson correlation")
            preds_df = pd.DataFrame({m: all_preds[m]['y_pred'].values for m in all_preds})
            corr = preds_df.corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis', title="Predictions correlation")
            st.plotly_chart(fig_corr, use_container_width=True)

        # Feature importance & SHAP in tabs
        st.subheader("Feature importance & SHAP (per model)")
        tabs = st.tabs(list(trained_models.keys()))
        for i,m in enumerate(trained_models.keys()):
            with tabs[i]:
                model = trained_models[m]['model']
                cols = trained_models[m]['cols']
                imp_df = get_feature_importances(model, m, cols)
                if imp_df['importance'].sum() == 0:
                    st.write("No feature importance values available.")
                else:
                    if normalize_imp:
                        imp_df['importance_norm'] = imp_df['importance'] / (imp_df['importance'].max() + 1e-9)
                        plot_col = 'importance_norm'
                    else:
                        plot_col = 'importance'
                    fig_imp = px.bar(imp_df.sort_values(plot_col, ascending=False).head(40), x=plot_col, y='feature', orientation='h',
                                     title=f"{m} feature importance", labels={plot_col:'importance','feature':'feature'})
                    st.plotly_chart(fig_imp, use_container_width=True)
                st.write("SHAP (sampled):")
                try:
                    Xs = test_df[cols].apply(pd.to_numeric, errors='coerce').fillna(method='ffill').fillna(0)
                    sampleX = Xs.sample(n=min(len(Xs), shap_sample), random_state=42)
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(sampleX)
                    plt.figure(figsize=(8,4))
                    shap.summary_plot(shap_values, sampleX, show=False)
                    st.pyplot(plt.gcf())
                    plt.clf()
                except Exception as e:
                    st.write(f"SHAP failed for {m}: {e}")

        # combined predictions download
        combined = None
        for m in all_preds:
            dfm = all_preds[m].rename(columns={'y_pred': f'y_pred_{m}'})
            if combined is None:
                combined = dfm
            else:
                combined = combined.merge(dfm[['date', f'y_pred_{m}']], on='date', how='left')
        if combined is not None:
            buf = combined.to_csv(index=False).encode('utf-8')
            st.download_button("Download combined predictions (CSV)", buf, "combined_predictions.csv", "text/csv")

        # store session
        st.session_state['trained_models'] = trained_models
        st.session_state['comparison'] = comp

# ---------- Quick predict (upload model) ----------
st.header("Quick predict (upload model)")
uploaded_model = st.file_uploader("Upload trained model (.pkl/.joblib) for quick predict", type=["pkl","joblib"])
if uploaded_model is not None:
    try:
        loaded = joblib.load(uploaded_model)
        latest = df_fe.iloc[[-1]][features].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
        if isinstance(loaded, xgb.Booster):
            pred_latest = loaded.predict(xgb.DMatrix(latest, feature_names=latest.columns.tolist()))
        else:
            pred_latest = loaded.predict(latest)
        st.write("Latest date:", df_fe.iloc[-1]['date'])
        st.write("Predicted generation (MW):", float(pred_latest[0]))
    except Exception as e:
        st.error(f"Quick predict failed: {e}")

# ---------- Date-based prediction panel ----------
st.header("Predict for a specific date")
model_source = st.radio("Model source for date prediction:", ("Use model trained in this session", "Upload a saved model (.pkl/.joblib)"))
selected_model_key = None
loaded_model = None
model_cols = features  # default

if model_source == "Use model trained in this session":
    trained = st.session_state.get('trained_models', {})
    if not trained:
        st.info("No models trained in this session. Train models first or choose 'Upload a saved model'.")
    else:
        selected_model_key = st.selectbox("Select a model from this session:", list(trained.keys()))
        if selected_model_key:
            loaded_model = trained[selected_model_key]['model']
            model_cols = trained[selected_model_key]['cols']
else:
    uploaded_model_for_date = st.file_uploader("Upload .pkl/.joblib model to use for date prediction", type=["pkl","joblib"], key="datepred_upload")
    if uploaded_model_for_date is not None:
        try:
            loaded_model = joblib.load(uploaded_model_for_date)
            st.success("Model loaded for date prediction.")
            model_cols = features
        except Exception as e:
            st.error(f"Failed to load uploaded model: {e}")
            loaded_model = None
            model_cols = features

min_date = df_fe['date'].min() if 'df_fe' in globals() else None
max_date = df_fe['date'].max() if 'df_fe' in globals() else None
default_date = (max_date + pd.Timedelta(days=1)) if max_date is not None else pd.Timestamp.today()
sel_date = st.date_input("Select date to predict for:", value=default_date, min_value=min_date, max_value=None)
predict_mode = st.radio("Prediction mode:", ("Use row from dataset (if exists)", "Construct features from last-known values (simple future forecast)"))
predict_button = st.button("Predict for selected date")

if predict_button:
    if loaded_model is None:
        st.error("No model available. Train a model first or upload one.")
    else:
        sel_date_ts = pd.to_datetime(sel_date)
        row_exists = sel_date_ts in pd.to_datetime(df_fe['date']).values

        if predict_mode == "Use row from dataset (if exists)":
            if not row_exists:
                st.error("Selected date not in processed data. Choose heuristic mode for future date.")
            else:
                row = df_fe[df_fe['date'] == sel_date_ts].iloc[0]
                X_row = row[model_cols].to_frame().T
                X_row = X_row.apply(pd.to_numeric, errors='coerce').fillna(0)
                try:
                    if isinstance(loaded_model, xgb.Booster):
                        pred = loaded_model.predict(xgb.DMatrix(X_row, feature_names=X_row.columns.tolist()))[0]
                    else:
                        pred = loaded_model.predict(X_row)[0]
                    st.success(f"Prediction for {sel_date_ts.date()}: **{pred:.3f} MW** (using dataset row)")
                    st.write("Features used (from dataset row):")
                    st.dataframe(X_row.T.rename(columns={0:"value"}))
                    csv_buf = X_row.to_csv(index=False).encode('utf-8')
                    st.download_button("Download features CSV", csv_buf, f"features_{sel_date_ts.date()}.csv", "text/csv")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        else:
            last_row = df_fe.iloc[-1].copy()
            new_row = last_row.copy()
            new_row['date'] = sel_date_ts
            new_row['month'] = sel_date_ts.month
            new_row['month_sin'] = np.sin(2 * np.pi * new_row['month'] / 12)
            new_row['month_cos'] = np.cos(2 * np.pi * new_row['month'] / 12)
            if 'generation_mw' in last_row.index:
                new_row['power_lag1'] = last_row['generation_mw']
            if 'power_lag7' in last_row.index:
                new_row['power_lag7'] = last_row['power_lag7']
            for col in ['precipitation_mm','temperature_c','shortwave_radiation']:
                if col in last_row.index:
                    new_row[col] = last_row[col]
            for col in ['power_7d_mean','power_30d_mean','rain_7d_mean']:
                if col in last_row.index:
                    new_row[col] = last_row[col]
            X_row = pd.DataFrame([new_row[model_cols].values], columns=model_cols)
            X_row = X_row.apply(pd.to_numeric, errors='coerce').fillna(0)
            try:
                if isinstance(loaded_model, xgb.Booster):
                    pred = loaded_model.predict(xgb.DMatrix(X_row, feature_names=X_row.columns.tolist()))[0]
                else:
                    pred = loaded_model.predict(X_row)[0]
                st.success(f"Forecast for {sel_date_ts.date()} (heuristic): **{pred:.3f} MW**")
                st.write("Heuristic features used (based on last-known row):")
                st.dataframe(X_row.T.rename(columns={0:"value"}))
                csv_buf = X_row.to_csv(index=False).encode('utf-8')
                st.download_button("Download features CSV", csv_buf, f"features_{sel_date_ts.date()}.csv", "text/csv")
                st.info("Note: heuristic future predictions reuse last-known inputs; for robust multi-day forecasts create a dedicated forecasting pipeline.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------- show session trained models ----------
if 'trained_models' in st.session_state:
    st.subheader("Session trained models")
    for k,v in st.session_state['trained_models'].items():
        st.write(f"{k} -> saved_name: {v.get('saved_name','(not saved)')}")

# End of app
