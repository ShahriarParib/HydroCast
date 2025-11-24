# app.py
"""
HydroCast — Viewer: show precomputed model predictions, comparisons and FI.
This lightweight app uses CSV artifacts stored in the repo:
 - mainset.csv
 - preds_lightgbm.csv, preds_xgboost.csv, preds_randomforest.csv
 - fi_lightgbm.csv, fi_xgboost.csv, fi_rf.csv
 - comparison_results.csv
This avoids heavy ML libs on the server and is safe for deployment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="HydroCast — Viewer", layout="wide")
st.title("HydroCast — Predictions & Comparison (Viewer)")

# ------------ Helper to safely read CSV if exists -------------
def load_csv(path):
    try:
        if path is None:
            return None
        if isinstance(path, (str,)):
            if os.path.exists(path):
                return pd.read_csv(path)
            else:
                return None
        return None
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        return None

# Sidebar: dataset source (repo or upload)
st.sidebar.header("Dataset & files")
use_repo = st.sidebar.checkbox("Use repo dataset (mainset.csv)", value=True)
uploaded_main = st.sidebar.file_uploader("Or upload your dataset (CSV) to override", type=["csv"])
if uploaded_main is not None:
    try:
        df_main = pd.read_csv(uploaded_main)
        st.sidebar.success("Using uploaded dataset")
    except Exception as e:
        st.sidebar.error(f"Uploaded dataset failed to read: {e}")
        df_main = None
elif use_repo:
    df_main = load_csv("mainset.csv")
    if df_main is None:
        st.sidebar.warning("mainset.csv not found in repo.")
else:
    df_main = None

# Load precomputed preds & FIs & comparison
pred_files = {
    "LightGBM": "preds_lightgbm.csv",
    "XGBoost": "preds_xgboost.csv",
    "RandomForest": "preds_randomforest.csv"
}
fi_files = {
    "LightGBM": "fi_lightgbm.csv",
    "XGBoost": "fi_xgboost.csv",
    "RandomForest": "fi_rf.csv"
}
preds = {}
for k,v in pred_files.items():
    df = load_csv(v)
    if df is not None:
        # ensure date column parsed
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        preds[k] = df

fis = {}
for k,v in fi_files.items():
    df = load_csv(v)
    if df is not None:
        fis[k] = df

comparison = load_csv("comparison_results.csv")
if comparison is None:
    st.sidebar.warning("comparison_results.csv not found — comparison table will be limited.")

# Show dataset preview
st.subheader("Dataset preview")
if df_main is not None:
    st.write("Columns:", df_main.columns.tolist())
    st.dataframe(df_main.head())
else:
    st.info("No dataset loaded. Upload a dataset or commit mainset.csv in the repo.")

# Left: comparison table + controls, Right: overlay
left, right = st.columns([1.2, 2])
with left:
    st.subheader("Model comparison")
    if comparison is not None:
        st.dataframe(comparison.reset_index(drop=True))
        st.download_button("Download comparison CSV", comparison.to_csv(index=False).encode('utf-8'),
                           "comparison_results.csv", "text/csv")
    else:
        st.info("No comparison CSV available.")

    st.markdown("---")
    st.subheader("Feature importances (bar)")
    model_for_fi = st.selectbox("Choose model to view FI", options=list(fis.keys()) if fis else [])
    if model_for_fi:
        fi = fis.get(model_for_fi)
        if fi is not None:
            # expected columns: feature, importance (if different, try to guess)
            if 'feature' not in fi.columns and fi.shape[1] >= 2:
                fi.columns = ['feature','importance'] + list(fi.columns[2:])
            fig_fi = px.bar(fi.sort_values('importance', ascending=True),
                            x='importance', y='feature', orientation='h',
                            title=f"{model_for_fi} feature importances")
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("No FI file found for this model.")

with right:
    st.subheader("Overlay: Actual vs Predictions")
    # pick a model to overlay or show all
    if preds:
        # find a 'true' series: check first preds table for y_true column
        sample = next(iter(preds.values()))
        if 'y_true' not in sample.columns and df_main is not None:
            # attempt to align: if main dataset has a power column, use it
            power_cols = [c for c in df_main.columns if 'power' in c.lower() or 'actual' in c.lower() or 'peak' in c.lower()]
            true_series = None
            true_dates = None
            if power_cols:
                true_col = power_cols[0]
                true_series = df_main[[true_col, df_main.columns[0]]].copy()
                true_series.columns = ['y_true','date']
                true_series['date'] = pd.to_datetime(true_series['date'], errors='coerce')
                true_series = true_series.sort_values('date')
                right.add_chart = True
            else:
                true_series = None
        else:
            true_series = sample[['date','y_true']].copy() if 'y_true' in sample.columns else None

        ov = go.Figure()
        if true_series is not None:
            ov.add_trace(go.Scatter(x=true_series['date'], y=true_series['y_true'], mode='lines', name='True', line=dict(color='black', dash='dash')))

        for m,dfm in preds.items():
            if dfm is None: continue
            if 'date' in dfm.columns and 'y_pred' in dfm.columns:
                ov.add_trace(go.Scatter(x=dfm['date'], y=dfm['y_pred'], mode='lines', name=f"Pred {m}", opacity=0.8))
        ov.update_layout(height=450, title="Overlay: Actual vs Predictions")
        st.plotly_chart(ov, use_container_width=True)
    else:
        st.info("No prediction CSVs found in repo.")

st.markdown("---")
st.subheader("Per-model Actual vs Pred (select model to view)")
model_select = st.selectbox("Model", options=list(preds.keys()) if preds else [])
if model_select:
    dfm = preds[model_select]
    if 'date' in dfm.columns and 'y_true' in dfm.columns and 'y_pred' in dfm.columns:
        fig = px.line(dfm.sort_values('date'), x='date', y=['y_true','y_pred'], labels={'value':'Generation (MW)','date':'Date'})
        fig.update_layout(title=f"{model_select}: Actual vs Pred")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Prediction CSV for this model must contain 'date','y_true','y_pred' columns.")

st.markdown("---")
st.subheader("Pairwise comparisons (RMSE differences, correlation)")
if comparison is not None:
    # try to show pairwise RMSE differences if present, else compute from preds if possible
    if 'rmse' in comparison.columns:
        st.write("Comparison table (selected metrics):")
        st.dataframe(comparison[['r2','rmse','mae']].round(4))
    # show correlation matrix if multiple preds present
    if len(preds) > 1:
        preds_df = {}
        for m,dfm in preds.items():
            if 'y_pred' in dfm.columns:
                preds_df[m] = dfm.sort_values('date')['y_pred'].values
        if preds_df:
            preds_df = pd.DataFrame(preds_df)
            corr = preds_df.corr()
            fig_corr = px.imshow(corr, text_auto=True, title="Prediction correlation matrix")
            st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")
st.subheader("Date-based lookup")
# allow the user to select a date and show predictions from CSVs + true value if available
min_date = None
max_date = None
if df_main is not None and df_main.shape[0] > 0:
    try:
        df_main_dates = pd.to_datetime(df_main.iloc[:,0], errors='coerce')
        min_date = df_main_dates.min()
        max_date = df_main_dates.max()
    except Exception:
        pass

sel_date = st.date_input("Choose date to lookup:", value=max_date if max_date is not None else pd.Timestamp.now().date(), min_value=min_date)
sel_date_ts = pd.to_datetime(sel_date)

st.write(f"Lookup for {sel_date_ts.date()}:")

rows = []
if df_main is not None:
    # try to find true value
    # heuristic: find a column that looks like power
    power_cols = [c for c in df_main.columns if 'power' in c.lower() or 'actual' in c.lower() or 'peak' in c.lower()]
    true_val = None
    if power_cols:
        tmp = df_main.copy()
        tmp.iloc[:,0] = pd.to_datetime(tmp.iloc[:,0], errors='coerce')
        sel_row = tmp[tmp.iloc[:,0] == sel_date_ts]
        if not sel_row.empty:
            true_val = float(sel_row[power_cols[0]].iloc[0])
    rows.append({"source":"True (dataset)","value":true_val})

for m,dfm in preds.items():
    if dfm is None: continue
    if 'date' in dfm.columns and 'y_pred' in dfm.columns:
        sel = dfm[pd.to_datetime(dfm['date']).dt.normalize() == sel_date_ts.normalize()]
        if not sel.empty:
            rows.append({"source":f"Pred {m}","value":float(sel['y_pred'].iloc[0])})
        else:
            rows.append({"source":f"Pred {m}","value":None})

res_df = pd.DataFrame(rows)
st.table(res_df)

st.markdown("---")
st.subheader("Downloads")
# combined preds download (if you want)
if preds:
    combined = None
    for m,dfm in preds.items():
        if dfm is None: continue
        tmp = dfm[['date','y_pred']].rename(columns={'y_pred':f'y_pred_{m}'})
        if combined is None:
            combined = tmp
        else:
            combined = combined.merge(tmp, on='date', how='outer')
    if combined is not None:
        st.download_button("Download combined predictions CSV", combined.to_csv(index=False).encode('utf-8'),
                           "combined_predictions.csv", "text/csv")

st.info("This viewer uses precomputed CSV files from the repo. If you want retraining/predicting on the server, ask me and I will add a training mode (B: retraining on server is slower and may fail due to package builds).")
