# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.base import BaseEstimator

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("Credit Card Fraud Detection")
st.markdown(
    "Upload transactions CSV or enter a single transaction. The app will use the saved model (`fraud_model.pkl`) "
    "and scaler (`scaler.pkl`) in the same folder to predict whether transactions are fraudulent."
)

# --- Helpers ---
@st.cache_resource
def load_model(model_path="fraud_model.pkl"):
    if not os.path.exists(model_path):
        st.error(f"Model file not found at `{model_path}`. Put `fraud_model.pkl` in the app folder.")
        return None
    model = joblib.load(model_path)
    return model

@st.cache_resource
def load_scaler(scaler_path="scaler.pkl"):
    if not os.path.exists(scaler_path):
        st.warning(f"Scaler not found at `{scaler_path}`. Predictions may fail if the model expects scaled input.")
        return None
    scaler = joblib.load(scaler_path)
    return scaler

def get_feature_names_from_model(model):
    # Try sklearn attribute first
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    # For pipelines, try to find by attribute
    if hasattr(model, "named_steps"):
        # try to locate an estimator inside pipeline
        for name, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def default_feature_list():
    # Common Kaggle credit card fraud dataset features
    # Time, V1..V28, Amount (target column is usually 'Class')
    feat = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    return feat

def prepare_dataframe_for_model(df, feature_names, scaler=None):
    # Ensure columns are present and in correct order
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required features: {missing}")
    X = df[feature_names].copy()
    # If scaler provided and appears appropriate, apply (scaler expects 2D numeric array)
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)
        except Exception:
            # fallback: assume scaler wasn't needed or incompatible
            pass
    return X

# --- Load model & scaler ---
with st.sidebar:
    st.header("Model settings")
    model_path = st.text_input("Model file path", value="fraud_model.pkl")
    scaler_path = st.text_input("Scaler file path (optional)", value="scaler.pkl")
    st.caption("If model or scaler are in a different folder, change the path here.")
    st.write("---")
    st.caption("If model was saved from a pipeline, `feature_names_in_` may be present and used automatically.")

model = load_model(model_path)
scaler = None
if os.path.exists(scaler_path):
    scaler = load_scaler(scaler_path)

if model is None:
    st.stop()

# determine expected features
feature_names = get_feature_names_from_model(model)
if feature_names is None:
    # fallback default features
    feature_names = default_feature_list()

# UI: either upload CSV or manual input
st.header("Input options")
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload a CSV with transactions (each row = 1 transaction)", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded CSV: {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None
    else:
        df = None

with col2:
    st.markdown("**Manual single transaction input**")
    st.info("If the model expects many features (e.g. V1..V28) this section will show them — it may be long.")
    # Build manual input form dynamically
    with st.form("manual_input"):
        manual_values = {}
        # If the feature list is large, place inside an expander
        with st.expander("Show / Edit input feature values", expanded=False):
            for feat in feature_names:
                # avoid adding 'Class' if it is present in feature_names by mistake
                if feat.lower() == "class":
                    continue
                # default to 0.0
                val = st.number_input(f"{feat}", value=0.0, format="%.6f", key=f"m_{feat}")
                manual_values[feat] = val
        submitted = st.form_submit_button("Predict single transaction")

# Prediction logic
def predict_df(X_df):
    try:
        preds = model.predict(X_df)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_df)
            # if binary classification, take prob of class 1
            if probs.shape[1] == 2:
                score = probs[:, 1]
            else:
                # fallback: take max probability
                score = probs.max(axis=1)
        else:
            # Some models only give decision_function
            if hasattr(model, "decision_function"):
                # convert to probability-like with sigmoid (not perfect)
                dfc = model.decision_function(X_df)
                score = 1 / (1 + np.exp(-dfc))
            else:
                score = np.zeros(len(preds))
        out_df = X_df.copy()
        out_df["prediction"] = preds
        out_df["fraud_score"] = score
        return out_df
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

results_df = None

# 1) If file was uploaded, use it
if df is not None:
    # If uploaded CSV columns do not match model features exactly, try inferring
    # Option A: if the uploaded df contains all required features -> proceed
    try:
        X = prepare_dataframe_for_model(df, feature_names, scaler=scaler)
        results_df = predict_df(X)
    except ValueError as ve:
        st.warning(str(ve))
        st.info("Attempting to infer features from uploaded CSV by matching common names...")
        # attempt to find subset intersection and use those columns if count matches
        common = [c for c in feature_names if c in df.columns]
        if len(common) >= max(3, int(0.6 * len(feature_names))):
            st.success(f"Found {len(common)} matching features; using those.")
            X = df[common].copy()
            # If scaler exists and feature count matches what scaler expects, apply.
            try:
                if scaler is not None:
                    X = pd.DataFrame(scaler.transform(X), columns=common, index=X.index)
            except Exception:
                pass
            results_df = predict_df(X)
        else:
            st.error(
                "Uploaded CSV does not contain enough required features to run the model. "
                "Consider uploading a CSV with the same columns used during training (e.g. Time, V1..V28, Amount)."
            )

# 2) If manual input submitted, build a single-row dataframe and predict
if 'submitted' in locals() and submitted:
    if len(manual_values) == 0:
        st.error("No manual inputs found.")
    else:
        single_df = pd.DataFrame([manual_values])
        # Ensure the single_df has columns in the model expected order
        # If model wanted extra features not provided, we will attempt to fill with zeros
        for f in feature_names:
            if f not in single_df.columns:
                single_df[f] = 0.0
        single_df = single_df[feature_names]
        try:
            X = prepare_dataframe_for_model(single_df, feature_names, scaler=scaler)
            results_df = predict_df(X)
        except Exception as e:
            st.error(f"Could not prepare the single input for prediction: {e}")
            st.stop()

# If we have results, show them
if results_df is not None:
    st.header("Predictions")
    # Show counts
    if "prediction" in results_df.columns:
        counts = results_df["prediction"].value_counts().rename_axis("prediction").reset_index(name="count")
        st.write("Prediction counts:")
        st.table(counts)
    st.dataframe(results_df.head(200))

    # Allow user to download results
    csv = results_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

    # Simple bar chart of fraud score distribution
    try:
        st.write("Fraud score distribution")
        st.bar_chart(results_df["fraud_score"].value_counts().sort_index())
    except Exception:
        pass

st.write("---")
st.markdown(
    "Notes:\n"
    "- If predictions look wrong, check that the uploaded CSV columns and order match what the model expects.\n"
    "- If the model was trained on scaled features, make sure `scaler.pkl` is provided and compatible.\n"
    "- Typical dataset features (Kaggle) are: `Time`, `V1`..`V28`, `Amount`. The target column is usually `Class`."
)
