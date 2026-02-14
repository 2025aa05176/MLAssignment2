import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

# Optional: xgboost may not be available in some environments
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


# ----------------------------
# Page config + lightweight UI
# ----------------------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.markdown(
    """
    <style>
      .main-title {font-size:34px; font-weight:800; margin-bottom: 0.25rem;}
      .subtle {color:#6b7280; margin-top:0;}
      .section-title {font-size:20px; font-weight:700; margin-top:18px;}
      .card {background:#ffffff; border:1px solid #e5e7eb; border-radius:12px; padding:14px;}
      .hint {color:#6b7280; font-size: 0.9rem;}
      .small {font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Machine Learning Classification — Assignment 2</div>', unsafe_allow_html=True)
st.markdown('<p class="subtle">UCI Breast Cancer Wisconsin (Diagnostic) dataset • 6 models • 6 metrics • CSV prediction download</p>', unsafe_allow_html=True)
st.markdown("---")


# ----------------------------
# Caching: data + trained models
# ----------------------------
@st.cache_data(show_spinner=False)
def load_split_data(test_size: float = 0.2, random_state: int = 42):
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X, y, X_train, X_test, y_train, y_test


def build_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbor": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
    }

    if _HAS_XGB:
        # Remove deprecated/unused params to avoid warnings and keep it fast.
        models["XGBoost (Ensemble)"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        )
    return models


def _safe_predict_proba(model, X_df: pd.DataFrame) -> np.ndarray:
    """
    Return probability for positive class if available.
    Falls back to decision_function -> sigmoid scaling if needed.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_df)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_df)
        # Min-max scale to [0,1] for AUC/presentation (not calibrated probability)
        s_min, s_max = float(np.min(scores)), float(np.max(scores))
        if s_max - s_min > 1e-9:
            return (scores - s_min) / (s_max - s_min)
        return np.zeros_like(scores, dtype=float)
    return np.zeros(X_df.shape[0], dtype=float)


@st.cache_resource(show_spinner=True)
def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """
    Train all models ONCE and compute their metrics ONCE.
    Cached across Streamlit reruns.
    """
    warnings.filterwarnings("ignore")

    models = build_models()

    rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model

        y_pred = model.predict(X_test)
        y_prob = _safe_predict_proba(model, X_test)

        rows.append(
            {
                "ML Model Name": name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "AUC": round(roc_auc_score(y_test, y_prob), 4),
                "Precision": round(precision_score(y_test, y_pred), 4),
                "Recall": round(recall_score(y_test, y_pred), 4),
                "F1 Score": round(f1_score(y_test, y_pred), 4),
                "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
            }
        )

    results_df = pd.DataFrame(rows)
    return fitted, results_df


# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Controls")

    X, y, X_train, X_test, y_train, y_test = load_split_data()

    fitted_models, results_df = train_and_evaluate_models(X_train, y_train, X_test, y_test)

    model_names = list(fitted_models.keys())
    selected_model_name = st.selectbox("Choose Model", model_names, index=0)

    st.markdown("---")
    st.subheader("Upload Test Dataset (CSV)")
    st.caption("Upload a CSV with the same 30 feature columns. If it includes 'target', it will be ignored.")
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    st.markdown("---")
    st.subheader("Download")
    st.caption("Download model comparison report as CSV.")
    report_csv = results_df.assign(Selected_Model=selected_model_name).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Model Performance Report (CSV)",
        data=report_csv,
        file_name="ML_Assignment_Report.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ----------------------------
# Main: model comparison table
# ----------------------------
st.markdown('<div class="section-title">Model Performance Comparison (All Models)</div>', unsafe_allow_html=True)

# Show results with minimal styling for speed and clarity
st.dataframe(results_df, use_container_width=True, height=260)

# ----------------------------
# Main: selected model details
# ----------------------------
selected_model = fitted_models[selected_model_name]
y_pred = selected_model.predict(X_test)
y_prob = _safe_predict_proba(selected_model, X_test)

acc = round(accuracy_score(y_test, y_pred), 4)
auc = round(roc_auc_score(y_test, y_prob), 4)
prec = round(precision_score(y_test, y_pred), 4)
rec = round(recall_score(y_test, y_pred), 4)
f1 = round(f1_score(y_test, y_pred), 4)
mcc = round(matthews_corrcoef(y_test, y_pred), 4)

st.markdown('<div class="section-title">Evaluation Metrics (Selected Model)</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", acc)
c2.metric("AUC", auc)
c3.metric("Precision", prec)

c4, c5, c6 = st.columns(3)
c4.metric("Recall", rec)
c5.metric("F1 Score", f1)
c6.metric("MCC", mcc)

# ----------------------------
# Confusion matrix (fast matplotlib, no seaborn dependency)
# ----------------------------
st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)

cm = confusion_matrix(y_test, y_pred)
fig_cm = plt.figure()
plt.imshow(cm,cmap="winter")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
for (i, j), val in np.ndenumerate(cm):
    plt.text(j, i, str(val), ha="center", va="center")
plt.xticks([0, 1], ["0", "1"])
plt.yticks([0, 1], ["0", "1"])
st.pyplot(fig_cm, use_container_width=False)
plt.close(fig_cm)

# ----------------------------
# Classification report (tabular)
# ----------------------------
st.markdown('<div class="section-title">Classification Report (Tabular)</div>', unsafe_allow_html=True)

report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
# Make support integer, keep others rounded
if "support" in report_df.columns:
    report_df["support"] = report_df["support"].round(0).astype(int)
for col in ["precision", "recall", "f1-score"]:
    if col in report_df.columns:
        report_df[col] = report_df[col].round(4)
st.dataframe(report_df.reset_index().rename(columns={"index": "Class"}), use_container_width=True)

# ----------------------------
# CSV prediction section (robust validation + fast)
# ----------------------------
st.markdown('<div class="section-title">Predictions on Uploaded Test Data</div>', unsafe_allow_html=True)

if uploaded_file is None:
    st.info("Upload a CSV file from the left sidebar to generate predictions.")
else:
    try:
        test_data = pd.read_csv(uploaded_file)

        # Drop target if present
        if "target" in test_data.columns:
            test_data = test_data.drop(columns=["target"])

        # Validate required columns
        required_cols = list(X.columns)
        missing = [c for c in required_cols if c not in test_data.columns]
        if missing:
            st.error(f"Uploaded file is missing required columns: {missing}")
        else:
            # Keep only required columns in correct order
            test_data = test_data[required_cols]

            # Convert to numeric safely
            test_data = test_data.apply(pd.to_numeric, errors="coerce")

            if test_data.isna().any().any():
                st.warning("Some values could not be converted to numeric and were set to NaN. Please review your CSV.")

            preds = selected_model.predict(test_data)
            probs = _safe_predict_proba(selected_model, test_data)

            out_df = test_data.copy()
            out_df["Predicted_Class"] = preds
            out_df["Score_Class_1"] = np.round(probs, 6)

            st.dataframe(out_df, use_container_width=True, height=360)

            pred_csv = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions as CSV",
                data=pred_csv,
                file_name="Predictions_Output.csv",
                mime="text/csv",
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"Could not read/process the uploaded CSV. Error: {e}")
