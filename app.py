import streamlit as st
import pandas as pd
import numpy as np
import io

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.markdown(
    """
    <style>
    .main-title {font-size:32px; font-weight:700;}
    .section-title {font-size:22px; font-weight:600; margin-top:20px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">Machine Learning Classification - Assignment 2</div>', unsafe_allow_html=True)
st.markdown("---")

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbor": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(),
    "XGBoost (Ensemble)": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("Upload Test Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (same feature structure as training data)",
    type=["csv"]
)

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_all = model.predict(X_test)
    y_prob_all = model.predict_proba(X_test)[:, 1]

    results.append({
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred_all), 4),
        "AUC": round(roc_auc_score(y_test, y_prob_all), 4),
        "Precision": round(precision_score(y_test, y_pred_all), 4),
        "Recall": round(recall_score(y_test, y_pred_all), 4),
        "F1 Score": round(f1_score(y_test, y_pred_all), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred_all), 4)
    })

results_df = pd.DataFrame(results)

st.markdown('<div class="section-title">Model Performance Comparison (All 6 Models)</div>', unsafe_allow_html=True)
st.dataframe(results)

st.markdown("---")

selected_model = models[selected_model_name]
selected_model.fit(X_train, y_train)

y_pred = selected_model.predict(X_test)
y_prob = selected_model.predict_proba(X_test)[:, 1]

st.markdown('<div class="section-title">Evaluation Metrics (Selected Model)</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 4))
col2.metric("AUC Score", round(roc_auc_score(y_test, y_prob), 4))
col3.metric("Precision", round(precision_score(y_test, y_pred), 4))

col4, col5, col6 = st.columns(3)
col4.metric("Recall", round(recall_score(y_test, y_pred), 4))
col5.metric("F1 Score", round(f1_score(y_test, y_pred), 4))
col6.metric("MCC Score", round(matthews_corrcoef(y_test, y_pred), 4))

st.markdown("---")

st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

st.markdown("---")

st.markdown('<div class="section-title">Classification Report</div>', unsafe_allow_html=True)

report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose().round(4)

st.dataframe(report_df)

st.markdown("---")

if uploaded_file:
    st.markdown('<div class="section-title">Predictions on Uploaded Test Data</div>', unsafe_allow_html=True)

    test_data = pd.read_csv(uploaded_file)

    if "target" in test_data.columns:
        test_data = test_data.drop(columns=["target"])

    test_data = test_data[X.columns]

    predictions = selected_model.predict(test_data)
    probabilities = selected_model.predict_proba(test_data)[:, 1]

    prediction_df = test_data.copy()
    prediction_df["Predicted_Class"] = predictions
    prediction_df["Probability_Class_1"] = probabilities

    st.dataframe(prediction_df)

st.markdown("---")

st.markdown('<div class="section-title">Download Complete Report (CSV)</div>', unsafe_allow_html=True)

full_report_df = results_df.copy()
full_report_df["Selected_Model"] = selected_model_name

csv_buffer = full_report_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Model Performance Report (CSV)",
    data=csv_buffer,
    file_name="ML_Assignment_Report.csv",
    mime="text/csv"
)
