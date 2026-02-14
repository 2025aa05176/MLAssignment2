import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="ML Classification Dashboard", layout="wide")

st.markdown("<h1 style='text-align:center;'>📊 ML Classification Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------------
# Auto-train if saved_models folder does not exist
# ---------------------------------------------------

if not os.path.exists("saved_models/model_results.csv"):
    st.info("Training models for first time deployment... Please wait.")
    subprocess.run(["python", "model/train_models.py"])

# Now safely load files
results = pd.read_csv("saved_models/model_results.csv")
scaler = joblib.load("saved_models/scaler.pkl")

# Sidebar controls
st.sidebar.title("Controls")
model_name = st.sidebar.selectbox("Select Model", results["ML Model Name"].tolist())
uploaded_file = st.sidebar.file_uploader("Upload Test CSV (must contain 'target' column)", type=["csv"])

# Show model comparison table
st.subheader("Model Comparison Table")
st.dataframe(results, use_container_width=True)

# ---------------------------------------------------
# Prediction Section
# ---------------------------------------------------

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "target" not in df.columns:
        st.error("Uploaded CSV must contain a 'target' column.")
    else:
        model = joblib.load(f"saved_models/{model_name}.pkl")

        X = df.drop("target", axis=1)
        y_true = df["target"]

        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)

        st.markdown("---")
        st.success(f"Predictions using {model_name}")
        st.write(f"Current Model: **{model_name}**")

        col1, col2 = st.columns(2)

        # ---------------- Confusion Matrix ----------------
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            ax.set_title(f"{model_name} Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        # ---------------- Classification Report ----------------
        with col2:
            st.subheader("Classification Report")
            report_df = pd.DataFrame(
                classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            ).transpose()
            st.dataframe(report_df, use_container_width=True)

        # ---------------- Download Predictions ----------------
        st.markdown("---")
        result_df = df.copy()
        result_df["Predicted"] = y_pred

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
