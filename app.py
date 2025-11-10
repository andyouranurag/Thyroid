# app.py — Updated for thyroid_model_training.py pipeline

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import os

# ------------------ Load Saved Artifacts ------------------
MODEL_DIR = "saved_models"

# Load models
lr_model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
svm_model = joblib.load(os.path.join(MODEL_DIR, "svm.pkl"))
rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))

# Load utilities
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
encoder = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
model_scores = joblib.load(os.path.join(MODEL_DIR, "model_scores.pkl"))
selected_features = joblib.load(os.path.join(MODEL_DIR, "selected_features.pkl"))

# Extract target encoder for decoding predictions
target_encoder = encoder.get("Response")

# ------------------ Determine Best Model ------------------
# Compare using recall_macro primarily (cost-sensitive goal)
best_model_name = max(model_scores, key=lambda m: model_scores[m]["recall_macro"])
best_model_score = model_scores[best_model_name]["recall_macro"]

# ------------------ Streamlit Setup ------------------
st.set_page_config(page_title="🩺 Thyroid Disease Classifier", layout="centered")

st.title("🩺 Thyroid Disease Classification Dashboard")
st.markdown("""
Predict **thyroid disease status** using multiple calibrated ML models:  
_Logistic Regression, SVM, Random Forest, and XGBoost._

**Objective:** Multi-class thyroid status classification  
**Priority:** Minimize false negatives (high recall)
""")

mode_choice = st.sidebar.radio(
    "Model Selection Mode",
    ("Automatic (Best Recall Model)", "Manual Selection")
)

if mode_choice == "Manual Selection":
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ("Logistic Regression", "SVM", "Random Forest", "XGBoost")
    )
else:
    model_choice = best_model_name
    st.sidebar.success(f"✅ Auto-selected: {best_model_name} (Recall: {best_model_score:.2%})")

# ------------------ Input Form ------------------
with st.form("thyroid_form"):
    st.subheader("Enter Patient Details 👇")

    # Build form dynamically based on selected features
    user_inputs = {}
    for feature in selected_features:
        if feature == "Age":
            user_inputs["Age"] = st.number_input("Age", min_value=0, max_value=120, step=1)
        elif feature in encoder:
            user_inputs[feature] = st.selectbox(feature, encoder[feature].classes_)
        else:
            user_inputs[feature] = st.text_input(feature)

    submit = st.form_submit_button("🔍 Predict Thyroid Status")

# ------------------ Prediction Section ------------------
if submit:
    # Convert to DataFrame
    input_data = pd.DataFrame([user_inputs])

    # Encode categorical columns
    for col in input_data.columns:
        if col in encoder and col != "Response" and col != "Age":
            input_data[col] = encoder[col].transform(input_data[col])

    # Scale numeric
    if "Age" in input_data.columns:
        input_data["Age"] = scaler.transform(input_data[["Age"]])

    # Model dictionary
    models = {
        "Logistic Regression": lr_model,
        "SVM": svm_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model
    }

    comparison_results = []
    for name, mdl in models.items():
        pred = mdl.predict(input_data)[0]
        prob = mdl.predict_proba(input_data)[0]
        conf = np.max(prob)
        decoded_pred = (
            target_encoder.inverse_transform([pred])[0]
            if target_encoder else str(pred)
        )
        comparison_results.append({
            "Model": name,
            "Prediction": decoded_pred,
            "Confidence (%)": conf * 100,
            "Accuracy (%)": model_scores[name]["accuracy"] * 100,
            "Recall (macro)": model_scores[name]["recall_macro"] * 100
        })

    comparison_df = pd.DataFrame(comparison_results)

    # Selected model
    selected_model = models[model_choice]
    selected_pred = selected_model.predict(input_data)[0]
    selected_conf = np.max(selected_model.predict_proba(input_data)[0])
    decoded_selected_pred = (
        target_encoder.inverse_transform([selected_pred])[0]
        if target_encoder else str(selected_pred)
    )

    # Display main prediction
    st.markdown(f"### 🧩 Predicted Thyroid Status: **{decoded_selected_pred}**")
    st.metric(label="Confidence", value=f"{selected_conf*100:.2f}%")
    st.progress(float(selected_conf))

    # Comparison table
    st.divider()
    st.subheader("📊 Model Comparison Summary")
    st.dataframe(comparison_df.set_index("Model"))

    # Visualization
    st.subheader("📈 Confidence Level Comparison")
    fig = px.bar(
        comparison_df,
        x="Model",
        y="Confidence (%)",
        color="Model",
        text=comparison_df["Confidence (%)"].apply(lambda x: f"{x:.2f}%"),
        title="Confidence Scores Across Models",
        labels={"Confidence (%)": "Confidence (%)"},
        height=400
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"🏆 Best Performing Model by Recall: **{best_model_name}** (Recall: {best_model_score:.2%})")

# ------------------ End of app.py ------------------
