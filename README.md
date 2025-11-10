# 🩺 Thyroid Disease Classification & Prediction App

A **machine learning–based diagnostic tool** that predicts **thyroid disease status** using multiple models — Logistic Regression, SVM, Random Forest, and XGBoost — with a **cost-sensitive focus** on minimizing false negatives.

Built and deployed using **Streamlit**.

---

## 🎯 **Project Objectives**

1️⃣ **Multi-Class Thyroid Status Classification**  
   Predicts thyroid health categories (e.g., Normal, Hypothyroid, Hyperthyroid, etc.) from patient data.

2️⃣ **Cost-Sensitive Optimization**  
   Models are tuned to **minimize false negatives**, ensuring that patients with thyroid conditions are not misclassified as normal.

3️⃣ **Model Comparison Interface**  
   The Streamlit dashboard allows both **manual** and **automatic** model selection and displays model-wise confidence and accuracy.

---

## 📊 **Dataset**

**Source:** [Thyroid Disease Dataset – Kaggle](https://www.kaggle.com/datasets)  
**Key Features:**
| Feature | Description |
|----------|-------------|
| Age | Patient's age |
| Gender | Male / Female |
| Smoking / Hx Smoking | Current or past smoking history |
| Hx Radiotherapy | Prior radiotherapy treatment |
| Thyroid Function | Measured thyroid functionality |
| Physical Examination | Clinical thyroid examination results |
| Adenopathy | Lymph node abnormality presence |
| Pathology | Histopathology test results |
| Focality | Single or multiple lesion presence |
| Risk | Clinical risk assessment |
| T, N, M, Stage | Tumor staging features |
| Response | Target variable (Thyroid Status) |

---

## 🧠 **Algorithms Used**

| Algorithm | Type | Notes |
|------------|------|-------|
| **Logistic Regression** | Linear Model | Baseline probabilistic model |
| **SVM (Support Vector Machine)** | Kernel-based | Strong performance on smaller, complex datasets |
| **Random Forest** | Ensemble (Bagging) | Handles non-linear data well |
| **XGBoost** | Ensemble (Boosting) | Best performance & low bias |

All models were trained with calibration to make probability outputs consistent.

---

## 🧩 **Cost-Sensitive Design**

Special focus was given to **minimize False Negatives** — because misclassifying a patient with thyroid disease as “normal” is clinically serious.

This was achieved by:
- Emphasizing **Recall** during model tuning
- Using **calibrated classifiers** for better probabilistic confidence
- Adjusting **class weights** where needed

---

## 💻 **Tech Stack**

| Component | Tool |
|------------|------|
| Language | Python |
| Frontend | Streamlit |
| ML Libraries | Scikit-learn, XGBoost |
| Model Persistence | Joblib |
| Data Handling | Pandas, NumPy |
| Deployment | Streamlit Cloud |

---


