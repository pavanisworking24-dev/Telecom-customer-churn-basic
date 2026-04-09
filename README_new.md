# 📡 Telecom Customer Churn Predictor

A end-to-end machine learning project that predicts whether a telecom customer is likely to churn, using a complete ML pipeline from EDA to SHAP interpretability.

## 🚀 Live Demo

👉 **[Launch the Streamlit App](https://customer-churn-predictor-website.streamlit.app/)**

---

## 📌 Project Overview

Customer churn is one of the most critical challenges in the telecom industry. This project builds a robust binary classification pipeline on the **IBM Telco Customer Churn dataset** (7,043 customers, 21 features) to identify at-risk customers before they leave — enabling proactive retention strategies.

---

## 🧪 ML Pipeline

| Step | Description |
|------|-------------|
| 1 | Data Loading & Exploratory Data Analysis (EDA) |
| 2 | Feature Engineering |
| 3 | Train-Test Split |
| 4 | Encoding & Scaling |
| 5 | Class Imbalance Handling (SMOTE) |
| 6 | Feature Selection (L1 Regularization) |
| 7 | Model Training (LR · Decision Tree · Random Forest · XGBoost) |
| 8 | Cross-Validation & Hyperparameter Tuning |
| 9 | Model Comparison |
| 10 | Threshold Tuning (Business Cost Optimization) |
| 11 | ROC-AUC · Precision-Recall · Business Cost Curves |
| 12 | SHAP Interpretability |

---

## 🏆 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.7424 | 0.5094 | 0.7968 | 0.6215 | **0.8392** |
| Decision Tree | 0.7310 | 0.4950 | 0.6631 | 0.5669 | 0.7893 |
| Random Forest | 0.7750 | 0.5697 | 0.6230 | 0.5951 | 0.8249 |
| XGBoost | 0.7750 | 0.5711 | 0.6123 | 0.5910 | 0.8316 |

**Best Model: Logistic Regression**
- ROC-AUC: `0.8392`
- Optimal Threshold: `0.35` (minimizes business cost)
- Final Recall: `0.9198`
- Final Precision: `0.4514`
- Final F1: `0.6056`

> The threshold was tuned to **0.35** (instead of the default 0.5) to maximize recall — catching as many churners as possible is more valuable than avoiding false alarms in a retention context.

---

## 🗂️ Repository Structure

```
├── pavan_said.ipynb          # Full ML pipeline notebook
├── best_churn_model.pkl      # Saved Logistic Regression model
├── scaler.pkl                # Fitted StandardScaler
├── selected_features.pkl     # L1-selected feature list
├── encoded_columns.pkl       # One-hot encoded column alignment
├── optimal_threshold.pkl     # Business-cost optimal threshold (0.35)
└── README.md
```

---

## ⚙️ Tech Stack

- **Language:** Python 3.12
- **ML:** scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Interpretability:** SHAP
- **Visualization:** Matplotlib, Seaborn
- **Web App:** Streamlit
- **Serialization:** Joblib

---

## 🛠️ Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/telecom-churn-predictor.git
cd telecom-churn-predictor

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py
```

---

## 📊 Dataset

**IBM Telco Customer Churn Dataset**
- 7,043 customer records
- 21 raw features → 51 after feature engineering → 32 after L1 selection
- Target: `Churn` (Yes / No)

---

## 👤 Author

**Pavan** — [GitHub](https://github.com/your-username) · [LinkedIn](https://linkedin.com/in/your-profile)

---

> ⭐ If you found this project helpful, consider giving it a star!
