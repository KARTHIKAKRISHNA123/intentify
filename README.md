

---

### 🛒 Intentify: Real-Time E-Commerce Conversion Engine

**Elevator Pitch (For Interviews):**
"Intentify is a real-time machine learning inference engine designed to optimize e-commerce conversion rates. It analyzes live user session telemetry to predict purchase intent, allowing frontend systems to trigger dynamic, targeted marketing interventions—like personalized discounts—only to users who need a nudge, thereby maximizing revenue while preserving profit margins."

**Comprehensive Description (For GitHub / Resume):**

**The Problem:**
Traditional e-commerce platforms suffer from "blanket marketing," burning revenue by offering universal discounts to all visitors, including those who would have purchased at full price. Furthermore, without real-time session visibility, high-intent users often bounce without intervention.

**The Solution:**
Built a cost-sensitive Machine Learning pipeline deployed as a cloud-native SaaS application. The system captures live clickstream data (page durations, bounce rates, behavioral patterns), computes engineered features on the fly via a custom middleware layer, and serves instantaneous conversion probabilities.

**Key Engineering Achievements:**

* **Algorithmic Optimization:** Engineered a custom `scikit-learn` pipeline handling severe class imbalance (3.8% vs 56.3% distribution) using algorithmic penalization (`class_weight='balanced'`).
* **Robust Feature Architecture:** Designed a dynamic `ColumnTransformer` for precise feature routing (One-Hot Encoding for categorical IDs, Standard Scaling for continuous metrics), eliminating data leakage and ensuring production stability.
* **Hyperparameter Pruning:** Automated tree-depth and leaf-node constraint discovery using `GridSearchCV`, optimizing strictly for the **F1-Score** to perfectly balance False Positives (wasted marketing spend) and False Negatives (lost sales).
* **High-Recall Performance:** Achieved an **86% Recall rate** on the minority class (buyers) with a finalized F1-score of 0.61, successfully capturing the vast majority of potential revenue generating sessions.
* **Cloud Deployment:** Serialized the optimized ML pipeline and deployed it as a containerized web application on Hugging Face Spaces using Streamlit, featuring real-time interactive parameter adjustments and integrated Laplace smoothing to prevent edge-case server crashes.

**Technical Stack:**

* **Core ML & Data:** Python, Pandas, NumPy, Scikit-Learn
* **Architecture & MLOps:** Pipeline encapsulation, Joblib serialization, GridSearch optimization
* **Deployment & UI:** Streamlit, Hugging Face Spaces, Git

  # 🛒 Intentify: Real-Time E-Commerce Conversion Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4.1-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 Overview
Intentify is a real-time machine learning inference engine designed to optimize e-commerce conversion rates. It analyzes live user session telemetry (clickstream data, page durations, bounce rates) to predict purchase intent. This allows frontend systems to trigger dynamic, targeted marketing interventions—like personalized discounts—only to users who need a nudge, maximizing revenue while preserving profit margins.

## 🚀 Key Engineering Features
* **Cost-Sensitive Learning:** Handles severe class imbalance (3.8% vs 56.3%) using algorithmic penalization (`class_weight='balanced'`).
* **Dynamic Middleware:** Computes engineered features (e.g., `avg_time_per_product`, `bounce_exit_score`) on the fly from raw telemetry with integrated Laplace smoothing to prevent division-by-zero crashes.
* **Hyperparameter Optimization:** Utilized `GridSearchCV` to optimize the F1-Score, perfectly balancing False Positives (wasted ad spend) and False Negatives (lost sales), achieving an **86% Recall rate** on the minority class.
* **Robust Pipeline Architecture:** Employs a Scikit-Learn `ColumnTransformer` for precise feature routing (OHE for categorical IDs, Standard Scaling for continuous metrics), eliminating data leakage.

## 🛠️ Tech Stack
* **Machine Learning:** `scikit-learn`, `pandas`, `numpy`, `joblib`
* **Web Framework:** `streamlit`
* **Deployment:** Hugging Face Spaces

## 📂 Repository Structure
```text
├── app.py                     # Streamlit frontend & inference middleware
├── intentify_model.pkl        # Serialized ML pipeline artifact
├── requirements.txt           # Dependency manifest
├── shopSense.ipynb            # Core ML training & evaluation notebook
├── shop_smart_ecommerce.csv   # Raw training dataset
└── README.md                  # Project documentation

---
