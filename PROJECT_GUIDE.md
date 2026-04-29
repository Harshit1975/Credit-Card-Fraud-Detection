# Credit Card Fraud Detection System

## 1. Project Explanation
**What is Credit Card Fraud Detection?**
Credit card fraud detection is the process of identifying potentially fraudulent transactions before or immediately after they are processed. 
**Simple Explanation**: It's like having a digital security guard that checks every purchase you make. If someone tries to buy a ₹1,00,000 laptop in a different country using your card, the guard stops it and asks you if it was really you.
**Technical Explanation**: It's a real-time binary classification machine learning problem where transactions are scored based on their probability of being fraudulent ($P(Fraud | Transaction)$). The system handles extreme class imbalance (often <1% fraud) and requires ultra-low latency (predictions in <100ms) to integrate into payment gateways.

**Why is it important & Real-world impact?**
- **Banks & Fintechs**: Minimizes financial loss, regulatory penalties, and reputational damage.
- **Customers**: Prevents unauthorized charges and maintains trust.
- **Problem solved**: Replaces slow, manual rule-based systems with dynamic, data-driven AI models that adapt to new fraud patterns.

**Workflow:**
`Transaction Data → Preprocessing & Scaling → Feature Engineering (Velocity, Patterns) → ML Model (XGBoost) → Fraud Prediction Score → Thresholding → Allow / Review / Block Alerts`

---

## 2. Tech Stack Options
**Option A (Easy)**: Scikit-learn Logistic Regression + Pandas + Streamlit (Good for basic understanding).
**Option B (Intermediate)**: Random Forest + SMOTE + Flask + Simple HTML/CSS (Good for solid ML pipelines).
**Option C (Advanced - Industry Standard)**: XGBoost/LightGBM + Optuna Tuning + FastAPI + Next.js with Tailwind CSS + User Auth (Best for Portfolios and roles).

**Selected Best Option: Option C (Advanced)**. This perfectly aligns with modern Data Science, ML Engineering, and Full-Stack MLOps roles.

---

## 3. Project Architecture
```text
[ Users / Customers ]
        |
        v (Next.js Dashboard - Auth, Visualization, Real-time Alerts)
        |
[ FastAPI Backend API ]
        |
        +-- /score (Batch Prediction)
        +-- /stream (Single Transaction)
        +-- /auth (Login/Register)
        |
[ Machine Learning Pipeline ]
        |
        +-- Preprocessing (SimpleImputer, StandardScaler, OneHotEncoder)
        +-- Feature Engineering (Time velocities, Ratios)
        +-- XGBoost Classifier (Cost-sensitive training)
        |
[ Data Storage / Simulators ]
        +-- Parquet / CSV datasets
```

---

## 4. Implementation Plan
- **Phase 1: Setup**: Initialize Next.js frontend and FastAPI backend.
- **Phase 2: Data Loading**: PII-safe transaction generation/loading.
- **Phase 3: Data Cleaning**: Chronological splitting.
- **Phase 4: EDA**: Visualizing class imbalance.
- **Phase 5: Feature Engineering**: Velocity counts, ratio features.
- **Phase 6: Model Training**: XGBoost with `scale_pos_weight` and Optuna.
- **Phase 7: Evaluation**: PR-AUC, Recall@FPR=1%, Confusion Matrix.
- **Phase 8: API Construction**: FastAPI serving the model pipeline.
- **Phase 9: Visualization**: Next.js dashboard with Login/Register and Dashboard UI.
- **Phase 10: GitHub Upload**: Repo setup, pushing code, writing README.

---

## 5. Virtual Simulation
Since we lack a real banking backend, we will simulate transactions.
1. **Normal Transactions**: Generated with patterns (e.g., $10-$50 at local grocery stores, daytime).
2. **Fraud Transactions**: Generated with anomalies (e.g., $5,000 internationally at 3 AM).
3. **Simulation**: A Python script pushes these transactions to the FastAPI `/stream` endpoint.
4. **Dashboard**: The Next.js frontend polls or receives these alerts and displays them in a red "REVIEW" state or green "ALLOW" state.
