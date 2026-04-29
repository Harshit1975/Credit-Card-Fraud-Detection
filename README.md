<img width="1708" height="857" alt="Screenshot 2026-04-28 155855" src="https://github.com/user-attachments/assets/3a5716e8-6895-433d-8617-575325e37fac" />
# 🛡️ Nexus Guard Enterprise: Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)
![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Imbalanced%20Learning-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end, full-stack Machine Learning pipeline and real-time streaming dashboard designed to detect fraudulent credit card transactions. Built with enterprise-grade architecture, this system simulates high-frequency financial telemetry, performs low-latency inference using a finely tuned XGBoost model, and visualizes Threat Intelligence on a Next.js frontend.

---

## 🌟 Key Features

* **Real-Time Data Streaming:** Simulates a live Kafka-style data stream by chronologically polling a massive Parquet data lake.
* **Predictive AI Engine (XGBoost):** Handles severe class imbalances (99% valid / 1% fraud) using advanced feature engineering, Synthetic Minority Over-sampling (SMOTE/Scale Pos Weight), and Hyperparameter tuning (Optuna).
* **Explainable AI (XAI):** Doesn't just block transactions; it returns human-readable risk factors (e.g., `High Velocity`, `International Geo`) to guide Fraud Analysts.
* **Enterprise Analyst Dashboard:** Built with Next.js, Tailwind CSS, and Recharts. Features dynamic thresholding, financial loss tracking, live threat topology, and real-time inference graphing.
* **Full-Stack MLOps Pipeline:** Includes automated data generation, preprocessing, model training, and a FastAPI inference gateway.

---

## 🏗️ System Architecture

### 1. Data Engineering (`/notebooks`)
* Synthetic Data Generation Engine simulating realistic user purchasing behaviors (geo-locations, merchant categories, velocity metrics).
* Raw data is processed, engineered (hashing, time-series velocity extraction), and saved as a highly compressed `.parquet` Data Lake.

### 2. Machine Learning (`/src`)
* `train_baselines.py`: Evaluates multiple models (Logistic Regression, Random Forest, LightGBM).
* `tune_optuna.py`: Performs Bayesian optimization to maximize the **Average Precision (PR-AUC)** metric, which is critical for highly imbalanced fraud datasets.

### 3. API Gateway (`/app`)
* **FastAPI** backend serving low-latency (`<25ms`) predictions.
* Exposes `/stream` (for manual and automated inference) and `/sample_transaction` (for data playback).

### 4. Frontend (`/frontend`)
* **Next.js & React** dashboard acting as the Fraud Operations Center.
* Features live Recharts activity graphs, Active Geo-Node load balancing simulation, and a Threat Intelligence action queue.

---

## 🚀 Quickstart Guide

### Prerequisites
* Python 3.10+
* Node.js v18+

### Step 1: Backend Setup
Clone the repository and install the Python dependencies.
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Generate Data & Train Model
Run the automated pipeline to generate synthetic data, extract features, and train the XGBoost model.
```bash
python run_pipeline.py
```

### Step 3: Start the Inference API
Launch the FastAPI backend.
```bash
uvicorn app.main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

### Step 4: Launch the Enterprise Dashboard
Open a new terminal, navigate to the frontend directory, and start the Next.js server.
```bash
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:3000` to view the dashboard! Use any credentials to login.

---

## 📊 Evaluation Metrics

Because fraud datasets are highly imbalanced, accuracy is a misleading metric. This model is evaluated strictly on:
* **PR-AUC (Precision-Recall Area Under Curve)**
* **F1-Score**
* **Business Cost Function:** Minimizing False Negatives (Financial Loss) while maintaining an acceptable False Positive Rate (Customer Friction).

The UI allows Analysts to dynamically slide the **Risk Threshold** (e.g., `0.95` vs `0.05`), instantly updating the business logic without requiring a model redeployment.

---

## 👨‍💻 Built By
*A Machine Learning Engineer passionate about MLOps, Data Engineering, and Full-Stack AI Product Development.*

<img width="1708" height="857" alt="Screenshot 2026-04-28 155855" src="https://github.com/user-attachments/assets/372897eb-5100-405d-8fa9-c847a7bd68e5" />

<img width="837" height="680" alt="Screenshot 2026-04-29 093605" src="https://github.com/user-attachments/assets/9057f9bf-b550-42a9-82f9-7db0998e527c" />

<img width="1282" height="827" alt="Screenshot 2026-04-28 160357" src="https://github.com/user-attachments/assets/f27e8031-c753-4bf2-98d3-c0d6e435c179" />


