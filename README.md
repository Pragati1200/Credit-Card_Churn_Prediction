# Credit Card Churn Prediction – Capstone Project (EXL)

## 📌 Project Overview

This project is part of the EXL Capstone Program, under the **Credit Card Analytics Division**. It aims to build a machine learning model to predict credit card churn based on customer demographic and behavioral data.

## 🧠 Objective

- Predict which customers are likely to churn.
- Identify early warning signals using data science and visual analytics.
- Host the solution using **AWS Aurora MySQL** and **AWS EC2**, simulating a production environment.

---

## 🗂️ Folder Structure


credit-card-churn-prediction/
├── data/ # Dataset files
├── notebooks/ # Jupyter notebooks (EDA, model training)
├── src/ # Python scripts and reusable code
├── aws/ # AWS EC2, Aurora setup and configs
├── presentation/ # PPT slides and project documentation
├── requirements.txt # Python package dependencies
├── README.md # This file
├── git_flow.md # Git flow and branching strategy


---

## 🧾 Dataset

- **File**: `exl_credit_card_customers.csv`
- **Source**: Provided by EXL (10-record sample)
- **Encoding Rule**: Only **One-Hot Encoding** allowed for categorical variables.

Churn definition:
- No transaction for 3+ months
- Monthly spend < 50% of last known average
- Label marked as `"Churn: Yes"`

---

## 🔧 Tech Stack

- **Language**: Python 3.x
- **ML Libraries**: pandas, scikit-learn, seaborn, matplotlib
- **Deployment**: AWS EC2 (Linux), AWS Aurora MySQL
- **Version Control**: Git + GitHub

---

## 🧪 Model Building

1. Data Preprocessing
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training & Evaluation
5. Deployment to AWS

---

## 🚀 Deployment Overview

- **Aurora MySQL**: Stores processed data and churn predictions.
- **EC2**: Hosts the model and exposes a REST API (optional).

---

## 🚦 How to Run the Project

```bash
# Step 1: Clone the repo
git clone <repo-url>
cd capstone1





# Step 2: Install dependencies
pip install -r requirements.txt

# run script on vscode
PS E:\exl training\day19\capstone1> & C:/Users/praga/AppData/Local/Microsoft/WindowsApps/python3.13.exe "e:/exl training/day19/capstone1/eda_featureeng_model.py"