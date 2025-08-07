import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans

# ------------------------- #
# 📥 Load Data               #
# ------------------------- #
data_path = r"E:\exl training\day19\capstone1\data\processed\cleaned3_credit_card_churn_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}. Please check the file path.")

df = pd.read_csv(data_path)

# ------------------------- #
# ✨ Feature Engineering      #
# ------------------------- #
df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)

df_encoded['BalanceSalaryRatio'] = df_encoded['Balance'] / (df_encoded['EstimatedSalary'] + 1)
df_encoded['Age_Tenure_Ratio'] = df_encoded['Age'] / (df_encoded['Tenure'] + 1)
# df_encoded['CreditUtilization'] = df_encoded['Balance'] / (df_encoded['CreditScore'] + 1)
df_encoded['Tenure_Salary_Ratio'] = df_encoded['Tenure'] / (df_encoded['EstimatedSalary'] + 1)
df_encoded['Log_Balance'] = np.log1p(df_encoded['Balance'])
df_encoded['Interaction_Feature'] = df_encoded['IsActiveMember'] * df_encoded['Age_Tenure_Ratio']

# ------------------------- #
# 🧼 Preprocessing            #
# ------------------------- #
X = df_encoded.drop(['CustomerID', 'Churn'], axis=1)
y = df_encoded['Churn']

imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

# ------------------------- #
# 🔵 Clustering Feature       #
# ------------------------- #
kmeans = KMeans(n_clusters=2, random_state=42)
X_scaled['Cluster'] = kmeans.fit_predict(X_scaled)

# ------------------------- #
# 🧠 Train/Test Split         #
# ------------------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------- #
# 🧪 Apply SMOTE              #
# ------------------------- #
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# ------------------------- #
# 🔍 Random Forest + Tuning  #
# ------------------------- #
param_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(class_weight='balanced', random_state=42)
clf_rf = RandomizedSearchCV(rf, param_rf, n_iter=10, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1)
clf_rf.fit(X_resampled, y_resampled)
y_proba_rf = clf_rf.predict_proba(X_test)[:, 1]
y_pred_rf = clf_rf.predict(X_test)

# 🌲 Gradient Boosting
param_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.1, 0.01],
    'max_depth': [3, 5]
}

gb = GradientBoostingClassifier(random_state=42)
clf_gb = RandomizedSearchCV(gb, param_gb, n_iter=6, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1)
clf_gb.fit(X_resampled, y_resampled)
y_proba_gb = clf_gb.predict_proba(X_test)[:, 1]
y_pred_gb = clf_gb.predict(X_test)

# 📉 Logistic Regression
param_lr = {
    'C': [0.01, 0.1, 1.0],
    'penalty': ['l2'],
    'solver': ['lbfgs'],
}

lr = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
clf_lr = RandomizedSearchCV(lr, param_lr, n_iter=3, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1)
clf_lr.fit(X_resampled, y_resampled)
y_proba_lr = clf_lr.predict_proba(X_test)[:, 1]
y_pred_lr = clf_lr.predict(X_test)

# 🔺 XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
param_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.01],
    'subsample': [0.8, 1.0],
}

clf_xgb = RandomizedSearchCV(xgb, param_xgb, n_iter=6, scoring='roc_auc', cv=3, random_state=42, n_jobs=-1)
clf_xgb.fit(X_resampled, y_resampled)
y_proba_xgb = clf_xgb.predict_proba(X_test)[:, 1]
y_pred_xgb = clf_xgb.predict(X_test)

# 🧱 HistGradientBoosting
hgb = HistGradientBoostingClassifier(random_state=42)
hgb.fit(X_resampled, y_resampled)
y_proba_hgb = hgb.predict_proba(X_test)[:, 1]
y_pred_hgb = hgb.predict(X_test)

# 🤝 Stacking Ensemble
estimators = [
    ('rf', clf_rf.best_estimator_),
    ('gb', clf_gb.best_estimator_),
    ('lr', clf_lr.best_estimator_),
    ('xgb', clf_xgb.best_estimator_)
]

stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), n_jobs=-1, cv=3)
stack.fit(X_resampled, y_resampled)
y_pred_stack = stack.predict(X_test)
y_proba_stack = stack.predict_proba(X_test)[:, 1]


from sklearn.svm import SVC

param_svm = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'poly']
}

svm = SVC(probability=True, class_weight='balanced', random_state=42)
clf_svm = RandomizedSearchCV(svm, param_svm, n_iter=5, scoring='roc_auc', cv=3, n_jobs=-1, random_state=42)
clf_svm.fit(X_resampled, y_resampled)

y_pred_svm = clf_svm.predict(X_test)
y_proba_svm = clf_svm.predict_proba(X_test)[:, 1]

# 📊 Evaluation Function
def evaluate_model(name, y_true, y_pred, y_proba):
    print(f"\n🔍 {name} Evaluation")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("AUC-ROC:", roc_auc_score(y_true, y_proba))
    print("AP Score:", average_precision_score(y_true, y_proba))
    print("\nConfusion Matrix:")
    # sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d')
    # plt.title(f"{name} Confusion Matrix")
    # plt.show()
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

# 🧾 Individual Evaluations
evaluate_model("Stacking Ensemble", y_test, y_pred_stack, y_proba_stack)
evaluate_model("Random Forest", y_test, y_pred_rf, y_proba_rf)
evaluate_model("Gradient Boosting", y_test, y_pred_gb, y_proba_gb)
evaluate_model("Logistic Regression", y_test, y_pred_lr, y_proba_lr)
evaluate_model("XGBoost", y_test, y_pred_xgb, y_proba_xgb)
evaluate_model("HistGradientBoosting", y_test, y_pred_hgb, y_proba_hgb)

evaluate_model("Support Vector Machine", y_test, y_pred_svm, y_proba_svm)

# 🌟 Feature Importance (Random Forest)
importances = pd.Series(clf_rf.best_estimator_.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(5)
print("\nTop Influential Features from Random Forest:")
print(top_features)
