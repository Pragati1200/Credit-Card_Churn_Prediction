import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier, plot_importance
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
# Load data
df = pd.read_csv("cleaned3_credit_card_churn_data.csv")

# Feature Engineering
df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
df['Age_Tenure_Ratio'] = df['Age'] / (df['Tenure'] + 1)
df['ZeroBalanceFlag'] = (df['Balance'] == 0).astype(int)
df['HighSalaryFlag'] = (df['EstimatedSalary'] > df['EstimatedSalary'].median()).astype(int)

# Fill missing Gender with mode before encoding
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)

# Features and Target
X = df_encoded.drop(['CustomerID', 'Churn'], axis=1)
y = df_encoded['Churn']

# Impute missing values
# from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_imputed)
# balance dataset using smote
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)


# Optional: PCA for dimensionality reduction
# pca = PCA(n_components=0.95)  # Keep 95% variance
# X_pca = pca.fit_transform(X_scaled)
# print(f"💡 PCA reduced features from {X.shape[1]} to {X_pca.shape[1]}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled , y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)
# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)
# XGBoost Model
# xgb_model = XGBClassifier( eval_metric='logloss', random_state=42)
# xgb_model.fit(X_train, y_train)
# Predictions
# y_pred = xgb_model.predict(X_test)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
print("📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# Accuracy, Precision, Recall
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted')

plt.ylabel('Actual')
plt.title('Confusion Matrix - Heatmap')
plt.tight_layout()
plt.show()

# Stratified K-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = cross_val_score(xgb_model, X_scaled , y, cv=cv, scoring='accuracy')
# print(f"\n🔁 5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
cv_scores = cross_val_score(best_model, X_resampled, y_resampled, cv=cv, scoring='accuracy')
print(f"\n 5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plot_importance(best_model, importance_type='gain', max_num_features=10)
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.show()


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    f1_score, roc_curve
)
import time
import csv

# Define models
models = {
    "XGBoost": best_model,
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "SVM (RBF Kernel)": SVC(probability=True, random_state=42)
}

results = []

plt.figure(figsize=(10, 7))
for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros_like(y_pred)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "AUC-ROC": auc,
        "Train Time (s)": train_time
    })

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

# Plot formatting
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Model Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print results as a table
print("\n📊 Model Comparison Summary:")
print("{:<25} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
    "Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC", "Time(s)"
))
for r in results:
    print("{:<25} {:.4f}    {:.4f}     {:.4f}   {:.4f}    {:.4f}   {:.2f}".format(
        r['Model'], r['Accuracy'], r['Precision'], r['Recall'],
        r['F1 Score'], r['AUC-ROC'], r['Train Time (s)']
    ))

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("model_comparison_results.csv", index=False)
print("\n✅ Model comparison saved to 'model_comparison_results.csv'")

# Add VotingClassifier (soft voting for probability-based ensemble)
voting_clf = VotingClassifier(
    estimators=[
        ("XGBoost", best_model),
        ("RandomForest", RandomForestClassifier(random_state=42)),
        ("Logistic", LogisticRegression(max_iter=1000, random_state=42))
    ],
    voting='soft'  # Use 'hard' for label majority voting
)

# Train and evaluate ensemble
voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)
y_proba_voting = voting_clf.predict_proba(X_test)[:, 1]

# Metrics
print("\n🧠 Voting Classifier Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred_voting))
print("Precision:", precision_score(y_test, y_pred_voting))
print("Recall:", recall_score(y_test, y_pred_voting))
print("F1 Score:", f1_score(y_test, y_pred_voting))
print("AUC-ROC:", roc_auc_score(y_test, y_proba_voting))

# Plot Voting ROC
fpr_v, tpr_v, _ = roc_curve(y_test, y_proba_voting)
plt.figure(figsize=(6, 4))
plt.plot(fpr_v, tpr_v, label="Voting Classifier (AUC = {:.2f})".format(roc_auc_score(y_test, y_proba_voting)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Voting Classifier")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
