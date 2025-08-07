import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)

# ------------------------- #
# Check if data file exists #
# ------------------------- #
data_path = r"E:\exl training\day19\capstone1\data\processed\cleaned3_credit_card_churn_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found: {data_path}. Please check the file path.")

# ------------------------- #
# 📁 Create Folder for EDA   #
# ------------------------- #
os.makedirs("eda", exist_ok=True)

# ------------------------- #
# 📥 Load Data               #
# ------------------------- #
df = pd.read_csv(data_path)

# ------------------------- #
# 📊 EDA Visualizations      #
# ------------------------- #

sns.boxplot(x='Churn', y='Age', data=df)
plt.title('Age vs Churn')
plt.tight_layout()
plt.savefig("eda/age_vs_churn.png")
plt.clf()

sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')
plt.tight_layout()
plt.savefig("eda/gender_distribution.png")
plt.clf()

sns.countplot(x='NumOfProducts', hue='Churn', data=df)
plt.title('Product Usage vs Churn')
plt.tight_layout()
plt.savefig("eda/product_vs_churn.png")
plt.clf()

corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig("eda/correlation_matrix.png")
plt.clf()

sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.tight_layout()
plt.savefig("eda/age_distribution.png")
plt.clf()

sns.countplot(x='Tenure', data=df)
plt.title('Tenure Analysis')
plt.tight_layout()
plt.savefig("eda/tenure_analysis.png")
plt.clf()

sns.countplot(x='Churn', data=df)
plt.title('Churn Count')
plt.tight_layout()
plt.savefig("eda/churn_vs_nonchurn.png")
plt.clf()

# ------------------------- #
# 🔧 Feature Engineering     #
# ------------------------- #

df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)

df_encoded['BalanceSalaryRatio'] = df_encoded['Balance'] / (df_encoded['EstimatedSalary'] + 1)
df_encoded['Age_Tenure_Ratio'] = df_encoded['Age'] / (df_encoded['Tenure'] + 1)

X = df_encoded.drop(['CustomerID', 'Churn'], axis=1)
y = df_encoded['Churn']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------- #
# 🧠 Train/Test Split         #
# ------------------------- #

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------- #
# 🔍 Random Forest + Tuning  #
# ------------------------- #

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42, class_weight='balanced')
search = RandomizedSearchCV(
    rf, param_distributions=param_dist,
    n_iter=20, cv=3, scoring='accuracy',
    verbose=1, random_state=42, n_jobs=-1
)

search.fit(X_train, y_train)
best_model = search.best_estimator_

# ------------------------- #
# 📈 Evaluation              #
# ------------------------- #

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_proba))

# ------------------------- #
# 🌟 Feature Importance       #
# ------------------------- #

importances = pd.Series(best_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(3)

print("\nTop 3 Influential Features:")
print(top_features)

# ------------------------- #
# 📉 ROC Curve               #
# ------------------------- #

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label='Random Forest (AUC = {:.2f})'.format(roc_auc_score(y_test, y_proba)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig("eda/roc_curve.png")
plt.clf()

# ------------------------- #
# 💾 Save Model & Metrics    #
# ------------------------- #

os.makedirs("model", exist_ok=True)

# Save the model
model_path = "model/churn_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)
print(f"Model saved to {model_path}")

# Save the metrics
metrics_path = "model/model_metrics.txt"
with open(metrics_path, "w") as f:
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred)) + "\n\n")
    
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred) + "\n")
    
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
    f.write(f"Precision: {precision_score(y_test, y_pred):.4f}\n")
    f.write(f"Recall: {recall_score(y_test, y_pred):.4f}\n")
    f.write(f"F1 Score: {f1_score(y_test, y_pred):.4f}\n")
    f.write(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}\n\n")
    
    f.write("Top 3 Influential Features:\n")
    for feature, importance in top_features.items():
        f.write(f"{feature}: {importance:.4f}\n")

print(f"Performance metrics saved to {metrics_path}")
