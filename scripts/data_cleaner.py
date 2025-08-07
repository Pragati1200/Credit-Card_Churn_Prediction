import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("exl_credit_card_churn_data.csv")
print("Initial rows:", len(df))

# Clean Gender
df['Gender'] = df['Gender'].astype(str).str.strip().str.capitalize()
df['Gender'] = df['Gender'].replace('', np.nan)
df['Gender'] = df['Gender'].fillna('Unknown')
print("After gender cleaning:", len(df))

# Fix HasCrCard
df['HasCrCard'] = df['HasCrCard'].replace({'Yes': 1, 'No': 0})
df['HasCrCard'] = pd.to_numeric(df['HasCrCard'], errors='coerce').fillna(0)

# Fix IsActiveMember
df['IsActiveMember'] = pd.to_numeric(df['IsActiveMember'], errors='coerce')
df['IsActiveMember'] = df['IsActiveMember'].fillna(0).clip(0, 1)
print("After HasCrCard & IsActiveMember fix:", len(df))

# Remove invalid Age/Salary
df = df[(df['Age'] > 0) & (df['Age'] <= 100)]
df = df[df['EstimatedSalary'] >= 0]
print("After Age & Salary filter:", len(df))

# Handle missing Balance
df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
df['Balance'] = df['Balance'].fillna(df['Balance'].median())

# Clean and convert Churn
df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')  # convert to float
df = df[df['Churn'].isin([0.0, 1.0])]                      # filter valid churn
df['Churn'] = df['Churn'].astype(int)                      # convert to int
print("After filtering Churn:", len(df))

# Save cleaned version
df.reset_index(drop=True, inplace=True)
df.to_csv("cleaned3_credit_card_churn_data.csv", index=False)

print("🎉 Cleaned data saved to: cleaned3_credit_card_churn_data.csv")
print("🧾 Final cleaned data shape:", df.shape)
