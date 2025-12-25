# Task 5: Credit Card Fraud Detection
# CODSOFT Data Science Internship

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load Dataset
df = pd.read_csv("creditcard.csv")
print("Dataset loaded successfully!\n")
print(df.head())

# 2. Dataset Info
print("\nDataset Info:")
print(df.info())

# 3. Feature & Target Selection
X = df.drop('Class', axis=1)
y = df['Class']

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluation
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

input("\nPress ENTER to exit...")
