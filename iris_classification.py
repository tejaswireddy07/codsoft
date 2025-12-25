# Task 3: Iris Flower Classification

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset
df = pd.read_csv("IRIS.csv")
print("Dataset loaded successfully!\n")
print(df.head())

# 2. Separate features and target
X = df.drop("species", axis=1)
y = df["species"]

# 3. Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

input("\nPress ENTER to exit...")
