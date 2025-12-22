# =========================================================
# CodeSoft Task 2: Movie Rating Prediction with Python
# =========================================================

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =========================================================
# 1. Load Dataset (Auto-detect CSV file)
# =========================================================

csv_file = None
for file in os.listdir():
    if file.lower().endswith(".csv"):
        csv_file = file
        break

if csv_file is None:
    raise FileNotFoundError("❌ No CSV file found in the current folder")

print(f"✅ Dataset detected: {csv_file}")

df = pd.read_csv(csv_file, encoding="latin1")

print("\n📌 First 5 rows of dataset:")
print(df.head())

# =========================================================
# 2. Select Required Columns
# =========================================================

# Standard columns present in IMDb Movies India dataset
required_columns = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Rating']

df = df[required_columns]

# =========================================================
# 3. Data Cleaning
# =========================================================

# Remove rows with missing ratings
df = df.dropna(subset=['Rating'])

# Fill missing categorical values
df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']] = (
    df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']].fillna("Unknown")
)

# Convert Rating to numeric
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df = df.dropna(subset=['Rating'])

print("\n✅ Data cleaned successfully")

# =========================================================
# 4. Encode Categorical Columns
# =========================================================

label_encoders = {}
for column in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

print("✅ Categorical encoding completed")

# =========================================================
# 5. Feature Selection
# =========================================================

X = df.drop('Rating', axis=1)
y = df['Rating']

# =========================================================
# 6. Train-Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================================
# 7. Model Training (Random Forest Regressor)
# =========================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\n✅ Model training completed")

# =========================================================
# 8. Model Evaluation
# =========================================================

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)

# =========================================================
# 9. Sample Prediction
# =========================================================

sample = X_test.iloc[0:1]
predicted_rating = model.predict(sample)

print("\n🎬 Sample Movie Rating Prediction:", predicted_rating[0])

# =========================================================
# END OF TASK 2
# =========================================================


