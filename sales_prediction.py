# Task 4: Sales Prediction using Python
# CODSOFT Data Science Internship

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Dataset
df = pd.read_csv("advertising.csv")
print("Dataset loaded successfully!\n")
print(df.head())

# 2. Check basic info
print("\nDataset Info:")
print(df.info())

# 3. Feature Selection
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Results:")
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)

# 8. Model Coefficients
print("\nModel Coefficients:")
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print(coeff_df)

input("\nPress ENTER to exit...")
