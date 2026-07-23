# ============================================================
# Assignment 3
# Salary Prediction using Polynomial Regression
# ============================================================

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# Task 1: Data Understanding
# ============================================================

# Load the dataset
df = pd.read_csv("Position_Salaries.csv")

# Display first five records
print("=" * 60)
print("First Five Records")
print("=" * 60)
print(df.head())

# Display dataset information
print("\n" + "=" * 60)
print("Dataset Information")
print("=" * 60)
df.info()

# Display summary statistics
print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)
print(df.describe())

# Identify input feature and target variable
print("\n" + "=" * 60)
print("Features")
print("=" * 60)

print("Input Feature :", "Level")
print("Target Variable :", "Salary")


# ============================================================
# Task 2: Data Preprocessing
# ============================================================

# Check for missing values
print("\n" + "=" * 60)
print("Missing Values")
print("=" * 60)
print(df.isnull().sum())

# Select feature and target
X = df[['Level']]
y = df['Salary']

print("\nSelected Feature:")
print(X.head())

print("\nSelected Target:")
print(y.head())

# Split dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Feature Shape :", X_train.shape)
print("Testing Feature Shape  :", X_test.shape)
print("Training Target Shape  :", y_train.shape)
print("Testing Target Shape   :", y_test.shape)

# ============================================================
# Task 3: Model Development
# ============================================================

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Transform the input feature into polynomial features (Degree = 3)
poly = PolynomialFeatures(degree=3)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train the Polynomial Regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)

print("\nPolynomial Regression model trained successfully!")

# Predict salaries for the test dataset
y_pred = model.predict(X_test_poly)

print("\nPredicted Salaries:")
print(y_pred)

# ============================================================
# Task 4: Model Evaluation
# ============================================================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("Model Evaluation")
print("=" * 60)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R2 Score: {r2:.4f}")

# ============================================================
# Visualization
# ============================================================

plt.figure(figsize=(8, 6))

# Original data
plt.scatter(X, y, color="blue", label="Actual Data")

# Smooth curve
X_grid = np.arange(X.min().values[0], X.max().values[0], 0.1).reshape(-1, 1)

plt.plot(
    X_grid,
    model.predict(poly.transform(X_grid)),
    color="red",
    linewidth=2,
    label="Polynomial Regression"
)

plt.title("Salary Prediction using Polynomial Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.legend()

plt.show()