# ============================================================
# Assignment 3
# Salary Prediction using Polynomial Regression
# ============================================================

# Import Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split

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