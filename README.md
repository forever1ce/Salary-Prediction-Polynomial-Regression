# Salary Prediction using Polynomial Regression

## Objective

The objective of this project is to predict employee salaries based on their position level using a Polynomial Regression model. Since the relationship between position level and salary is non-linear, Polynomial Regression provides a better fit than Linear Regression.

---

## Dataset

**Position Salaries Dataset**

Kaggle Link:
https://www.kaggle.com/datasets/akram24/position-salaries

> **Note:** The dataset is not included in this repository. Please download it from the Kaggle link above.

---

## Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## Methodology

1. Loaded the dataset using Pandas.
2. Performed data understanding by viewing records, dataset information, and summary statistics.
3. Checked for missing values.
4. Selected **Level** as the input feature and **Salary** as the target variable.
5. Split the dataset into 80% training and 20% testing sets.
6. Applied Polynomial Feature Transformation with Degree = 3.
7. Trained a Polynomial Regression model.
8. Predicted salaries for the test dataset.
9. Evaluated the model using MAE, MSE, and R² Score.
10. Visualized the original data and Polynomial Regression curve.

---

## Results

The Polynomial Regression model successfully captured the non-linear relationship between position level and salary.

Evaluation Metrics:

- Mean Absolute Error (MAE): **70635.25**
- Mean Squared Error (MSE): **6263853282.86**
- R² Score: **0.8763**

The regression curve closely follows the salary trend and performs significantly better than a simple linear model for this dataset.

---

## Conclusion

Polynomial Regression effectively models the non-linear relationship between employee position level and salary. Compared to Linear Regression, it provides a much better fit for datasets where the target variable changes non-linearly. The model achieved a good R² score, indicating strong predictive performance. One major advantage of Polynomial Regression is its ability to capture curved patterns without requiring additional input features. However, choosing a very high polynomial degree may lead to overfitting, so selecting an appropriate degree is important for maintaining good generalization.

---

## Repository Structure

```
Assignment-3.py
README.md
requirements.txt
.gitignore
```

---

## Author

**Tanishk Dhyani**