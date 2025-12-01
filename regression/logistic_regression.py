# TODO
# 1. Implement solver using Newton's method
# 2. Implement multilinear regression

import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LG

file_path = Path(__file__)
file_dir = file_path.parent
base_dir = file_dir.parent
data_path = base_dir / "data/breast_cancer_data.csv"

# Gather data
with open(data_path) as csvfile:
    reader = csv.DictReader(csvfile)
    data = list(reader)

X = np.array([d["radius_mean"] for d in data], dtype=float)
y = np.array([0 if d["diagnosis"] == "B" else 1 for d in data], dtype=int)

print(np.sum(y) / len(y))

np.random.seed(42)


def test_indexes(x, test_fraction=0.20):
    return np.random.choice(len(x), int(test_fraction * len(x)), replace=False)


test_indexes = test_indexes(X, test_fraction=0.20)

X_train = np.array([X[i] for i in np.arange(len(X)) if i not in test_indexes])
X_test = np.array([X[i] for i in np.arange(len(X)) if i in test_indexes])

y_train = np.array([y[i] for i in np.arange(len(X)) if i not in test_indexes])
y_test = np.array([y[i] for i in np.arange(len(X)) if i in test_indexes])


class LogisticRegression:
    def __init__(self, learning_rate=0.001, max_iter=1000, tol=0.001):
        # Hyperparamenters
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

        # State variables
        self.theta_ = None
        self.costs_ = []

        # Variables for normalization
        self.x_max_ = None
        self.x_min_ = None

    def _h(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y, solver="batch"):
        # Data statistics
        self.x_max_, self.x_min_ = X.max(), X.min()
        m = len(X)

        # Normalize data
        X_norm = (X - self.x_min_) / (self.x_max_ - self.x_min_)

        self.theta_ = np.zeros(X.ndim + 1)
        previous_cost = float("-inf")

        if solver == "batch":
            for i in range(self.max_iter):

                y_pred = self._h(self.theta_[0] + self.theta_[1:] * X_norm)
                error = y - y_pred

                # Gradients (using averages)
                d_theta0 = (1 / m) * np.sum(error)
                d_theta1 = (1 / m) * np.sum(error * X_norm)

                # Update theta
                self.theta_[0] += self.learning_rate * d_theta0
                self.theta_[1] += self.learning_rate * d_theta1

                # Cost (log-likelihood) for monitoring
                current_cost = (
                    np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)) / m
                )
                self.costs_.append(current_cost)

                if np.abs(current_cost - previous_cost) < self.tol:
                    print("Batch GA converged at iteration ", i)
                    break

                previous_cost = current_cost

        return self

    def predict(self, X):
        if self.theta_ is None:
            raise Exception("The model has not been trained (use .fit())")

        X_norm = (X - self.x_min_) / (self.x_max_ - self.x_min_)

        y_pred = np.round(self._h(self.theta_[0] + self.theta_[1] * X_norm))

        return y_pred


model1 = LogisticRegression(learning_rate=1.5, tol=0.000001, max_iter=5000)
model1.fit(X_train, y_train, solver="batch")
y_pred1 = model1.predict(X_test)

model2 = LG(tol=0.000001, max_iter=5000)
model2.fit(X_train.reshape(-1, 1), y_train)
y_pred2 = model2.predict(X_test.reshape(-1, 1))

correc1 = np.mean(y_pred1 == y_test)
correc2 = np.mean(y_pred2 == y_test)
print(f"Correctly classified (model 1): {np.round(correc1 * 100, 2)}%")
print(f"Correctly classified (model 2): {np.round(correc2 * 100, 2)}%")

plt.scatter(X_test, y_test, color="blue", label="actual data", alpha=0.2)

x_range = np.linspace(X.min(), X.max(), 300)
x_range_norm = (x_range - model1.x_min_) / (model1.x_max_ - model1.x_min_)
z = model1.theta_[0] + model1.theta_[1] * x_range_norm
y_sigmoid = model1._h(z)

plt.plot(x_range, y_sigmoid, color="red", label="logistic curve")

plt.legend(loc="lower right")
plt.show()
