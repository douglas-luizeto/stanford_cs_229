import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import os

script_path = Path(__file__)
base_dir = (script_path.parent).parent
data_path = base_dir / "data/Housing.csv"

# Gather data
with open(data_path) as file:
    reader = csv.DictReader(file)
    housing_data = list(reader)

X = np.array([house["area"] for house in housing_data], dtype=int)
y = np.array([house["price"] for house in housing_data], dtype=int)

# Create train and test set with normalized data
np.random.seed(42)


def test_indexes(x, test_fraction=0.20):
    return np.random.choice(len(x), int(test_fraction * len(x)), replace=False)


test_indexes = test_indexes(X, test_fraction=0.20)

X_train = np.array([X[i] for i in np.arange(len(X)) if i not in test_indexes])
X_test = np.array([X[i] for i in np.arange(len(X)) if i in test_indexes])

y_train = np.array([y[i] for i in np.arange(len(X)) if i not in test_indexes])
y_test = np.array([y[i] for i in np.arange(len(X)) if i in test_indexes])


class BatchLinearRegression:
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
        self.y_max_ = None
        self.y_min_ = None

        # Variables for Data
        self.X = None
        self.y = None

    def fit(self, X, y):
        # Data statistics
        self.x_max_, self.x_min_ = X.max(), X.min()
        self.y_max_, self.y_min_ = y.max(), y.min()
        m = len(X)

        # Normalize data
        X_norm = (X - self.x_min_) / (self.x_max_ - self.x_min_)
        y_norm = (y - self.y_min_) / (self.y_max_ - self.y_min_)

        print(np.max(X), np.min(X), np.max(y), np.min(y))

        self.theta_ = np.zeros(X.ndim + 1)
        previous_cost = float("inf")

        for i in range(self.max_iter):

            y_pred = self.theta_[0] + self.theta_[1:] * X_norm
            error = y_pred - y_norm

            # Gradients (using averages)
            d_theta0 = (1 / m) * np.sum(error)
            d_theta1 = (1 / m) * np.sum(error * X_norm)

            # Update theta
            self.theta_[0] -= self.learning_rate * d_theta0
            self.theta_[1] -= self.learning_rate * d_theta1

            # Cost (MSE) for monitoring
            current_cost = np.sum(error**2) / (2 * m)
            self.costs_.append(current_cost)

            print("iter: ", i)
            print("theta: ", self.theta_)

            if np.abs(current_cost - previous_cost) < self.tol:
                print("Convergence at iteration ", i)
                break

            previous_cost = current_cost

        return self

    def predict(self, X):
        if self.theta_ is None:
            raise Exception("The model has not been trained (use .fit())")

        X_norm = (X - self.x_min_) / (self.x_max_ - self.x_min_)

        y_pred_norm = self.theta_[0] + self.theta_[1] * X_norm

        y_pred_real = y_pred_norm * (self.y_max_ - self.y_min_) + self.y_min_

        return y_pred_real


model = BatchLinearRegression(learning_rate=0.9, tol=0.00000001, max_iter=5000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

x_range = np.linspace(np.min(X_train), np.max(X_train), 1000)
# plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test, color="green")
plt.plot(X_test, y_pred, color="orange")
plt.show()
