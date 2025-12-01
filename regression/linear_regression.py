# TODO
# 1. Implement multivariate regression
# 2. Implement learning rate decay
# 3. Implement Mini-batch GD
# 4. Display model comparison (also sklearn)

import numpy as np
import csv
from pathlib import Path
import matplotlib.pyplot as plt

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


class LinearRegression:
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

    def fit(self, X, y, solver="batch"):
        # Data statistics
        self.x_max_, self.x_min_ = X.max(), X.min()
        self.y_max_, self.y_min_ = y.max(), y.min()
        m = len(X)

        # Normalize data
        X_norm = (X - self.x_min_) / (self.x_max_ - self.x_min_)
        y_norm = (y - self.y_min_) / (self.y_max_ - self.y_min_)

        self.theta_ = np.zeros(X.ndim + 1)
        previous_cost = float("inf")

        if solver == "batch":
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

                # print("iter: ", i)
                # print("theta: ", self.theta_)

                if np.abs(current_cost - previous_cost) < self.tol:
                    print("Batch GD converged at iteration ", i)
                    break

                previous_cost = current_cost

        if solver == "sgd":
            for epoch in range(self.max_iter):
                # Random Shuffling
                indices = np.random.permutation(m)
                X_shuffled = X_norm[indices]
                y_shuffled = y_norm[indices]

                for i in range(len(X_norm)):
                    x_i = X_shuffled[i]
                    y_i = y_shuffled[i]

                    prediction = self.theta_[0] + self.theta_[1] * x_i
                    error = prediction - y_i

                    # Update gradient

                    self.theta_[0] -= self.learning_rate * error
                    self.theta_[1] -= self.learning_rate * error * x_i

                # Cost of epoch
                full_pred = self.theta_[0] + self.theta_[1] * X_norm
                current_cost = np.sum((full_pred - y_norm) ** 2) / (2 * m)
                self.costs_.append(current_cost)

                # print("epoch: ", epoch)
                # print(f"iter: {i}/{m}")
                # print("theta: ", self.theta_)

                if abs(previous_cost - current_cost) < self.tol:
                    print("SGD converged at epoch: ", epoch)
                    break
                previous_cost = current_cost

        if solver == "normal":
            X_augmented = np.c_[np.ones(len(X_norm)), X_norm]
            self.theta_ = (
                np.linalg.pinv(X_augmented.T @ X_augmented) @ X_augmented.T @ y_norm
            )
        return self

    def predict(self, X):
        if self.theta_ is None:
            raise Exception("The model has not been trained (use .fit())")

        X_norm = (X - self.x_min_) / (self.x_max_ - self.x_min_)

        y_pred_norm = self.theta_[0] + self.theta_[1] * X_norm

        y_pred_real = y_pred_norm * (self.y_max_ - self.y_min_) + self.y_min_

        return y_pred_real


# Solver = Batch Gradient Descent
model1 = LinearRegression(learning_rate=1.5, tol=0.000001, max_iter=5000)
model1.fit(X_train, y_train, solver="batch")
y_pred1 = model1.predict(X_test)

# Solver = Stochastic Gradient Descent
model2 = LinearRegression(learning_rate=0.001, tol=0.000001, max_iter=5000)
model2.fit(X_train, y_train, solver="sgd")
y_pred2 = model2.predict(X_test)

# Solver = Normal Equations
model3 = LinearRegression(max_iter=5000)
model3.fit(X_train, y_train, solver="normal")
y_pred3 = model3.predict(X_test)

x_range = np.linspace(np.min(X_train), np.max(X_train), 1000)


plt.scatter(X_test, y_test, color="blue")
plt.scatter(X_train, y_train, color="blue", alpha=0.2)
plt.plot(X_test, y_pred1, color="green", label="BGD")
plt.plot(X_test, y_pred2, color="red", label="SGD")
plt.plot(X_test, y_pred3, color="purple", label="Normal")

plt.legend(loc="upper right")
plt.show()
