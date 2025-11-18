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

X = np.array([house["area"] for house in housing_data])
y = np.array([house["price"] for house in housing_data])

# Create train and test set
test_fraction = 0.20
test_indexes = np.random.choice(len(X), int(0.2 * len(X)), replace=False)

X_train = np.array(
    [X[i] for i in np.arange(len(X)) if i not in test_indexes], dtype=int
)
X_test = np.array([X[i] for i in np.arange(len(X)) if i in test_indexes], dtype=int)

y_train = np.array(
    [y[i] for i in np.arange(len(X)) if i not in test_indexes], dtype=int
)
y_test = np.array([y[i] for i in np.arange(len(X)) if i in test_indexes], dtype=int)

# Solve linear regression using batch gradient descent

# Hyperparamenters
tol = 0.01
learning_rate = 0.00000000001

# Parameters
theta = np.array([-1, 2])

print(np.sqrt(np.sum(theta * theta)))


def h(x, theta):
    return theta[0] + theta[1] * x


iteration = 0

while True:
    iteration += 1

    update = [
        np.sum([h(X_train[i], theta) - y_train[i] for i in np.arange(len(X_train))]),
        np.sum(
            [
                X_train[i] * (h(X_train[i], theta) - y_train[i])
                for i in np.arange(len(X_train))
            ]
        ),
    ]

    print(np.multiply(learning_rate, update))

    new_theta = theta - np.multiply(learning_rate, update)
    # err = np.max(new_theta - theta)
    err = np.sqrt(np.sum((new_theta - theta) * (new_theta - theta)))

    print("iteration: ", iteration)
    print("theta: ", theta)
    print("error: ", err)

    if err < tol:
        break

    theta = new_theta


x_range = np.linspace(np.min(X_train), np.max(X_train), 1000)
plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test, color="green")
plt.plot(x_range, theta[0] + theta[1] * x_range, color="red")
plt.show()

print(np.round(theta, 2))
