import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('data-reg.csv', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X = np.array(X)
y = np.array(y)
X = np.c_[np.ones((X.shape[0], 1)), X]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)


def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


alpha_values = np.linspace(0.1, 10, 100)
rmse_values = []

for alpha in alpha_values:
    I = np.eye(X_train.shape[1])
    w = np.linalg.inv(X_train.T @ X_train + alpha * I) @ X_train.T @ y_train
    y_pred = X @ w
    rmse = calculate_rmse(y_train, y_pred)
    rmse_values.append(rmse)

best_alpha = alpha_values[np.argmin(rmse_values)]

plt.plot(alpha_values, rmse_values, label='RMSE')
plt.xlabel('Alpha')
plt.ylabel('RMSE')
plt.title('RMSE vs Alpha')
plt.legend()
plt.show()

print(f"Selected alpha value: {best_alpha}")
print(f"Root Mean Squared Error: {min(rmse_values)}")

w = np.linalg.inv(X_test.T @ X_test + alpha * I) @ X_test.T @ y_test
y_pred = X @ w
print(calculate_rmse(y_test, y_pred))
