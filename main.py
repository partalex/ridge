import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(20)


def calculate_rmse(y_true, y_pred):
    """Izračunavanje korena srednje kvadratne greške (RMSE)"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def k_fold_split(X, k):
    """Ručno implementirana podela podataka na foldove"""
    n = len(X)
    fold_size = n // k
    indices = np.random.permutation(n)
    folds = []
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate((indices[:i * fold_size], indices[(i + 1) * fold_size:]))
        folds.append((train_indices, test_indices))
    return folds


# Učitavanje i centriranje podataka
data = pd.read_csv('data-reg.csv', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
y -= np.mean(y)

n_samples, n_features = X.shape

# Kreiranje polinomijalnih karakteristika (originalne, kvadratne i kombinovane)
num_new_features = n_features + n_features + (n_features * (n_features - 1) // 2)
X_poly = np.empty((n_samples, num_new_features))
start = 0
X_poly[:, start:start + n_features] = X
start += n_features

# Dodavanje kvadrata i međusobnih kombinacija
for i in range(n_features):
    X_poly[:, start] = X[:, i] ** 2
    start += 1
    for j in range(i, n_features):
        if i != j:
            X_poly[:, start] = X[:, i] * X[:, j]
            start += 1

# Standardizacija
X_poly = (X_poly - np.mean(X_poly, axis=0)) / np.std(X_poly, axis=0)

# Dodavanje bias-a
X = np.c_[np.ones((X_poly.shape[0], 1)), X_poly]

# Podela na trening i test skup (80% trening, 20% test)
train_size = int(0.8 * n_samples)
indices = np.random.permutation(n_samples)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train_initial, X_test_initial = X[train_indices], X[test_indices]
y_train_initial, y_test_initial = y[train_indices], y[test_indices]

# Podela trening skupa na strukove
k = 10
folds = k_fold_split(X_train_initial, k)

# Definisanje opsega za lambda i treniranje modela uz unakrsnu validaciju
lambda_values = np.linspace(10, 50)
lambda_rmse = []

for lambda_val in lambda_values:
    fold_rmses = []
    for train_indices, test_indices in folds:
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        I = np.eye(X_train.shape[1])
        I[0, 0] = 0
        w = np.linalg.inv(X_train.T @ X_train + lambda_val * I) @ X_train.T @ y_train
        y_pred = X_test @ w
        fold_rmses.append(calculate_rmse(y_test, y_pred))
    lambda_rmse.append(np.mean(fold_rmses))

# Pronalazak najbolje vrednosti lambda
best_lambda = lambda_values[np.argmin(lambda_rmse)]
best_rmse = min(lambda_rmse)

# Prikaz rezultata
plt.plot(lambda_values, lambda_rmse, label='RMSE')
plt.scatter(best_lambda, best_rmse, color='red', label=f'Najbolje lambda: {best_lambda:.2f} sa greškom {best_rmse:.2f}',
            zorder=5)
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.title('RMSE po lambda vrednosti')
plt.legend()
plt.show()

print(f"Najbolje lambda: {best_lambda} sa RMSE: {best_rmse}.")

# Ponovno treniranje sa najboljom vrednošću lambda i računanje RMSE za ceo model
I = np.eye(X_train_initial.shape[1])
I[0, 0] = 0
w = np.linalg.inv(X_train_initial.T @ X_train_initial + best_lambda * I) @ X_train_initial.T @ y_train_initial
y_pred = X_test_initial @ w
print(f"RMSE na test skupu: {calculate_rmse(y_test_initial, y_pred)}.")
