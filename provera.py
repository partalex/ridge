import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge

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


data = pd.read_csv('data-reg.csv', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Centriranje y
y -= np.mean(y)

# Kreiranje polinomijalnih karakteristika
X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)

# Standardizacija
X = StandardScaler().fit_transform(X)

# Podela na trening i test skup
train_size = int(0.8 * len(X))
indices = np.random.permutation(len(X))
train_indices, test_indices = indices[:train_size], indices[train_size:]
X_train_start, X_test_start = X[train_indices], X[test_indices]
y_train_start, y_test_start = y[train_indices], y[test_indices]

# K-Fold validacija
alpha_range = np.linspace(10, 50)
rmse_scores = []
kf = k_fold_split(X_train_start, 10)

# Računanje RMSE za različite vrednosti alpha
for alpha in alpha_range:
    fold_rmses = []
    for train_index, val_index in kf:
        X_train, X_val = X_train_start[train_index], X_train_start[val_index]
        y_train, y_val = y_train_start[train_index], y_train_start[val_index]
        ridge = Ridge(alpha=alpha).fit(X_train, y_train)
        y_pred_val = ridge.predict(X_val)
        fold_rmses.append(calculate_rmse(y_val, y_pred_val))
    rmse_scores.append(np.mean(fold_rmses))

# Prikaz rezultata
plt.plot(alpha_range, rmse_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Alpha')
plt.ylabel('Prosečna RMSE')
plt.title('RMSE u zavisnosti od Alpha')
plt.grid(True)
plt.show()

# Pronalazak najbolje alpha vrednosti
best_alpha = alpha_range[np.argmin(rmse_scores)]
best_rmse = min(rmse_scores)
print(f"Najbolje pronađeno alpha: {best_alpha} sa RMSE: {best_rmse}")

# Testiranje na test skupu
ridge = Ridge(alpha=best_alpha).fit(X_train_start, y_train_start)
y_pred_test = ridge.predict(X_test_start)
print(f"RMSE za test skup: {calculate_rmse(y_test_start, y_pred_test)}")
