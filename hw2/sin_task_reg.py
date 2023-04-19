import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import matplotlib
import my_model
matplotlib.use('TkAgg')

# Task 4 data
np.random.seed(48763)
m = 15
x = np.linspace(0, 1, num=m)
np.random.shuffle(x)
y = np.sin(2 * np.pi * x) + np.random.normal(loc=0, scale=0.2, size=m)

x_gt = np.linspace(0, 1, num=1000)
y_gt = np.sin(2 * np.pi * x_gt)

# Task 6 lambdas
lambdas = [0, 0.001, 1, 1000]
colors = ['red', 'green', 'blue', 'orange']

# Task 4: Polynomial Regression (Degree 14)
degree = 14
X_poly = my_model.my_transform(x.reshape(-1, 1), degree=degree)
plt.scatter(x, y, label='Data', color='grey')
for i, l in enumerate(lambdas):
    model_ridge = my_model.MyRidgeRegression(alpha=l)
    mse_cv_ridge = -np.mean(cross_val_score(model_ridge, X_poly, y, cv=5, scoring='neg_mean_squared_error'))
    model_ridge.fit(X_poly, y)
    y_pred_ridge = model_ridge.predict(X_poly)
    mse_train_pr = np.mean((y_pred_ridge - y) ** 2)
    x_plt = np.linspace(0, 1, num=1000)
    y_pred_plt = model_ridge.predict(my_model.my_transform(x_plt.reshape(-1, 1), degree=degree))
    
    # plt.figure(figsize=(8, 6))
    plt.plot(x_plt, y_pred_plt, color=colors[i], label=f'Polynomial Regression (Degree {degree}, lambda={l:.6f})')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.title(f'Polynomial Regression (Degree 14, lambda={l:.6f})')


    print(f'\nPolynomial Regression (Degree 14, lambda={l:.6f})')
    print(f'Model Parameters: {model_ridge.coef_}')
    print(f'Training Error (MSE): {mse_train_pr:.3f}')
    print(f'CV Error (MSE): {mse_cv_ridge:.3f}')
plt.plot(x_gt, y_gt, color='black', label='Ground Truth')
plt.legend()
plt.show()