import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import matplotlib
import my_model
matplotlib.use('TkAgg')

# Task 1: Generate data
np.random.seed(48763)

# Task 4 data
x_full = np.linspace(0, 1, num=100)
y_full = np.sin(2 * np.pi * x_full) + np.random.normal(loc=0, scale=0.2, size=100)

# Task 5 degrees
degree = 14
colors = ['red', 'green', 'blue']

# m is the number of train data points
for i, m in enumerate([10, 80, 320]):
    x = np.linspace(0, 1, num=m)
    np.random.shuffle(x)
    y = np.sin(2 * np.pi * x) + np.random.normal(loc=0, scale=0.2, size=m)

    # Task 5: Perform Polynomial Regression
    X_poly = my_model.my_transform(x.reshape(-1, 1), degree=degree)
    model_pr = my_model.MyLinearRegression()
    model_pr.fit(X_poly, y)
    y_pred_pr = model_pr.predict(X_poly)
    mse_cv_pr = -np.mean(cross_val_score(model_pr, X_poly, y, cv=5, scoring='neg_mean_squared_error'))
    x_plt = np.linspace(0, 1, num=1000)
    y_pred_plt = model_pr.predict(my_model.my_transform(x_plt.reshape(-1, 1), degree=degree))
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label='Data', color=colors[i])
    
    # plt.plot(x, y_pred_pr, color=colors[i], label=f'Polynomial Regression (Degree {degree})')
    plt.plot(x_plt, y_pred_plt, color=colors[i], label=f'Polynomial Regression (Degree {degree})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Polynomial Regression (m={m})')
    plt.legend()
    plt.show()
    
    print(f'\nPolynomial Regression (m={m})')
    print(f'Model Parameters: {model_pr.coef_}')
    print(f'CV Error (MSE): {mse_cv_pr:.3f}')
    print(f'Training Error (MSE): {np.mean((y_pred_pr - y) ** 2):.3f}')