import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use('TkAgg')

# Task 1: Generate data
np.random.seed(48763)
x = np.linspace(-3, 3, num=15)
np.random.shuffle(x)
y = 2 * x + np.random.normal(loc=0, scale=1, size=15)

# Task 3: Perform Polynomial Regression
degrees = [1, 5, 10, 14]
colors = ['blue', 'green', 'orange', 'purple']

plt.figure(figsize=(8, 6))
plt.scatter(x, y, label='Data')

for degree, color in zip(degrees, colors):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(x.reshape(-1, 1))
    model_pr = LinearRegression()
    model_pr.fit(X_poly, y)
    y_pred_pr = model_pr.predict(X_poly)
    mse_train_pr = np.mean((y_pred_pr - y) ** 2)
    mse_cv_pr = -np.mean(cross_val_score(model_pr, X_poly, y, cv=5, scoring='neg_mean_squared_error'))
    x_plt = np.linspace(-3, 3, num=1000)
    y_pred_plt = model_pr.predict(poly.fit_transform(x_plt.reshape(-1, 1)))
    plt.plot(x_plt, y_pred_plt, color=color, label=f'Polynomial Regression (Degree {degree})')
    print(f'\nPolynomial Regression (Degree {degree})')
    print(f'Model Parameters: {model_pr.coef_}')
    print(f'Training Error (MSE): {mse_train_pr:.3f}')
    print(f'CV Error (MSE): {mse_cv_pr:.3f}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.legend()
plt.show()