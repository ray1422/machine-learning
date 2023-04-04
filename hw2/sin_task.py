import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use('TkAgg')

# Task 1: Generate data
np.random.seed(48763)
x = np.linspace(0, 1, num=15)
np.random.shuffle(x)
y = np.sin(2 * np.pi * x) + np.random.normal(loc=0, scale=0.2, size=15)



# Task 3: Perform Polynomial Regression
degrees = [1, 2, 5, 10]
colors = ['red', 'green', 'orange', 'purple']

for degree, color in zip(degrees, colors):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(x.reshape(-1, 1))
    model_pr = LinearRegression()
    model_pr.fit(X_poly, y)
    y_pred_pr = model_pr.predict(X_poly)
    mse_train_pr = np.mean((y_pred_pr - y) ** 2)
    cv_errs = cross_val_score(model_pr, X_poly, y, cv=5, scoring='neg_mean_squared_error')
    mse_cv_pr = -np.mean(cv_errs)
    x_plt = np.linspace(0, 1, num=1000)
    y_pred_plt = model_pr.predict(poly.fit_transform(x_plt.reshape(-1, 1)))
    plt.plot(x_plt, y_pred_plt, color=color, label=f'Polynomial Regression (Degree {degree})')
    
    print(f'\nPolynomial Regression (Degree {degree})')
    print(f'Model Parameters: {model_pr.coef_}')
    print(f'Training Error (MSE): {mse_train_pr:.3f}')
    print(f'CV Error (MSE): {mse_cv_pr:.3f}', cv_errs)
plt.scatter(x, y, label='Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.legend()
plt.show()