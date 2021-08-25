import numpy as np

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt

boston = datasets.load_boston()

X_p = boston.data[:, np.newaxis, 5]
y_p = boston.target

plt.scatter(X_p, y_p)
plt.show()

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_p, y_p, test_size=0.65)

poli_reg = PolynomialFeatures(degree = 2)

X_train_poli = poli_reg.fit_transform(X_train_p)
X_test_poli = poli_reg.fit_transform(X_test_p)


pr = linear_model.LinearRegression()
pr.fit(X_train_poli, y_train_p)

Y_pred_pr = pr.predict(X_test_poli)

plt.scatter(X_test_p, y_test_p)
plt.plot(X_test_p, Y_pred_pr, color='red', linewidth=3)

plt.title('Regresión Polinomial')
plt.xlabel('Número de habitaciones')
plt.ylabel('Valor medio')
plt.show()
