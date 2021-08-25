import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn import datasets
from sklearn.model_selection import train_test_split

boston = datasets.load_boston()

X_svr = boston.data[:, np.newaxis, 5]
y_svr = boston.target

plt.scatter(X_svr, y_svr)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_svr, y_svr, test_size=0.65)

svr = SVR(kernel='linear', C=1.0, epsilon=0.2)

svr.fit(X_train, y_train)

Y_pred = svr.predict(X_test)

plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred, color='red', linewidth=3)

plt.title('MODELO VECTORES DE SOPORTE REGRESIÓN')
plt.xlabel('Número de habitaciones')
plt.ylabel('Valor medio')
plt.show()

print()
print('DATOS DEL MODELO VECTORES DE SOPORTE REGRESIÓN')
print()
print('Precisión del modelo:')
print(svr.score(X_train, y_train))
