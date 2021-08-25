import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class RegresionLineal:

    def __init__(self):

        self.boston = datasets.load_boston()
        self.X = self.boston.data[:, np.newaxis, 7]
        self.y = self.boston.target

    def GraficaNormal(self):

        plt.scatter(self.X, self.y)
        plt.title('Regresión')
        plt.xlabel('Distancia')
        plt.ylabel('Precio')
        plt.show()

    def GraficaRegresion(self):

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.65)
        lr = linear_model.LinearRegression()
        lr.fit(X_train, y_train)
        Y_pred = lr.predict(X_test)

        plt.scatter(X_test, y_test)
        plt.plot(X_test, Y_pred, color='black', linewidth=1)
        plt.title('Regresión')
        plt.xlabel('Distancia')
        plt.ylabel('Precio')
        plt.show()

        print('Valor de la pendiente o coeficiente "a":', lr.coef_)

        print('Valor de la intersección o coeficiente "b":', lr.intercept_)

        print('La ecuación del modelo es igual a:')
        print('y = ', lr.coef_, 'x ', lr.intercept_)

        print('Precisión del modelo:')
        print(lr.score(X_train, y_train))

EL = RegresionLineal()
EL.GraficaNormal()
EL.GraficaRegresion()

