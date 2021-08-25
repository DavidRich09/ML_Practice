import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn import datasets
from sklearn.model_selection import train_test_split


class VecSoporte:

    def __init__(self):
        self.boston = datasets.load_boston()
        
        self.X_svr = self.boston.data[:, np.newaxis, 5]
        self.y_svr = self.boston.target

        
        
        self.Graficar()

    def Graficar(self):
        plt.scatter(self.X_svr, self.y_svr)
        plt.title('Vectores de soporte de regresión')
        plt.xlabel('Distancia')
        plt.ylabel('Precio')
        plt.show()

    def EntrenarYPredecir(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X_svr, self.y_svr, test_size=0.65)

        svr = SVR(kernel='linear', C=1.0, epsilon=0.2)

        svr.fit(X_train, y_train)

        Y_pred = svr.predict(X_test)

        plt.scatter(X_test, y_test)
        plt.plot(X_test, Y_pred, color='red', linewidth=3)
        
        plt.title('Vectores de soporte de regresión')
        plt.xlabel('Distancia')
        plt.ylabel('Precio')
        plt.show()
        
        print('Precisión del modelo:')
        print(svr.score(X_train, y_train))
        

VSR = VecSoporte()
VSR.Graficar()
VSR.EntrenarYPredecir()
