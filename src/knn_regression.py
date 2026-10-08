import numpy as np

class KNNRegresion:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.k = None
        self.p = None

    def fit(self, X_train, y_train, k=3, p=2):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k
        self.p = p

    def distancia_minkowski(self, x1, x2):
        return np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)

    def predecir_una_instancia(self, x):
        distancias = []

        for i in range(len(self.X_train)):
            distancia = self.distancia_minkowski(x, self.X_train[i])
            distancias.append((distancia, self.y_train[i]))

        distancias.sort(key=lambda elemento: elemento[0])

        vecinos = distancias[:self.k]
        valores_vecinos = [vecino[1] for vecino in vecinos]

        return np.mean(valores_vecinos)

    def predict(self, X_test):
        predicciones = []

        for x in X_test:
            prediccion = self.predecir_una_instancia(x)
            predicciones.append(prediccion)

        return np.array(predicciones)