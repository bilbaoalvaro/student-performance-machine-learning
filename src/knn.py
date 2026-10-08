#Estas funciones las he cogido tal cual de la práctica de KNN 2:

import numpy as np

def minkowski_distance(a, b, p=2):
    dist = 0
    for i in range(len(a)):
        dist += (abs(a[i] - b[i])) ** p
    return float(dist ** (1 / p))


class knn:
    def __init__(self):
        self.k = None
        self.p = None
        self.x_train = None
        self.y_train = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, k: int = 5, p: int = 2):
        if len(X_train) != len(y_train):
            raise ValueError("Length of X_train and y_train must be equal.")
        
        if not isinstance(k, int) or not isinstance(p, int) or not (k > 0) or not (p > 0):
            raise ValueError("k and p must be positive integers.")
        
        self.k = k
        self.p = p
        self.x_train = X_train
        self.y_train = y_train

    def predict(self, X: np.ndarray) -> np.ndarray:
        resultado = []
        for x in X:
            distancias = self.compute_distances(x)
            indice_vecinos = self.get_k_nearest_neighbors(distancias)
            knn_labels = self.y_train[indice_vecinos]
            clase = self.most_common_label(knn_labels)
            resultado.append(clase)
        return np.array(resultado)

    def predict_proba(self, X):
        clases = np.unique(self.y_train)
        resultado = []

        for x in X:
            distancias = self.compute_distances(x)
            indice_vecinos = self.get_k_nearest_neighbors(distancias)
            knn_labels = self.y_train[indice_vecinos]

            counts = {}
            for clase in knn_labels:
                if clase in counts:
                    counts[clase] += 1
                else:
                    counts[clase] = 1

            fila = []
            for elemento in clases:
                fila.append(counts.get(elemento, 0) / self.k)

            resultado.append(fila)

        return np.array(resultado)

    def compute_distances(self, point: np.ndarray) -> np.ndarray:
        lista_dist = []
        for i in range(len(self.x_train)):
            dist = minkowski_distance(self.x_train[i], point, self.p)
            lista_dist.append(dist)
        return np.array(lista_dist)

    def get_k_nearest_neighbors(self, distances: np.ndarray) -> np.ndarray:
        pares = []

        for i in range(len(distances)):
            pares.append((distances[i], i))

        pares_ordenados = sorted(pares, key=lambda t: t[0])

        vecinos = pares_ordenados[:self.k]

        indices = []
        for vecino in vecinos:
            indices.append(vecino[1])

        return np.array(indices)

    def most_common_label(self, knn_labels: np.ndarray) -> int:
        diccionario = {}
        for clase in knn_labels:
            if clase in diccionario:
                diccionario[clase] += 1
            else:
                diccionario[clase] = 1

        maximo = 0
        clase_guardada = None

        for clase in diccionario:
            if diccionario[clase] > maximo:
                maximo = diccionario[clase]
                clase_guardada = clase
            elif diccionario[clase] == maximo and clase_guardada is not None and clase < clase_guardada:
                clase_guardada = clase

        return clase_guardada

    def __str__(self):
        return f"kNN model (k={self.k}, p={self.p})"
    
    