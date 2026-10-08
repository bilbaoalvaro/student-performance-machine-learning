import numpy as np

class LogisticRegressor:  #Códigos cogidos de la práctica 5
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self,X,y,learning_rate=0.01,num_iterations=1000,penalty=None,l1_ratio=0.5,C=1.0,verbose=False,print_every=100,):
        m, n = X.shape

        self.weights = np.zeros(n)
        self.bias = 0

        for i in range(num_iterations):
            z = self.bias + np.dot(X, self.weights)
            y_hat = self.sigmoid(z)

            loss = self.log_likelihood(y, y_hat)

            if verbose and i % print_every == 0:
                print(f"Iteration {i}: Loss {loss}")

            dw = (-1 / m) * np.dot((y - y_hat), X)
            db = -np.mean(y - y_hat)

            if penalty == "lasso":
                dw = self.lasso_regularization(dw, m, C)
            elif penalty == "ridge":
                dw = self.ridge_regularization(dw, m, C)
            elif penalty == "elasticnet":
                dw = self.elasticnet_regularization(dw, m, C, l1_ratio)

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict_proba(self, X):
        z = self.bias + np.dot(X, self.weights)
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        classification_result = []

        for p in probabilities:
            if float(p) >= threshold:
                classification_result.append(1)
            else:
                classification_result.append(0)

        return np.array(classification_result)

    @staticmethod
    def log_likelihood(y, y_hat):
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))


class LogisticRegressorMulticlase:
    def __init__(self):
        self.modelos = {}
        self.clases = None

    def fit(self,X,y,learning_rate=0.01,num_iterations=1000,penalty=None,l1_ratio=0.5,C=1.0,verbose=False,print_every=100,):
        self.clases = np.unique(y)
        self.modelos = {}

        for clase in self.clases:
            y_binario = np.where(y == clase, 1, 0)

            modelo = LogisticRegressor()
            modelo.fit(X,y_binario,learning_rate=learning_rate,num_iterations=num_iterations,penalty=penalty,l1_ratio=l1_ratio,C=C,verbose=verbose,print_every=print_every,)

            self.modelos[clase] = modelo

    def predict_proba(self, X):
        probabilidades = []

        for clase in self.clases:
            probs_clase = self.modelos[clase].predict_proba(X)
            probabilidades.append(probs_clase)

        probabilidades = np.column_stack(probabilidades)
        return probabilidades

    def predict(self, X):
        probabilidades = self.predict_proba(X)
        indices_maximos = np.argmax(probabilidades, axis=1)
        predicciones = self.clases[indices_maximos]
        return predicciones