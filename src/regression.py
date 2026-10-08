import numpy as np

class LinearRegressor:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit_multiple(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n, p = X.shape

        Xb = np.ones((n, p + 1))
        Xb[:, 1:] = X

        theta = np.linalg.pinv(Xb.T @ Xb) @ (Xb.T @ y)

        self.intercept = theta[0]
        self.coefficients = theta[1:]

    def predict(self, X):
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        X = np.asarray(X, dtype=float)

        if np.ndim(X) == 1:
            return self.intercept + self.coefficients * X
        else:
            predictions = []
            for i in range(len(X)):
                suma = self.intercept
                for j in range(len(X[i])):
                    suma += X[i][j] * self.coefficients[j]
                predictions.append(suma)

        return np.array(predictions)


def evaluate_regression(y_true, y_pred):
    tss = 0
    for i in range(len(y_pred)):
        tss += (y_true[i] - np.mean(y_true)) ** 2

    rss = 0
    for i in range(len(y_pred)):
        rss += (y_true[i] - y_pred[i]) ** 2

    r_squared = 1 - (rss / tss)

    rmse = 0
    for i in range(len(y_pred)):
        rmse += (y_true[i] - y_pred[i]) ** 2
    rmse = (rmse / len(y_pred)) ** 0.5

    mae = 0
    for i in range(len(y_pred)):
        mae += abs(y_true[i] - y_pred[i])
    mae = mae / len(y_pred)

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}