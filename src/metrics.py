import numpy as np

def accuracy_score_manual(y_real, y_pred):
    aciertos = 0

    for i in range(len(y_real)):
        if y_real[i] == y_pred[i]:
            aciertos += 1

    return aciertos / len(y_real)


def confusion_matrix_multiclase(y_real, y_pred):
    clases = np.unique(np.concatenate((y_real, y_pred)))

    matriz = np.zeros((len(clases), len(clases)), dtype=int)

    for i in range(len(y_real)):
        fila = None
        columna = None

        for j in range(len(clases)):
            if clases[j] == y_real[i]:
                fila = j
            if clases[j] == y_pred[i]:
                columna = j

        matriz[fila, columna] += 1

    return clases, matriz


def f1_macro_manual(y_real, y_pred):
    clases = np.unique(np.concatenate((y_real, y_pred)))

    lista_f1 = []

    for clase in clases:
        tp = 0
        fp = 0
        fn = 0

        for i in range(len(y_real)):
            if y_real[i] == clase and y_pred[i] == clase:
                tp += 1
            elif y_real[i] != clase and y_pred[i] == clase:
                fp += 1
            elif y_real[i] == clase and y_pred[i] != clase:
                fn += 1

        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        lista_f1.append(f1)

    return np.mean(lista_f1)

def mae_manual(y_real, y_pred):
    return np.mean(np.abs(y_real - y_pred))

def rmse_manual(y_real, y_pred):
    return np.sqrt(np.mean((y_real - y_pred) ** 2))

def r2_manual(y_real, y_pred):
    rss = np.sum((y_real - y_pred) ** 2)
    tss = np.sum((y_real - np.mean(y_real)) ** 2)
    return 1 - rss / tss
