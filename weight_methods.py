import numpy as np
from scipy.optimize import linprog


# METHODES DE PONDERATION


def ahp(comparaison_matrix):
    comparaison_matrix = np.asarray(comparaison_matrix, dtype=float)

    if comparaison_matrix.ndim != 2:
        raise ValueError("La matrice AHP doit être une matrice 2D.")

    if comparaison_matrix.shape[0] != comparaison_matrix.shape[1]:
        raise ValueError("La matrice AHP doit être carrée.")

    if np.any(comparaison_matrix <= 0):
        raise ValueError("Les valeurs de la matrice AHP doivent être strictement positives.")

    sum_col = np.sum(comparaison_matrix, axis=0)

    if np.any(sum_col == 0):
        raise ValueError("Une colonne a une somme nulle.")

    normalized_matrix = comparaison_matrix / sum_col
    weights = np.mean(normalized_matrix, axis=1)

    return weights


def bwm(bo, ow):
    bo = np.asarray(bo, dtype=float)
    ow = np.asarray(ow, dtype=float)

    if len(bo) != len(ow):
        raise ValueError("BO et OW doivent avoir la même longueur.")

    if len(bo) < 2:
        raise ValueError("Il faut au moins 2 critères.")

    if np.any(bo < 1) or np.any(ow < 1):
        raise ValueError("Les valeurs BO et OW doivent être comprises entre 1 et 9.")

    n = len(bo)

    B = np.argmin(bo)
    W = np.argmin(ow)

    if not np.isclose(bo[B], 1):
        raise ValueError("BO doit contenir 1 pour le critère Best.")

    if not np.isclose(ow[W], 1):
        raise ValueError("OW doit contenir 1 pour le critère Worst.")

    c = np.zeros(n + 1)
    c[-1] = 1

    A_ub = []
    b_ub = []

    for j in range(n):
        row1 = np.zeros(n + 1)
        row1[B] = 1
        row1[j] -= bo[j]
        row1[-1] = -1
        A_ub.append(row1)
        b_ub.append(0)

        row2 = np.zeros(n + 1)
        row2[B] = -1
        row2[j] += bo[j]
        row2[-1] = -1
        A_ub.append(row2)
        b_ub.append(0)

    for j in range(n):
        row1 = np.zeros(n + 1)
        row1[j] = 1
        row1[W] -= ow[j]
        row1[-1] = -1
        A_ub.append(row1)
        b_ub.append(0)

        row2 = np.zeros(n + 1)
        row2[j] = -1
        row2[W] += ow[j]
        row2[-1] = -1
        A_ub.append(row2)
        b_ub.append(0)

    A_eq = [np.r_[np.ones(n), 0]]
    b_eq = [1]

    bounds = [(0, None)] * (n + 1)

    result = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs"
    )

    if not result.success:
        raise ValueError("Optimisation BWM échouée : " + result.message)

    return result.x[:n], result.x[-1]


def entropy(D):
    D = np.asarray(D, dtype=float)

    if np.any(D < 0):
        raise ValueError("La matrice d'entropie ne doit pas contenir de valeurs négatives.")

    m, n = D.shape

    if m < 2:
        raise ValueError("Il faut au moins 2 alternatives.")

    column_sums = D.sum(axis=0)

    if np.any(column_sums == 0):
        raise ValueError("Une colonne a une somme nulle.")

    P = D / column_sums
    P_safe = np.where(P == 0, 1, P)

    k = 1 / np.log(m)
    E = -k * np.sum(P * np.log(P_safe), axis=0)

    diversification = 1 - E

    if np.isclose(diversification.sum(), 0):
        weights = np.ones(n) / n
    else:
        weights = diversification / diversification.sum()

    return weights


def critic(X, criteria_types):
    X = np.asarray(X, dtype=float)
    m, n = X.shape

    if len(criteria_types) != n:
        raise ValueError("Le nombre de types doit correspondre au nombre de critères.")

    R = np.zeros_like(X)

    for j in range(n):
        xmin = np.min(X[:, j])
        xmax = np.max(X[:, j])

        if xmax == xmin:
            R[:, j] = 0
            continue

        if criteria_types[j] == "+":
            R[:, j] = (X[:, j] - xmin) / (xmax - xmin)
        elif criteria_types[j] == "-":
            R[:, j] = (xmax - X[:, j]) / (xmax - xmin)
        else:
            raise ValueError("Le type doit être '+' ou '-'.")

    correlation = np.corrcoef(R, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0)
    np.fill_diagonal(correlation, 1.0)

    std = np.std(R, axis=0, ddof=1)

    C = np.zeros(n)
    for j in range(n):
        independence = np.sum(1 - np.abs(correlation[j, :]))
        C[j] = std[j] * independence

    if np.isclose(C.sum(), 0):
        weights = np.ones(n) / n
    else:
        weights = C / C.sum()

    return weights
