import numpy as np


# OUTILS COMMUNS

def validate_weights(weights):
    weights = np.asarray(weights, dtype=float)

    if np.any(weights < 0):
        raise ValueError("Les poids doivent être positifs ou nuls.")

    if not np.isclose(weights.sum(), 1, atol=1e-6):
        raise ValueError("La somme des poids doit être égale à 1.")

    return weights


def validate_matrix(X):
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("La matrice doit être 2D.")

    if X.shape[0] < 2 or X.shape[1] < 2:
        raise ValueError("Il faut au moins 2 alternatives et 2 critères.")

    if not np.all(np.isfinite(X)):
        raise ValueError("La matrice contient des valeurs invalides.")

    return X

# METHODES DE CLASSEMENT

def waspas(X, weights, criteria_types, lam=0.5):
    X = validate_matrix(X)
    weights = validate_weights(weights)

    m, n = X.shape

    if len(criteria_types) != n:
        raise ValueError("Le nombre de types doit correspondre au nombre de critères.")

    if not 0 <= lam <= 1:
        raise ValueError("Lambda doit être compris entre 0 et 1.")

    R = np.zeros_like(X)

    for j in range(n):
        xmin = np.min(X[:, j])
        xmax = np.max(X[:, j])

        if xmax == xmin:
            R[:, j] = 1
        elif criteria_types[j] == "+":
            R[:, j] = X[:, j] / xmax
        elif criteria_types[j] == "-":
            if np.any(X[:, j] <= 0):
                raise ValueError(
                    "WASPAS nécessite des valeurs strictement positives "
                    "pour les critères négatifs."
                )
            R[:, j] = xmin / X[:, j]
        else:
            raise ValueError("Le type doit être '+' ou '-'.")

    Q1 = np.sum(R * weights, axis=1)
    Q2 = np.prod(R ** weights, axis=1)
    Q = lam * Q1 + (1 - lam) * Q2

    ranking = np.argsort(-Q) + 1

    return Q, ranking


def topsis(X, weights, criteria_types):
    X = validate_matrix(X)
    weights = validate_weights(weights)

    m, n = X.shape

    if len(criteria_types) != n:
        raise ValueError("Le nombre de types doit correspondre au nombre de critères.")

    denominator = np.sqrt(np.sum(X ** 2, axis=0))

    if np.any(denominator == 0):
        raise ValueError("Une colonne contient uniquement des zéros.")

    R = X / denominator
    V = R * weights

    ideal_positive = np.zeros(n)
    ideal_negative = np.zeros(n)

    for j in range(n):
        if criteria_types[j] == "+":
            ideal_positive[j] = np.max(V[:, j])
            ideal_negative[j] = np.min(V[:, j])
        elif criteria_types[j] == "-":
            ideal_positive[j] = np.min(V[:, j])
            ideal_negative[j] = np.max(V[:, j])
        else:
            raise ValueError("Le type doit être '+' ou '-'.")

    S_plus = np.sqrt(np.sum((V - ideal_positive) ** 2, axis=1))
    S_minus = np.sqrt(np.sum((V - ideal_negative) ** 2, axis=1))

    denominator_rc = S_plus + S_minus

    RC = np.divide(
        S_minus,
        denominator_rc,
        out=np.zeros_like(S_minus),
        where=denominator_rc != 0
    )

    ranking = np.argsort(-RC) + 1

    return RC, ranking


def vikor(X, weights, criteria_types, v=0.5):
    X = validate_matrix(X)
    weights = validate_weights(weights)

    m, n = X.shape

    if len(criteria_types) != n:
        raise ValueError(
            "Le nombre de types doit correspondre au nombre de critères."
        )

    if not 0 <= v <= 1:
        raise ValueError("v doit être compris entre 0 et 1.")

    # 1. Déterminer les meilleures et pires valeurs
    best_values = np.zeros(n)
    worst_values = np.zeros(n)

    for j in range(n):
        if criteria_types[j] == "+":
            # Critère bénéfice : plus grand = meilleur
            best_values[j] = np.max(X[:, j])
            worst_values[j] = np.min(X[:, j])

        elif criteria_types[j] == "-":
            # Critère coût : plus petit = meilleur
            best_values[j] = np.min(X[:, j])
            worst_values[j] = np.max(X[:, j])

        else:
            raise ValueError("Le type doit être '+' ou '-'.")

    # 2. Calcul du regret normalisé
    regret = np.zeros_like(X, dtype=float)

    for j in range(n):

        denominator = worst_values[j] - best_values[j]

        # Si toutes les valeurs sont identiques
        if np.isclose(denominator, 0):
            regret[:, j] = 0.0

        elif criteria_types[j] == "+":
            # Critère bénéfice
            regret[:, j] = (
                (best_values[j] - X[:, j])
                / denominator
            )

        else:
            # Critère coût
            regret[:, j] = (
                (X[:, j] - best_values[j])
                / denominator
            )

    # 3. Regret pondéré
    weighted_regret = regret * weights

    # 4. S : utilité globale
    S = np.sum(weighted_regret, axis=1)

    # 5. R : regret maximal
    R = np.max(weighted_regret, axis=1)

    # 6. Valeurs idéales et anti-idéales
    S_star = np.min(S)
    S_minus = np.max(S)

    R_star = np.min(R)
    R_minus = np.max(R)

    # 7. Normalisation de S
    if np.isclose(S_minus, S_star):
        S_part = np.zeros(m)
    else:
        S_part = (
            (S - S_star)
            / (S_minus - S_star)
        )

    # 8. Normalisation de R
    if np.isclose(R_minus, R_star):
        R_part = np.zeros(m)
    else:
        R_part = (
            (R - R_star)
            / (R_minus - R_star)
        )

    # 9. Score VIKOR Q
    Q = v * S_part + (1 - v) * R_part

    # 10. Classement : plus petit Q = meilleur
    ranking = np.argsort(Q) + 1

    return Q, ranking