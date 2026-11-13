import numpy as np


FUZZY_SCALE = {
    "Très mauvais (VP)": (0, 0, 1),
    "Mauvais (P)": (0, 1, 3),
    "Moyennement mauvais (MP)": (1, 3, 5),
    "Passable (F)": (3, 5, 7),
    "Moyennement bon (MG)": (5, 7, 9),
    "Bon (G)": (7, 9, 10),
    "Très bon (VG)": (9, 10, 10),
}

FUZZY_WEIGHT_SCALE = {
    "Très faible (VL)": (0.0, 0.0, 0.1),
    "Faible (L)": (0.0, 0.1, 0.3),
    "Moyennement faible (ML)": (0.1, 0.3, 0.5),
    "Moyen (M)": (0.3, 0.5, 0.7),
    "Moyennement élevé (MH)": (0.5, 0.7, 0.9),
    "Élevé (H)": (0.7, 0.9, 1.0),
    "Très élevé (VH)": (0.9, 1.0, 1.0),
}


def linguistic_to_tfn(label, scale=None):
    scale = FUZZY_SCALE if scale is None else scale
    if label not in scale:
        raise ValueError(f"Évaluation inconnue : {label}")
    return scale[label]


def fuzzy_matrix_from_labels(labels, scale=None):
    """Convert linguistic labels to triangular fuzzy numbers."""
    labels = np.asarray(labels, dtype=object)
    if labels.ndim not in (1, 2):
        raise ValueError("Les évaluations linguistiques doivent être en 1D ou 2D.")

    fuzzy = np.zeros((*labels.shape, 3), dtype=float)
    for index in np.ndindex(labels.shape):
        fuzzy[index] = linguistic_to_tfn(labels[index], scale)
    return fuzzy


def aggregate_fuzzy(arrays):
    """Arithmetic aggregation of triangular evaluations from decision makers."""
    if not arrays:
        raise ValueError("Au moins une matrice de décideur est requise.")
    values = [np.asarray(array, dtype=float) for array in arrays]
    if any(not np.all(np.isfinite(value)) for value in values):
        raise ValueError("Les évaluations floues doivent être finies.")
    try:
        result = np.mean(np.stack(values), axis=0)
    except ValueError as error:
        raise ValueError("Les matrices des décideurs doivent avoir la même forme.") from error
    if result.shape[-1] != 3 or np.any(result < 0) \
            or np.any(result[..., 0] > result[..., 1]) \
            or np.any(result[..., 1] > result[..., 2]):
        raise ValueError("Les TFN agrégés doivent respecter 0 <= l <= m <= u.")
    return result


def defuzzify_tfn(fuzzy_matrix):
    """Centroid defuzzification for triangular fuzzy numbers."""
    fuzzy_matrix = np.asarray(fuzzy_matrix, dtype=float)
    if fuzzy_matrix.ndim < 1 or fuzzy_matrix.shape[-1] != 3:
        raise ValueError("Chaque nombre flou doit avoir 3 composantes.")
    if not np.all(np.isfinite(fuzzy_matrix)):
        raise ValueError("Les évaluations floues doivent être finies.")
    return fuzzy_matrix.mean(axis=-1)


def normalize_fuzzy_matrix(fuzzy_matrix, criteria_types):
    """Normalize triangular fuzzy evaluations following Chen (2000)."""
    matrix = np.asarray(fuzzy_matrix, dtype=float).copy()

    if matrix.ndim != 3 or matrix.shape[2] != 3:
        raise ValueError("La matrice doit avoir la forme (m, n, 3).")
    _, n, _ = matrix.shape
    if len(criteria_types) != n:
        raise ValueError("Nombre de types de critères incorrect.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Valeurs floues invalides.")
    if np.any(matrix < 0) or np.any(matrix[:, :, 0] > matrix[:, :, 1]) \
            or np.any(matrix[:, :, 1] > matrix[:, :, 2]):
        raise ValueError("Les TFN doivent respecter 0 <= l <= m <= u.")

    normalized = np.zeros_like(matrix)
    for j, criterion_type in enumerate(criteria_types):
        if criterion_type == "+":
            maximum = np.max(matrix[:, j, 2])
            normalized[:, j, :] = 0 if maximum == 0 else matrix[:, j, :] / maximum
        elif criterion_type == "-":
            column = matrix[:, j, :].copy()
            if np.any(column[:, 0] <= 0):
                column += 1e-6
            minimum_lower = np.min(column[:, 0])
            normalized[:, j, 0] = minimum_lower / column[:, 2]
            normalized[:, j, 1] = minimum_lower / column[:, 1]
            normalized[:, j, 2] = minimum_lower / column[:, 0]
        else:
            raise ValueError("Le type doit être '+' ou '-'.")

    return normalized


def fuzzy_topsis(fuzzy_matrix, weights, criteria_types):
    """Fuzzy TOPSIS using crisp or triangular fuzzy criterion weights."""
    matrix = np.asarray(fuzzy_matrix, dtype=float)
    criterion_weights = np.asarray(weights, dtype=float)

    if matrix.ndim != 3 or matrix.shape[2] != 3:
        raise ValueError("Forme attendue : (alternatives, critères, 3).")
    m, n, _ = matrix.shape
    if m < 2 or n < 2:
        raise ValueError("Il faut au moins 2 alternatives et 2 critères.")
    if criterion_weights.shape == (n,):
        if not np.all(np.isfinite(criterion_weights)) \
                or np.any(criterion_weights < 0) \
                or not np.isclose(criterion_weights.sum(), 1):
            raise ValueError("Les poids doivent être positifs et sommer à 1.")
        criterion_weights = np.repeat(criterion_weights[:, None], 3, axis=1)
    elif criterion_weights.shape == (n, 3):
        if not np.all(np.isfinite(criterion_weights)) \
                or np.any(criterion_weights < 0) \
                or np.any(criterion_weights[:, 0] > criterion_weights[:, 1]) \
                or np.any(criterion_weights[:, 1] > criterion_weights[:, 2]):
            raise ValueError("Les poids flous doivent respecter 0 <= l <= m <= u.")
    else:
        raise ValueError("Poids : forme (n,) ou (n, 3) attendue.")

    normalized = normalize_fuzzy_matrix(matrix, criteria_types)
    weighted = normalized * criterion_weights[None, :, :]

    distance_positive = np.sqrt(np.mean((weighted - 1.0) ** 2, axis=2)).sum(axis=1)
    distance_negative = np.sqrt(np.mean(weighted ** 2, axis=2)).sum(axis=1)
    denominator = distance_positive + distance_negative
    scores = np.divide(
        distance_negative,
        denominator,
        out=np.zeros_like(distance_negative),
        where=denominator > 0,
    )
    return scores, np.argsort(-scores) + 1
