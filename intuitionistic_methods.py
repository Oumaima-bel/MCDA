import numpy as np


IF_SCALE = {
    "Très mauvais": (0.10, 0.90),
    "Mauvais": (0.25, 0.70),
    "Moyennement mauvais": (0.35, 0.60),
    "Passable": (0.50, 0.45),
    "Moyennement bon": (0.60, 0.30),
    "Bon": (0.75, 0.20),
    "Très bon": (0.90, 0.10),
}

IF_WEIGHT_SCALE = {
    "Très peu important": (0.10, 0.90),
    "Peu important": (0.35, 0.60),
    "Moyen": (0.50, 0.45),
    "Important": (0.75, 0.20),
    "Très important": (0.90, 0.10),
}


def _validate_if(values):
    values = np.asarray(values, dtype=float)
    if values.shape[-1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("Chaque évaluation intuitionniste doit contenir (mu, nu) finis.")
    if np.any(values < 0) or np.any(values.sum(axis=-1) > 1):
        raise ValueError("Les degrés intuitionnistes doivent respecter mu >= 0, nu >= 0, mu + nu <= 1.")
    return values


def if_matrix_from_labels(labels, scale):
    """Convert one- or two-dimensional linguistic labels to (mu, nu) pairs."""
    labels = np.asarray(labels, dtype=object)
    output = np.zeros((*labels.shape, 2), dtype=float)
    for index in np.ndindex(labels.shape):
        if labels[index] not in scale:
            raise ValueError(f"Évaluation inconnue : {labels[index]}")
        output[index] = scale[labels[index]]
    return _validate_if(output)


def ifwa(arrays, sigma=None):
    """Aggregate intuitionistic evaluations with IFWA decision-maker weights."""
    if not arrays:
        raise ValueError("Au moins une matrice de décideur est requise.")
    values = np.stack([_validate_if(array) for array in arrays])
    count = values.shape[0]

    weights = np.full(count, 1 / count) if sigma is None else np.asarray(sigma, dtype=float)
    if weights.shape != (count,) or not np.all(np.isfinite(weights)) \
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Les poids des décideurs doivent être positifs et sommer à 1.")
    weights = weights.reshape((count,) + (1,) * (values.ndim - 2))

    membership = 1 - np.prod((1 - values[..., 0]) ** weights, axis=0)
    non_membership = np.prod(values[..., 1] ** weights, axis=0)
    return _validate_if(np.stack([membership, non_membership], axis=-1))


def if_score(values):
    """Defuzzify intuitionistic numbers to scores in [0, 1]."""
    values = _validate_if(values)
    return (1 + values[..., 0] - values[..., 1]) / 2


def crisp_to_if_weights(weights):
    """Convert normalized crisp weights to intuitionistic (mu, nu) weights."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or not np.all(np.isfinite(weights)) \
            or np.any(weights < 0) or weights.max(initial=0) <= 0:
        raise ValueError("Les poids crisp doivent être un vecteur positif non nul.")
    membership = weights / weights.max()
    return np.stack([membership, 1 - membership], axis=-1)


def if_topsis(decision_matrix, weights, criteria_types):
    """Intuitionistic fuzzy TOPSIS with weighted decision-maker evaluations."""
    decision_matrix = _validate_if(decision_matrix)
    weights = _validate_if(weights)
    if decision_matrix.ndim != 3 or decision_matrix.shape[0] < 2 \
            or decision_matrix.shape[1] < 2:
        raise ValueError("D doit avoir la forme (alternatives >= 2, critères >= 2, 2).")

    _, n, _ = decision_matrix.shape
    if weights.shape != (n, 2) or len(criteria_types) != n:
        raise ValueError("Dimensions incohérentes entre D, W et les types de critères.")
    if any(kind not in ("+", "-") for kind in criteria_types):
        raise ValueError("Le type doit être '+' ou '-'.")

    membership = decision_matrix[:, :, 0] * weights[None, :, 0]
    non_membership = (
        decision_matrix[:, :, 1] + weights[None, :, 1]
        - decision_matrix[:, :, 1] * weights[None, :, 1]
    )
    hesitation = 1 - membership - non_membership
    if np.any(hesitation < -1e-12):
        raise ValueError("Le produit pondéré a produit une évaluation invalide.")
    hesitation = np.maximum(hesitation, 0)

    benefit = np.asarray([kind == "+" for kind in criteria_types])
    mu_positive = np.where(benefit, membership.max(axis=0), membership.min(axis=0))
    nu_positive = np.where(
        benefit, non_membership.min(axis=0), non_membership.max(axis=0)
    )
    mu_negative = np.where(benefit, membership.min(axis=0), membership.max(axis=0))
    nu_negative = np.where(
        benefit, non_membership.max(axis=0), non_membership.min(axis=0)
    )
    pi_positive = 1 - mu_positive - nu_positive
    pi_negative = 1 - mu_negative - nu_negative

    separation_positive = np.sqrt(
        (membership - mu_positive) ** 2
        + (non_membership - nu_positive) ** 2
        + (hesitation - pi_positive) ** 2
    ).sum(axis=1) / (2 * n)
    separation_negative = np.sqrt(
        (membership - mu_negative) ** 2
        + (non_membership - nu_negative) ** 2
        + (hesitation - pi_negative) ** 2
    ).sum(axis=1) / (2 * n)

    denominator = separation_positive + separation_negative
    scores = np.divide(
        separation_negative,
        denominator,
        out=np.zeros_like(separation_negative),
        where=denominator > 0,
    )
    return scores, np.argsort(-scores) + 1
