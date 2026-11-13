import numpy as np
import pandas as pd


def spearman(rank1, rank2):
    """Calculate Spearman's rank correlation for two complete rankings."""
    first = np.asarray(rank1, dtype=float)
    second = np.asarray(rank2, dtype=float)

    if first.ndim != 1 or second.ndim != 1 or first.size < 2 \
            or first.size != second.size:
        raise ValueError("Les deux classements doivent avoir la même taille (>= 2).")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Les classements doivent contenir des rangs finis.")
    expected = np.arange(1, first.size + 1)
    if not np.array_equal(np.sort(first), expected) \
            or not np.array_equal(np.sort(second), expected):
        raise ValueError("Chaque classement doit être une permutation des rangs 1 à n.")

    differences = first - second
    n = first.size
    return float(1 - 6 * np.sum(differences ** 2) / (n * (n ** 2 - 1)))


def spearman_matrix(ranks_dict):
    """Return pairwise Spearman correlations for named rankings."""
    if not ranks_dict:
        raise ValueError("Au moins un classement est requis.")

    names = list(ranks_dict)
    for name in names:
        spearman(ranks_dict[name], ranks_dict[name])

    matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            value = spearman(ranks_dict[first], ranks_dict[second])
            matrix.loc[first, second] = value
            matrix.loc[second, first] = value
    return matrix
