import io

import numpy as np
import pandas as pd


def excel_sheet_names(uploaded):
    uploaded.seek(0)
    return pd.ExcelFile(uploaded).sheet_names


def read_raw_table(uploaded, sheet=None):
    """Lit un CSV (séparateur détecté) ou une feuille Excel, tout en texte."""
    if uploaded.name.lower().endswith(".csv"):
        content = uploaded.getvalue()
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode("latin-1")
        try:
            raw = pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype=str)
        except Exception:
            raw = pd.read_csv(io.StringIO(text), dtype=str)
    else:
        uploaded.seek(0)
        raw = pd.read_excel(uploaded, sheet_name=sheet or 0, dtype=str)

    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    raw.columns = [str(c).strip() for c in raw.columns]

    if raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError("Le fichier est vide.")
    return raw


def parse_ratio(cell):
    """Nombre, décimale avec virgule ou fraction (1/3). Vide -> NaN."""
    text = str(cell).replace("\u00a0", "").replace(" ", "").strip()
    if text == "" or text.lower() == "nan":
        return np.nan
    text = text.replace(",", ".")
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            return float(numerator) / float(denominator)
        return float(text)
    except (ValueError, ZeroDivisionError):
        return np.nan


def read_ahp_file(uploaded, sheet, criterion_names):
    """
    Matrice AHP n×n (n = nombre de critères).
      - 1re ligne : noms des critères ; 1re colonne : noms (optionnelle)
      - seule la partie au-dessus de la diagonale est lue (1, 3, 1/3, ...) ;
        le reste est recalculé par réciprocité.
    Si les noms correspondent aux critères, l'ordre est réaligné dessus.
    """
    n = len(criterion_names)
    raw = read_raw_table(uploaded, sheet)

    def is_text(cell):
        return (
            isinstance(cell, str) and cell.strip() != ""
            and np.isnan(parse_ratio(cell))
        )

    has_row_names = any(is_text(c) for c in raw.iloc[:, 0])
    if has_row_names:
        row_names = [str(c).strip() for c in raw.iloc[:, 0]]
        data = raw.iloc[:, 1:]
    else:
        row_names = None
        data = raw

    values = data.apply(lambda col: col.map(parse_ratio)).to_numpy(dtype=float)
    if values.shape != (n, n):
        raise ValueError(
            f"La matrice doit être {n}×{n} (nombre de critères) ; "
            f"le fichier contient {values.shape[0]}×{values.shape[1]}."
        )

    col_names = [str(c).strip() for c in data.columns]
    note = ""
    if set(col_names) == set(criterion_names):
        col_perm = [col_names.index(c) for c in criterion_names]
        if row_names is not None:
            if set(row_names) != set(criterion_names):
                raise ValueError(
                    "Les noms des lignes ne correspondent pas à ceux des colonnes."
                )
            row_perm = [row_names.index(c) for c in criterion_names]
        else:
            row_perm = col_perm
        values = values[np.ix_(row_perm, col_perm)]
    else:
        note = (
            "Les noms du fichier diffèrent des critères : "
            "l'ordre du fichier a été utilisé."
        )

    upper = np.triu_indices(n, 1)
    upper_values = values[upper]
    if np.isnan(upper_values).any() or np.any(upper_values <= 0):
        bad = np.where(np.isnan(upper_values) | (upper_values <= 0))[0][0]
        raise ValueError(
            f"Valeur manquante ou invalide au-dessus de la diagonale "
            f"({criterion_names[upper[0][bad]]} / {criterion_names[upper[1][bad]]})."
        )

    matrix = np.ones((n, n))
    matrix[upper] = upper_values
    matrix[(upper[1], upper[0])] = 1.0 / upper_values
    return {"values": matrix, "note": note}


def read_bwm_file(uploaded, sheet, criterion_names):
    """
    Fichier BWM : colonnes « Critère » (optionnelle), « BO » et « OW »,
    une ligne par critère. Valeurs entre 1 et 9.
    """
    n = len(criterion_names)
    raw = read_raw_table(uploaded, sheet)
    columns = list(raw.columns)

    def find(prefixes, exact):
        for c in columns:
            low = c.lower()
            if low in exact or low.startswith(prefixes):
                return c
        return None

    bo_col = find(("best",), ("bo", "best-to-others", "best to others"))
    ow_col = find(("others",), ("ow", "others-to-worst", "others to worst"))
    if bo_col is None or ow_col is None or bo_col == ow_col:
        if len(columns) < 2:
            raise ValueError("Colonnes « BO » et « OW » introuvables.")
        bo_col, ow_col = columns[-2], columns[-1]

    name_cols = [c for c in columns if c not in (bo_col, ow_col)]

    bo = raw[bo_col].map(parse_ratio).to_numpy(dtype=float)
    ow = raw[ow_col].map(parse_ratio).to_numpy(dtype=float)
    for label, vector in (("BO", bo), ("OW", ow)):
        if np.isnan(vector).any():
            raise ValueError(f"Valeur manquante ou non numérique dans la colonne {label}.")
        if np.any(vector < 1) or np.any(vector > 9):
            raise ValueError(f"Les valeurs de {label} doivent être comprises entre 1 et 9.")

    if len(bo) != n:
        raise ValueError(
            f"Le fichier contient {len(bo)} lignes ; {n} critères attendus."
        )

    note = ""
    if name_cols:
        names = [str(c).strip() for c in raw[name_cols[0]]]
        if set(names) == set(criterion_names):
            perm = [names.index(c) for c in criterion_names]
            bo, ow = bo[perm], ow[perm]
        else:
            note = (
                "Les noms du fichier diffèrent des critères : "
                "l'ordre du fichier a été utilisé."
            )
    return {"bo": [float(v) for v in bo], "ow": [float(v) for v in ow], "note": note}


def read_decision_file(uploaded, sheet=None, labels=None):
    """
    Lit un fichier CSV ou Excel.
    Format attendu :
      - 1re ligne : noms des critères
      - 1re colonne (optionnelle, texte) : noms des alternatives
      - le reste : valeurs numériques (mode Crisp) ou appréciations
        linguistiques (mode Fuzzy, `labels` = liste des termes de l'échelle).
        En mode Fuzzy, un numéro (1, 2, 3...) désigne le terme de même rang.
    """
    raw = read_raw_table(uploaded, sheet)

    def to_num(series):
        cleaned = (
            series.astype(str)
            .str.replace("\u00a0", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        return pd.to_numeric(cleaned, errors="coerce")

    if labels is None:
        # ---------- Crisp : valeurs numériques ----------
        first_col = raw.iloc[:, 0]
        if to_num(first_col).isna().any():
            alternatives = first_col.astype(str).str.strip().tolist()
            data = raw.iloc[:, 1:]
        else:
            alternatives = [f"A{i + 1}" for i in range(len(raw))]
            data = raw

        values = data.apply(to_num)

        if values.isna().any().any():
            row, col = np.argwhere(values.isna().to_numpy())[0]
            raise ValueError(
                f"Valeur manquante ou non numérique (ligne {row + 2}, "
                f"colonne « {data.columns[col]} »)."
            )
        values_array = values.to_numpy(dtype=float)
    else:
        # ---------- Fuzzy : appréciations linguistiques ----------
        lookup = {str(l).strip().lower(): l for l in labels}

        def to_label(cell):
            text = str(cell).strip()
            if text.lower() in lookup:
                return lookup[text.lower()]
            try:
                rank = float(text.replace(",", "."))
                if rank.is_integer() and 1 <= int(rank) <= len(labels):
                    return labels[int(rank) - 1]
            except ValueError:
                pass
            return None

        first_is_labels = all(to_label(c) is not None for c in raw.iloc[:, 0])
        if first_is_labels:
            alternatives = [f"A{i + 1}" for i in range(len(raw))]
            data = raw
        else:
            alternatives = raw.iloc[:, 0].astype(str).str.strip().tolist()
            data = raw.iloc[:, 1:]

        converted = data.apply(lambda col: col.map(to_label))
        if converted.isna().any().any():
            row, col = np.argwhere(converted.isna().to_numpy())[0]
            raise ValueError(
                f"Appréciation inconnue « {data.iloc[row, col]} » "
                f"(ligne {row + 2}, colonne « {data.columns[col]} »). "
                f"Termes valides : {', '.join(str(l) for l in labels)}."
            )
        values_array = converted.to_numpy(dtype=object)

    criteria = [str(c) for c in data.columns]

    if values_array.shape[0] < 2 or values_array.shape[1] < 2:
        raise ValueError("La matrice doit avoir au moins 2 alternatives et 2 critères.")
    if len(set(alternatives)) != len(alternatives):
        raise ValueError("Les noms des alternatives doivent être uniques.")
    if len(set(criteria)) != len(criteria):
        raise ValueError("Les noms des critères doivent être uniques.")

    return {
        "alternatives": alternatives,
        "criteria": criteria,
        "values": values_array,
    }
