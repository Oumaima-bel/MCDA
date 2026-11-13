import streamlit as st
import numpy as np
import pandas as pd


from classification_methods import (
    validate_weights, waspas, topsis, vikor, vikor_compromise
)
from weight_methods import ahp, ahp_consistency, bwm, entropy, critic
from fuzzy_methods import (
    FUZZY_SCALE, FUZZY_WEIGHT_SCALE, fuzzy_matrix_from_labels,
    aggregate_fuzzy, defuzzify_tfn, fuzzy_topsis
)
from intuitionistic_methods import (
    IF_SCALE, IF_WEIGHT_SCALE, if_matrix_from_labels, ifwa,
    if_score, if_topsis, crisp_to_if_weights
)
from robustness import spearman_matrix


# CONFIGURATION

st.set_page_config(
    page_title="MCDA - Pondération & Classement",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# THEME : couleurs douces (mauve, jaune, bleu, orange, rose)
# ==========================================

PALETTE_CSS = """
<style>
:root {
    --mauve:        #B9A7D9;
    --mauve-clair:  #EAE3F5;
    --mauve-fonce:  #6D5A93;
    --jaune:        #F6E7A8;
    --jaune-clair:  #FBF5D6;
    --bleu:         #A8C5E6;
    --bleu-clair:   #E3EDF8;
    --bleu-fonce:   #4C719C;
    --orange:       #F4C8A0;
    --orange-clair: #FBE9D8;
    --orange-fonce: #B8733A;
    --rose:         #F2C3D5;
    --rose-clair:   #FBE6EE;
    --rose-fonce:   #A8577A;
    --texte:        #3F3A4D;
}

/* Fond général */
.stApp {
    background: linear-gradient(180deg, #FAF7FD 0%, #F5F0FA 100%);
    color: var(--texte);
}

/* Barre latérale */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--mauve-clair) 0%, var(--rose-clair) 100%);
    border-right: 3px solid var(--mauve);
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--mauve-fonce);
}

/* Titres */
h1 {
    color: var(--mauve-fonce) !important;
    background: linear-gradient(90deg, var(--mauve-clair), var(--rose-clair));
    padding: 0.5rem 1rem !important;
    border-radius: 14px;
    border-left: 8px solid var(--mauve);
}
h2 {
    color: var(--bleu-fonce) !important;
    border-bottom: 3px solid var(--bleu);
    padding-bottom: 0.25rem !important;
}
h3 {
    color: var(--orange-fonce) !important;
    border-left: 5px solid var(--orange);
    padding-left: 0.6rem !important;
}

/* Cartes de métriques (λmax, CI, CR, ε*) */
div[data-testid="stMetric"] {
    background: var(--rose-clair);
    border: 2px solid var(--rose);
    border-radius: 14px;
    padding: 0.7rem 1rem;
}
div[data-testid="stMetricLabel"] { color: var(--rose-fonce); }
div[data-testid="stMetricValue"] { color: var(--mauve-fonce); }

/* Messages (info / succès / avertissement / erreur) */
div[data-testid="stAlert"] {
    border-radius: 12px;
    color: var(--texte);
}
div[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) {
    background-color: var(--bleu-clair);
    border-left: 6px solid var(--bleu);
}
div[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {
    background-color: var(--jaune-clair);
    border-left: 6px solid #E2CB6B;
}
div[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) {
    background-color: var(--orange-clair);
    border-left: 6px solid var(--orange);
}
div[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) {
    background-color: var(--rose-clair);
    border-left: 6px solid var(--rose);
}

/* Onglets (décideurs) */
button[data-baseweb="tab"] {
    background: var(--mauve-clair);
    border-radius: 10px 10px 0 0;
    margin-right: 4px;
    color: var(--mauve-fonce);
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: var(--mauve);
    color: #FFFFFF;
}
div[data-baseweb="tab-highlight"] {
    background-color: var(--orange) !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: var(--jaune-clair);
    border: 2px solid var(--jaune);
    border-radius: 12px;
}
div[data-testid="stExpander"] summary {
    color: var(--orange-fonce);
    font-weight: 600;
}

/* Tableaux / éditeurs de données */
div[data-testid="stDataFrame"],
div[data-testid="stDataEditor"] {
    border: 2px solid var(--bleu);
    border-radius: 12px;
    overflow: hidden;
    background: #FFFFFF;
}

/* Champs de saisie et listes déroulantes */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] {
    background-color: #FFFFFF;
    border-color: var(--mauve) !important;
    border-radius: 10px;
}

/* Boutons +/- des nombres */
button[data-testid="stNumberInputStepUp"],
button[data-testid="stNumberInputStepDown"] {
    background-color: var(--mauve-clair);
    color: var(--mauve-fonce);
}

/* Boutons radio */
div[role="radiogroup"] label {
    background: #FFFFFF;
    border: 1px solid var(--rose);
    border-radius: 10px;
    padding: 0.2rem 0.6rem;
    margin-bottom: 0.2rem;
}

/* Sliders */
div[data-baseweb="slider"] div[role="slider"] {
    background-color: var(--orange) !important;
    border-color: var(--orange-fonce) !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:first-child {
    background: linear-gradient(90deg, var(--bleu), var(--mauve)) !important;
}

/* Légendes */
div[data-testid="stCaptionContainer"] {
    color: var(--mauve-fonce);
}

/* Séparateurs */
hr {
    border-color: var(--mauve) !important;
}
</style>
"""

st.markdown(PALETTE_CSS, unsafe_allow_html=True)


st.title("📊 Application MCDA")


# INTERFACE

st.sidebar.header("⚙️ Paramètres")

n_alternatives = st.sidebar.number_input(
    "Nombre d'alternatives",
    min_value=2,
    max_value=20,
    value=4,
    step=1
)

n_criteria = st.sidebar.number_input(
    "Nombre de critères",
    min_value=2,
    max_value=15,
    value=4,
    step=1
)


decision_mode = st.sidebar.radio(
    "Nature des évaluations",
    ["Crisp", "Fuzzy"],#, "Fuzzy intuitionniste"
    help=(
        "Crisp : valeurs numériques précises. "
        "Fuzzy : variables linguistiques (TFN). "#"Fuzzy intuitionniste : couples (μ, ν)."
        
    )
)

decision_makers = 1
if decision_mode != "Crisp":
    decision_makers = int(st.sidebar.number_input(
        "Nombre de décideurs", min_value=1, max_value=10, value=1, step=1
    ))

ranking_options = ["WSM", "WPM", "WASPAS", "TOPSIS", "VIKOR"]

if decision_mode == "Fuzzy":
    ranking_options.append("Fuzzy TOPSIS")
# elif decision_mode == "Fuzzy intuitionniste":
#     ranking_options.append("IF-TOPSIS")

ranking_method = st.sidebar.selectbox(
    "Méthode de classement",
    ranking_options
)


st.sidebar.divider()

weight_options = ["AHP", "BWM", "Entropie", "CRITIC"]
if decision_mode != "Crisp":
    weight_options.append("Linguistique (flou)")

weight_method = st.sidebar.selectbox("Méthode de pondération", weight_options)


# NOMS

st.header("Définition du problème")

col1, col2 = st.columns(2)

with col1:
    alternative_names_text = st.text_input(
        "Noms des alternatives",
        value=", ".join([f"A{i+1}" for i in range(n_alternatives)]),
        help="Séparez les noms par des virgules."
    )

with col2:
    criterion_names_text = st.text_input(
        "Noms des critères",
        value=", ".join([f"C{i+1}" for i in range(n_criteria)]),
        help="Séparez les noms par des virgules."
    )

alternative_names = [
    x.strip() for x in alternative_names_text.split(",") if x.strip()
]

criterion_names = [
    x.strip() for x in criterion_names_text.split(",") if x.strip()
]

if len(alternative_names) != n_alternatives:
    st.warning("Le nombre de noms d'alternatives ne correspond pas au nombre choisi.")

if len(criterion_names) != n_criteria:
    st.warning("Le nombre de noms de critères ne correspond pas au nombre choisi.")


# TYPES DES CRITERES

st.subheader("Type des critères")

types_values = []

cols = st.columns(n_criteria)

for j, col in enumerate(cols):
    with col:
        criterion_type = st.selectbox(
            criterion_names[j] if j < len(criterion_names) else f"C{j+1}",
            ["+", "-"],
            format_func=lambda x: "Positif (+)" if x == "+" else "Négatif (-)",
            key=f"type_{j}"
        )
        types_values.append(criterion_type)

criteria_types = types_values


# ==========================================
# MATRICE DE DECISION
# ==========================================

st.header("Matrice de décision")

st.info(
    "Les lignes représentent les alternatives et "
    "les colonnes représentent les critères."
)

fuzzy_X = None
if_X = None


def label_table(options, key, index, columns, default):
    frame = pd.DataFrame(default, index=index, columns=columns)
    column_config = {
        name: st.column_config.SelectboxColumn(
            name, options=options, required=True
        )
        for name in columns
    }
    return st.data_editor(
        frame,
        column_config=column_config,
        use_container_width=True,
        num_rows="fixed",
        key=key,
    )

try:
    if decision_mode == "Crisp":
        X_df = pd.DataFrame(
            np.ones((n_alternatives, n_criteria)),
            index=alternative_names,
            columns=criterion_names,
        )
        edited_X = st.data_editor(
            X_df, use_container_width=True, num_rows="fixed",
            key="decision_matrix_crisp"
        )
        X = edited_X.to_numpy(dtype=float)

    else:
        scale = FUZZY_SCALE if decision_mode == "Fuzzy" else IF_SCALE
        options = list(scale)
        convert = (
            fuzzy_matrix_from_labels
            if decision_mode == "Fuzzy"
            else if_matrix_from_labels
        )
        st.caption(
            "Choisissez une appréciation linguistique par décideur, "
            "alternative et critère."
        )

        decision_matrices = []
        tabs = st.tabs([f"Décideur {index + 1}" for index in range(decision_makers)])
        for index, tab in enumerate(tabs):
            with tab:
                edited_labels = label_table(
                    options,
                    f"decision_matrix_{decision_mode}_{index}",
                    alternative_names,
                    criterion_names,
                    options[3],
                )
                decision_matrices.append(
                    convert(edited_labels.to_numpy(dtype=object), scale)
                )

        if decision_mode == "Fuzzy":
            fuzzy_X = aggregate_fuzzy(decision_matrices)
            X = defuzzify_tfn(fuzzy_X)
        else:
            if_X = ifwa(decision_matrices)
            X = if_score(if_X)

        with st.expander(
            "Afficher la matrice numérique défloutée"
        ):
            st.dataframe(
                pd.DataFrame(
                    X,
                    index=alternative_names,
                    columns=criterion_names
                ).round(4),
                use_container_width=True
            )

except Exception as e:
    st.error(f"Erreur de matrice de décision : {e}")
    st.stop()


# INPUT DE LA METHODE DE PONDERATION

st.header(" Pondération des critères")

weights = None
epsilon = None
fuzzy_W = None
if_W = None

try:

    # ---------------- AHP ----------------
    if weight_method == "AHP":
        st.subheader("Matrice de comparaison par paires — AHP")
        st.caption(
            "Renseignez la partie supérieure (échelle de Saaty, "
            "par exemple 1, 3 ou 1/3) ; la partie inférieure est réciproque."
        )
        ahp_df = pd.DataFrame(
            np.ones((n_criteria, n_criteria)),
            index=criterion_names,
            columns=criterion_names,
        )
        edited_ahp = st.data_editor(
            ahp_df, use_container_width=True, num_rows="fixed", key="ahp_matrix"
        )
        comparison_matrix = edited_ahp.to_numpy(dtype=float)
        upper_indices = np.triu_indices(n_criteria, 1)
        if not np.all(np.isfinite(comparison_matrix[upper_indices])) \
                or np.any(comparison_matrix[upper_indices] <= 0):
            raise ValueError("Les valeurs au-dessus de la diagonale doivent être finies et > 0.")
        for i in range(n_criteria):
            comparison_matrix[i, i] = 1.0
            for j in range(i + 1, n_criteria):
                comparison_matrix[j, i] = 1 / comparison_matrix[i, j]

        weights = ahp(comparison_matrix)
        lambda_max, ci, cr, consistent = ahp_consistency(comparison_matrix)
        col1, col2, col3 = st.columns(3)
        col1.metric("λmax", f"{lambda_max:.4f}")
        col2.metric("CI", f"{ci:.4f}")
        col3.metric("CR", f"{cr:.4f}")
        if consistent:
            st.success("Matrice consistante (CR < 0,1).")
        else:
            st.warning("Matrice non consistante (CR ≥ 0,1) : revoyez vos jugements.")

    # ---------------- BWM ----------------
    elif weight_method == "BWM":

        st.subheader("Comparaisons Best-to-Others et Others-to-Worst")

        st.write(
            "Entrez les valeurs de 1 à 9. "
            "Pour BO, le critère Best doit avoir 1. "
            "Pour OW, le critère Worst doit avoir 1."
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Best → Others (BO)**")

            bo = []

            for j in range(n_criteria):
                value = st.number_input(
                    criterion_names[j],
                    min_value=1.0,
                    max_value=9.0,
                    value=1.0 if j == 0 else 2.0,
                    step=1.0,
                    key=f"bo_{j}"
                )
                bo.append(value)

        with col2:
            st.markdown("**Others → Worst (OW)**")

            ow = []

            for j in range(n_criteria):
                value = st.number_input(
                    criterion_names[j],
                    min_value=1.0,
                    max_value=9.0,
                    value=1.0 if j == n_criteria - 1 else 2.0,
                    step=1.0,
                    key=f"ow_{j}"
                )
                ow.append(value)

        weights, epsilon = bwm(bo, ow)

    # ---------------- Entropie ----------------
    elif weight_method == "Entropie":

        st.info(
            "Les poids d'entropie sont calculés directement à partir "
            "de la matrice de décision."
        )

        weights = entropy(X)

    # ---------------- CRITIC ----------------
    elif weight_method == "CRITIC":

        st.info(
            "Les poids CRITIC sont calculés à partir de la dispersion "
            "et de la corrélation entre les critères."
        )

        weights = critic(X, criteria_types)

    elif weight_method == "Linguistique (flou)":
        st.subheader("Importance linguistique des critères")
        scale = FUZZY_WEIGHT_SCALE if decision_mode == "Fuzzy" else IF_WEIGHT_SCALE
        options = list(scale)
        convert = (
            fuzzy_matrix_from_labels
            if decision_mode == "Fuzzy"
            else if_matrix_from_labels
        )

        decision_weights = []
        tabs = st.tabs([f"Décideur {index + 1}" for index in range(decision_makers)])
        for index, tab in enumerate(tabs):
            with tab:
                edited_weights = label_table(
                    options,
                    f"criterion_weights_{decision_mode}_{index}",
                    ["Importance"],
                    criterion_names,
                    options[len(options) // 2],
                )
                decision_weights.append(
                    convert(edited_weights.to_numpy(dtype=object)[0], scale)
                )

        if decision_mode == "Fuzzy":
            fuzzy_W = aggregate_fuzzy(decision_weights)
            crisp_weights = defuzzify_tfn(fuzzy_W)
        else:
            if_W = ifwa(decision_weights)
            crisp_weights = if_score(if_W)
        if np.isclose(crisp_weights.sum(), 0):
            raise ValueError("Les évaluations linguistiques donnent des poids nuls.")
        weights = crisp_weights / crisp_weights.sum()

    weights = validate_weights(weights)

except Exception as e:
    st.error(f"Erreur de pondération : {e}")
    st.stop()


# AFFICHAGE DES POIDS

st.subheader(f"📌 Poids obtenus — {weight_method}")



weights_df = pd.DataFrame({
    "Critère": criterion_names,
    "Poids": weights,
    "Pourcentage": weights * 100
})

st.dataframe(
    weights_df.style.format({
        "Poids": "{:.4f}",
        "Pourcentage": "{:.2f}%"
    }),
    use_container_width=True,
    hide_index=True
)

if weight_method == "BWM":
    st.metric("ε* — cohérence BWM", f"{epsilon:.6f}")


# CLASSEMENT

st.header("Classement des alternatives")

try:
    if ranking_method in ("WSM", "WPM", "WASPAS"):
        if ranking_method == "WASPAS":
            lam = st.slider(
                "λ — équilibre WSM/WPM",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
            )
        else:
            lam = 1.0 if ranking_method == "WSM" else 0.0
        scores, ranking = waspas(X, weights, criteria_types, lam=lam)
        score_label = f"Score {ranking_method}"
        best_is_high = True

    elif ranking_method == "TOPSIS":
        scores, ranking = topsis(X, weights, criteria_types)
        score_label = "Coefficient TOPSIS (RC*)"
        best_is_high = True

    elif ranking_method == "VIKOR":
        v = st.slider(
            "v — poids de l'utilité de groupe",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )
        scores, ranking, S_vik, R_vik = vikor(
            X, weights, criteria_types, v=v, return_details=True
        )
        score_label = "Score VIKOR (Q)"
        best_is_high = False

    elif ranking_method == "Fuzzy TOPSIS":
        if fuzzy_X is None:
            st.error("Fuzzy TOPSIS nécessite le mode Fuzzy.")
            st.stop()
        scores, ranking = fuzzy_topsis(
            fuzzy_X,
            fuzzy_W if fuzzy_W is not None else weights,
            criteria_types,
        )
        score_label = "Coefficient Fuzzy TOPSIS"
        best_is_high = True

    elif ranking_method == "IF-TOPSIS":
        if if_X is None:
            st.error("IF-TOPSIS nécessite le mode Fuzzy intuitionniste.")
            st.stop()
        if_weights = if_W if if_W is not None else crisp_to_if_weights(weights)
        scores, ranking = if_topsis(if_X, if_weights, criteria_types)
        score_label = "Coefficient IF-TOPSIS"
        best_is_high = True

    # Ranking index -> rang
    ranks = np.empty(len(ranking), dtype=int)

    for position, alternative_index in enumerate(ranking, start=1):
        ranks[alternative_index - 1] = position

    results_df = pd.DataFrame({
        "Alternative": alternative_names,
        score_label: scores,
        "Rang": ranks
    })

    results_df = results_df.sort_values("Rang")

    st.success(
        f"Classement calculé avec **{ranking_method}** "
        f"et les poids **{weight_method}**."
    )

    st.dataframe(
        results_df.style.format({
            score_label: "{:.6f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📊 Scores")

    chart_df = results_df.set_index("Alternative")[[score_label]]
    st.bar_chart(chart_df, color="#B9A7D9")

    st.subheader("🏆 Résultat")

    best_alternative = results_df.iloc[0]["Alternative"]

    if best_is_high:
        st.write(
            f"**Alternative classée première : {best_alternative}** "
            f"(score le plus élevé)."
        )
    else:
        st.write(
            f"**Alternative classée première : {best_alternative}** "
            f"(score {ranking_method} le plus faible)."
        )

    if ranking_method == "VIKOR":
        st.subheader("Classements S, R, Q et conditions de compromis")
        rank_s = np.empty(len(S_vik), dtype=int)
        rank_r = np.empty(len(R_vik), dtype=int)
        rank_q = np.empty(len(scores), dtype=int)
        rank_s[np.argsort(S_vik)] = np.arange(1, len(S_vik) + 1)
        rank_r[np.argsort(R_vik)] = np.arange(1, len(R_vik) + 1)
        rank_q[np.argsort(scores)] = np.arange(1, len(scores) + 1)
        st.dataframe(
            pd.DataFrame({
                "Alternative": alternative_names,
                "S": S_vik,
                "Rang S": rank_s,
                "R": R_vik,
                "Rang R": rank_r,
                "Q": scores,
                "Rang Q": rank_q,
            }).round(4),
            use_container_width=True,
            hide_index=True,
        )
        compromise = vikor_compromise(scores, S_vik, R_vik)
        st.write(
            f"Avantage acceptable : "
            f"{'✅' if compromise['advantage'] else '❌'}"
        )
        st.write(
            f"Stabilité acceptable (meilleure en S et R) : "
            f"{'✅' if compromise['stability'] else '❌'}"
        )
        solution_names = [
            alternative_names[index] for index in compromise["solutions"]
        ]
        st.write("**Solution(s) de compromis :** " + ", ".join(solution_names))

except Exception as e:
    st.error(f"Erreur de classement : {e}")


# COMPARAISON DES METHODES (ROBUSTESSE)

# st.header("Comparaison des méthodes (corrélation de Spearman)")

# try:
#     def to_ranks(order):
#         rank_values = np.empty(len(order), dtype=int)
#         for position, alternative_index in enumerate(order, start=1):
#             rank_values[alternative_index - 1] = position
#         return rank_values

#     all_ranks = {
#         "WSM": to_ranks(waspas(X, weights, criteria_types, lam=1.0)[1]),
#         "WPM": to_ranks(waspas(X, weights, criteria_types, lam=0.0)[1]),
#         "WASPAS": to_ranks(waspas(X, weights, criteria_types)[1]),
#         "TOPSIS": to_ranks(topsis(X, weights, criteria_types)[1]),
#         "VIKOR": to_ranks(vikor(X, weights, criteria_types)[1]),
#     }

#     if fuzzy_X is not None:
#         all_ranks["Fuzzy TOPSIS"] = to_ranks(
#             fuzzy_topsis(
#                 fuzzy_X,
#                 fuzzy_W if fuzzy_W is not None else weights,
#                 criteria_types,
#             )[1]
#         )
#     if if_X is not None:
#         comparison_weights = (
#             if_W if if_W is not None else crisp_to_if_weights(weights)
#         )
#         all_ranks["IF-TOPSIS"] = to_ranks(
#             if_topsis(if_X, comparison_weights, criteria_types)[1]
#         )

#     st.subheader("Rangs par méthode")
#     st.dataframe(
#         pd.DataFrame(all_ranks, index=alternative_names),
#         use_container_width=True,
#     )
#     st.subheader("Matrice de corrélation de Spearman")
#     st.dataframe(spearman_matrix(all_ranks).round(2), use_container_width=True)
# except Exception as e:
#     st.warning(f"Comparaison impossible : {e}")


# RAPPEL

with st.expander("ℹ️ Rappel des méthodes"):

    st.markdown("""
### Pondération

- **AHP** : comparaison par paires des critères.
- **BWM** : comparaison Best-to-Others et Others-to-Worst.
- **Entropie** : poids basés sur la quantité d'information/discrimination.
- **CRITIC** : poids basés sur la dispersion et la corrélation.

### Classement

- **WSM** : somme pondérée. Plus grand score = meilleur.
- **WPM** : produit pondéré. Plus grand score = meilleur.
- **WASPAS** : combine WSM et WPM. Plus grand score = meilleur.
- **TOPSIS** : proximité de la solution idéale. Plus grand RC* = meilleur.
- **VIKOR** : solution de compromis. Plus petit Q = meilleur.
- **Fuzzy TOPSIS** : classement à partir d'évaluations floues triangulaires.
- **IF-TOPSIS** : classement intuitionniste à partir de (μ, ν, π).
""")