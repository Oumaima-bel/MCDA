import streamlit as st
import numpy as np
import pandas as pd


from classification_methods import validate_weights, waspas, topsis, vikor
from weight_methods import ahp, bwm, entropy, critic

# CONFIGURATION

st.set_page_config(
    page_title="MCDA - Pondération & Classement",
    page_icon="📊",
    layout="wide"
)

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

st.sidebar.divider()

weight_method = st.sidebar.selectbox(
    "Méthode de pondération",
    ["AHP", "BWM", "Entropie", "CRITIC"]
)

ranking_method = st.sidebar.selectbox(
    "Méthode de classement",
    ["WASPAS", "TOPSIS", "VIKOR"]
)


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


# MATRICE DE DECISION

st.header(" Matrice de décision")

st.info(
    "Entrez la performance de chaque alternative pour chaque critère. "
    "Les lignes représentent les alternatives et les colonnes les critères."
)

default_X = np.ones((n_alternatives, n_criteria))

X_df = None

try:
    

    X_df = pd.DataFrame(
        default_X,
        index=alternative_names if len(alternative_names) == n_alternatives else None,
        columns=criterion_names if len(criterion_names) == n_criteria else None
    )

    edited_X = st.data_editor(
        X_df,
        use_container_width=True,
        num_rows="fixed",
        key="decision_matrix"
    )

    X = edited_X.to_numpy(dtype=float)

except Exception as e:
    st.error(f"Erreur dans la matrice de décision : {e}")
    st.stop()


# INPUT DE LA METHODE DE PONDERATION

st.header(" Pondération des critères")

weights = None
epsilon = None

try:

    # ---------------- AHP ----------------
    if weight_method == "AHP":

        st.subheader("Matrice de comparaison par paires — AHP")

        st.caption(
            "La matrice doit être carrée. La diagonale doit être égale à 1."
        )

        ahp_default = np.eye(n_criteria)

        

        ahp_df = pd.DataFrame(
            ahp_default,
            index=criterion_names,
            columns=criterion_names
        )

        edited_ahp = st.data_editor(
            ahp_df,
            use_container_width=True,
            num_rows="fixed",
            key="ahp_matrix"
        )

        comparison_matrix = edited_ahp.to_numpy(dtype=float)

        weights = ahp(comparison_matrix)

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

    if ranking_method == "WASPAS":

        lam = st.slider(
            "λ — équilibre WSM/WPM",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )

        scores, ranking = waspas(
            X,
            weights,
            criteria_types,
            lam=lam
        )

        score_label = "Score WASPAS"
        best_is_high = True

    elif ranking_method == "TOPSIS":

        scores, ranking = topsis(
            X,
            weights,
            criteria_types
        )

        score_label = "Coefficient TOPSIS (RC*)"
        best_is_high = True

    else:

        v = st.slider(
            "v — poids de l'utilité de groupe",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )

        scores, ranking = vikor(
            X,
            weights,
            criteria_types,
            v=v
        )

        score_label = "Score VIKOR (Q)"
        best_is_high = False

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
    st.bar_chart(chart_df)

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
            f"(score VIKOR le plus faible)."
        )

except Exception as e:
    st.error(f"Erreur de classement : {e}")


# RAPPEL

with st.expander("ℹ️ Rappel des méthodes"):

    st.markdown("""
### Pondération

- **AHP** : comparaison par paires des critères.
- **BWM** : comparaison Best-to-Others et Others-to-Worst.
- **Entropie** : poids basés sur la quantité d'information/discrimination.
- **CRITIC** : poids basés sur la dispersion et la corrélation.

### Classement

- **WASPAS** : combine WSM et WPM. Plus grand score = meilleur.
- **TOPSIS** : proximité de la solution idéale. Plus grand RC* = meilleur.
- **VIKOR** : solution de compromis. Plus petit Q = meilleur.
""")