# Application MCDA : pondération des critères et classement des alternatives

Application web interactive, développée avec Streamlit, pour l'aide à la décision multicritère (MCDA). Elle permet de calculer les poids des critères, de classer des alternatives avec plusieurs méthodes et de comparer la cohérence des classements obtenus.

## Fonctionnalités

### Types d'évaluations

- **Crisp** : valeurs numériques précises.
- **Fuzzy** : appréciations linguistiques converties en nombres flous triangulaires (TFN), avec prise en charge de plusieurs décideurs (agrégation des évaluations).

### Méthodes de pondération des critères

| Méthode | Principe |
|---|---|
| AHP | Comparaisons par paires (échelle de Saaty), avec calcul de λmax, CI et CR |
| BWM | Comparaisons Best-to-Others et Others-to-Worst, avec indicateur de cohérence ε* |
| Entropie | Poids calculés à partir de la dispersion de l'information dans la matrice de décision |
| CRITIC | Poids calculés à partir de la dispersion et de la corrélation entre critères |
| Linguistique (flou) | Importance de chaque critère exprimée par des termes linguistiques (mode Fuzzy) |

### Méthodes de classement des alternatives

| Méthode | Remarque |
|---|---|
| WSM | Somme pondérée |
| WPM | Produit pondéré |
| WASPAS | Combinaison de WSM et WPM, paramètre λ réglable |
| TOPSIS | Proximité de la solution idéale |
| VIKOR | Solution de compromis, paramètre v réglable, conditions de compromis affichées |
| Fuzzy TOPSIS | Classement à partir de nombres flous triangulaires (mode Fuzzy) |

### Analyse de robustesse

Tableau des rangs obtenus par chaque méthode et matrice de corrélation de Spearman pour mesurer la concordance des classements.

### Import de données depuis un fichier

Les fichiers .csv, .xlsx et .xls sont acceptés. Le séparateur du CSV est détecté automatiquement, la virgule décimale est acceptée et un sélecteur de feuille apparaît pour les classeurs Excel à plusieurs feuilles. Un modèle CSV téléchargeable est proposé pour chaque type d'import.

| Donnée | Format attendu |
|---|---|
| Matrice de décision (Crisp) | 1re ligne : noms des critères. 1re colonne : noms des alternatives (facultative). Le reste : valeurs numériques. |
| Matrices de décision (Fuzzy) | Même structure, avec les termes de l'échelle linguistique (ou leur numéro dans l'échelle). Un fichier par décideur, de mêmes dimensions. |
| Matrice AHP | Matrice carrée critères par critères. Seule la partie au-dessus de la diagonale est lue, les fractions (1/3) sont acceptées. |
| BO et OW (BWM) | Colonnes `Critère` (facultative), `BO` et `OW`, une ligne par critère, valeurs entre 1 et 9. |

Les valeurs importées restent modifiables dans les tableaux. En cas d'erreur, le message indique la ligne et la colonne concernées.

## Installation

Prérequis : Python 3.9 ou supérieur.

```bash
git clone <url-du-depot>
cd <nom-du-depot>

python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

pip install streamlit numpy pandas openpyxl xlrd
```

`openpyxl` est nécessaire pour lire les fichiers .xlsx et `xlrd` pour les anciens fichiers .xls. Si les modules du projet utilisent d'autres bibliothèques (par exemple SciPy), ajoutez-les à la commande d'installation.

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre dans le navigateur, par défaut à l'adresse `http://localhost:8501`.

## Utilisation rapide

1. Dans la barre latérale, choisissez la nature des évaluations (Crisp ou Fuzzy), le nombre d'alternatives et de critères (ou importez un fichier qui les fixe), la méthode de classement et la méthode de pondération.
2. Saisissez les noms des alternatives et des critères, puis indiquez le type de chaque critère : Positif (+) si plus la valeur est grande, mieux c'est, ou Négatif (-) si plus la valeur est petite, mieux c'est.
3. Remplissez ou importez la matrice de décision.
4. Renseignez les données de la méthode de pondération choisie (matrice AHP, comparaisons BO et OW, ou importances linguistiques). Les méthodes Entropie et CRITIC n'ont besoin d'aucune saisie.
5. Consultez le classement, le graphique des scores, puis la comparaison des méthodes.

Un mode d'emploi détaillé est également disponible dans l'application, dans la section repliable en haut de page.

## Structure du projet

```
.
├── app.py                        Interface Streamlit
├── classification_methods.py     WSM, WPM, WASPAS, TOPSIS, VIKOR
├── weight_methods.py             AHP, BWM, Entropie, CRITIC
├── fuzzy_methods.py              Échelles linguistiques, TFN, Fuzzy TOPSIS
├── intuitionistic_methods.py     Méthodes floues intuitionnistes (importées par app.py)
└── robustness.py                 Corrélation de Spearman entre classements
```

`app.py` importe encore `intuitionistic_methods.py` : ce fichier doit rester présent dans le dépôt.

## Interprétation des résultats

- **WSM, WPM, WASPAS, TOPSIS, Fuzzy TOPSIS** : plus le score est élevé, meilleure est l'alternative.
- **VIKOR** : plus le score Q est faible, meilleure est l'alternative.
- **AHP** : une matrice est considérée comme cohérente lorsque CR est inférieur à 0,1.
- **BWM** : plus ε* est proche de 0, plus les jugements sont cohérents.
- **Spearman** : une valeur proche de 1 indique que deux méthodes produisent des classements très semblables.

