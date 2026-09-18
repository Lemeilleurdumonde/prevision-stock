"""
forecasting.py
-----------------
ÉTAPES 4 & 5 : Prévision des ventes + variables saisonnières.

Principe général
----------------
Chaque série de vente est définie par la combinaison (SKU, Magasin) :
c'est à ce niveau que le stock et les commandes fournisseur sont gérés
dans le fichier (voir config.SERIES_KEYS), donc c'est aussi à ce niveau
qu'on prévoit la demande.

Pour CHAQUE série, on compare plusieurs modèles :
- 3 modèles de référence (baselines) très simples ;
- Holt-Winters (lissage exponentiel, si `statsmodels` est installé) ;
- une Random Forest utilisant des variables temporelles.

On choisit le modèle qui obtient la meilleure erreur sur un backtest
historique (les derniers mois sont mis de côté comme test), plutôt que
de partir du principe que le modèle le plus complexe est le meilleur.

Pas de fuite de données (data leakage)
---------------------------------------
Toutes les variables retardées (lag1, lag3_mean...) sont calculées avec
`.shift(1)` : la valeur du mois M n'utilise JAMAIS l'information du
mois M lui-même, seulement les mois précédents.
"""

import numpy as np
import pandas as pd

from src.config import (
    COL_DATE, COL_SKU, COL_MAGASIN, COL_QTE_VENDUE, COL_PROMOTION,
    SERIES_KEYS, ROLLING_WINDOWS, BACKTEST_HORIZON_MOIS,
    MIN_POINTS_POUR_PREVISION, SAISON_PERIODES, HORIZONS_PREVISION,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_RANDOM_STATE,
)
from src.utils import saison_depuis_mois

HORIZON_MAX = max(HORIZONS_PREVISION)


# ------------------------------------------------------------------
# Préparation des séries
# ------------------------------------------------------------------
def preparer_serie(df: pd.DataFrame, sku: str, magasin: str) -> pd.DataFrame:
    """
    Extrait et trie la série temporelle (Date, Quantité vendue, Promotion)
    pour un couple (SKU, Magasin) donné.
    """
    sous_df = df[(df[COL_SKU] == sku) & (df[COL_MAGASIN] == magasin)]
    sous_df = sous_df.sort_values(COL_DATE)
    return sous_df[[COL_DATE, COL_QTE_VENDUE, COL_PROMOTION]].reset_index(drop=True)


def generer_dates_futures(dates_historiques, horizon: int) -> pd.DatetimeIndex:
    """
    Génère `horizon` dates futures, espacées du même intervalle médian
    que celui observé dans l'historique (les dates du fichier ne sont
    pas exactement alignées sur des débuts de mois calendaires).
    """
    dates = pd.DatetimeIndex(sorted(dates_historiques))
    ecarts = dates.to_series().diff().dropna()
    delta = ecarts.median() if len(ecarts) > 0 else pd.Timedelta(days=30)

    futures = [dates[-1] + delta * (i + 1) for i in range(horizon)]
    return pd.DatetimeIndex(futures)


# ------------------------------------------------------------------
# Variables saisonnières / calendaires (ÉTAPE 5)
# ------------------------------------------------------------------
def construire_features(dates, valeurs) -> pd.DataFrame:
    """
    Construit un tableau de variables explicatives pour chaque point de
    la série : mois, trimestre, saison, année, et variables retardées
    (lag1, moyennes mobiles des 3 et 6 derniers mois).

    Toutes les variables retardées utilisent `.shift(1)` pour ne JAMAIS
    inclure la valeur du mois qu'on cherche à expliquer (pas de fuite).
    """
    serie = pd.Series(valeurs, index=pd.DatetimeIndex(dates)).sort_index()

    feat = pd.DataFrame(index=serie.index)
    feat["annee"] = feat.index.year
    feat["mois"] = feat.index.month
    feat["trimestre"] = feat.index.quarter
    feat["saison"] = feat["mois"].apply(saison_depuis_mois)
    feat["lag1"] = serie.shift(1)

    for fenetre in ROLLING_WINDOWS:
        feat[f"lag_moy_{fenetre}"] = serie.shift(1).rolling(window=fenetre, min_periods=1).mean()

    feat["cible"] = serie.values
    return feat


# ------------------------------------------------------------------
# Modèles de référence (baselines)
# ------------------------------------------------------------------
def forecast_moyenne_historique(valeurs, horizon: int) -> np.ndarray:
    """Prévision = moyenne de tout l'historique disponible, répétée sur l'horizon."""
    return np.full(horizon, np.mean(valeurs))


def forecast_moyenne_mobile(valeurs, horizon: int, fenetre: int = 3) -> np.ndarray:
    """Prévision = moyenne des `fenetre` derniers mois connus, répétée sur l'horizon."""
    fenetre = min(fenetre, len(valeurs))
    return np.full(horizon, np.mean(valeurs[-fenetre:]))


def forecast_meme_mois_annees_precedentes(dates_train, valeurs_train, dates_futures) -> np.ndarray:
    """
    Prévision = moyenne historique des ventes pour CE MÊME MOIS calendaire
    les années précédentes (ex: prévoir juillet 2026 = moyenne des juillets
    passés). Si le mois n'a pas d'historique, on utilise la moyenne globale.
    """
    serie = pd.Series(valeurs_train, index=pd.DatetimeIndex(dates_train))
    moyenne_par_mois = serie.groupby(serie.index.month).mean()
    moyenne_globale = serie.mean()

    return np.array([
        moyenne_par_mois.get(d.month, moyenne_globale) for d in dates_futures
    ])


# ------------------------------------------------------------------
# Modèle avancé 1 : Holt-Winters (lissage exponentiel)
# ------------------------------------------------------------------
def forecast_holt_winters(valeurs, horizon: int, seasonal_periods: int = SAISON_PERIODES):
    """
    Prévision par lissage exponentiel Holt-Winters.

    - Si `statsmodels` n'est pas installé, retourne None (le modèle est
      simplement exclu de la comparaison, sans faire planter le programme).
    - Si l'historique est trop court pour une saisonnalité annuelle
      (< 2 cycles, soit 24 points), on utilise une version sans
      saisonnalité (tendance uniquement).
    - Si le modèle échoue pour une autre raison, retourne None.
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        return None

    valeurs = np.asarray(valeurs, dtype=float)

    try:
        if len(valeurs) >= 2 * seasonal_periods:
            modele = ExponentialSmoothing(
                valeurs, trend="add", seasonal="add",
                seasonal_periods=seasonal_periods, initialization_method="estimated",
            )
        elif len(valeurs) >= 4:
            modele = ExponentialSmoothing(
                valeurs, trend="add", seasonal=None, initialization_method="estimated",
            )
        else:
            return None

        ajuste = modele.fit(optimized=True)
        prevision = np.asarray(ajuste.forecast(horizon))
        return np.clip(prevision, 0, None)  # une vente ne peut pas être négative
    except Exception:
        return None


# ------------------------------------------------------------------
# Modèle avancé 2 : Random Forest avec variables temporelles
# ------------------------------------------------------------------
def forecast_random_forest(dates_train, valeurs_train, dates_futures):
    """
    Entraîne une Random Forest sur les variables calendaires/retardées de
    la série, puis prévoit récursivement les dates futures : chaque
    prévision sert de "lag1" pour prévoir le mois suivant.

    Retourne None si l'historique est trop court pour entraîner un modèle
    de façon fiable (moins de 5 exemples d'entraînement).
    """
    from sklearn.ensemble import RandomForestRegressor

    feat = construire_features(dates_train, valeurs_train).dropna()
    colonnes_x = ["mois", "trimestre", "lag1"] + [f"lag_moy_{f}" for f in ROLLING_WINDOWS]

    if len(feat) < 5:
        return None

    modele = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, random_state=RF_RANDOM_STATE,
    )
    modele.fit(feat[colonnes_x], feat["cible"])

    historique = list(valeurs_train)
    previsions = []
    for d in dates_futures:
        lag1 = historique[-1]
        ligne = {"mois": d.month, "trimestre": d.quarter, "lag1": lag1}
        for fenetre in ROLLING_WINDOWS:
            ligne[f"lag_moy_{fenetre}"] = np.mean(historique[-fenetre:])

        x = pd.DataFrame([ligne])[colonnes_x]
        pred = max(0.0, float(modele.predict(x)[0]))  # une vente ne peut pas être négative
        previsions.append(pred)
        historique.append(pred)

    return np.array(previsions)


# ------------------------------------------------------------------
# Métriques de backtest
# ------------------------------------------------------------------
def calculer_metriques(y_reel, y_predit) -> dict:
    """
    Calcule MAE, RMSE, WAPE, MAPE et le biais de prévision.

    - WAPE (Weighted Absolute Percentage Error) : métrique principale,
      robuste même quand certaines ventes réelles sont proches de 0.
    - MAPE : calculé en ignorant les points où la valeur réelle est 0
      (sinon division par zéro).
    - Biais : moyenne (prévision - réel). Positif = on sur-prévoit,
      négatif = on sous-prévoit.
    """
    y_reel = np.asarray(y_reel, dtype=float)
    y_predit = np.asarray(y_predit, dtype=float)
    erreurs = y_predit - y_reel

    mae = float(np.mean(np.abs(erreurs)))
    rmse = float(np.sqrt(np.mean(erreurs ** 2)))

    somme_abs_reel = np.sum(np.abs(y_reel))
    wape = float(np.sum(np.abs(erreurs)) / somme_abs_reel * 100) if somme_abs_reel > 0 else np.nan

    masque = y_reel != 0
    mape = float(np.mean(np.abs(erreurs[masque] / y_reel[masque])) * 100) if masque.any() else np.nan

    biais = float(np.mean(erreurs))

    return {"MAE": mae, "RMSE": rmse, "WAPE": wape, "MAPE": mape, "Biais": biais}


# ------------------------------------------------------------------
# Backtest : comparaison des modèles sur une série
# ------------------------------------------------------------------
def backtester_serie(dates, valeurs, horizon_test: int = BACKTEST_HORIZON_MOIS) -> pd.DataFrame:
    """
    Compare tous les modèles sur les `horizon_test` derniers mois connus
    (mis de côté comme test), pour UNE série (SKU, Magasin).

    Retourne un DataFrame avec une ligne par modèle testé et ses
    métriques d'erreur. Retourne None si la série est trop courte.
    """
    n = len(valeurs)
    if n <= horizon_test + MIN_POINTS_POUR_PREVISION:
        return None

    dates = pd.DatetimeIndex(dates)  # garantit des éléments pd.Timestamp (avec .month, .quarter)
    valeurs = np.asarray(valeurs, dtype=float)

    dates_train, valeurs_train = dates[:-horizon_test], valeurs[:-horizon_test]
    dates_test, valeurs_test = dates[-horizon_test:], valeurs[-horizon_test:]

    previsions_par_modele = {
        "Moyenne historique": forecast_moyenne_historique(valeurs_train, horizon_test),
        "Moyenne mobile (3 mois)": forecast_moyenne_mobile(valeurs_train, horizon_test),
        "Moyenne même mois (N-1)": forecast_meme_mois_annees_precedentes(
            dates_train, valeurs_train, dates_test
        ),
    }

    hw = forecast_holt_winters(valeurs_train, horizon_test)
    if hw is not None:
        previsions_par_modele["Holt-Winters"] = hw

    rf = forecast_random_forest(dates_train, valeurs_train, dates_test)
    if rf is not None:
        previsions_par_modele["Random Forest"] = rf

    lignes = []
    for nom_modele, previsions in previsions_par_modele.items():
        metriques = calculer_metriques(valeurs_test, previsions)
        metriques["Modele"] = nom_modele
        lignes.append(metriques)

    return pd.DataFrame(lignes).sort_values("WAPE")


def backtester_toutes_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lance le backtest pour toutes les combinaisons (SKU, Magasin) du
    dataset, et rassemble les résultats dans un seul DataFrame.
    """
    resultats = []
    combinaisons = df[SERIES_KEYS].drop_duplicates().values

    for sku, magasin in combinaisons:
        serie = preparer_serie(df, sku, magasin)
        resultat = backtester_serie(serie[COL_DATE].values, serie[COL_QTE_VENDUE].values)
        if resultat is None:
            continue
        resultat["SKU"] = sku
        resultat["Magasin"] = magasin
        resultats.append(resultat)

    if not resultats:
        return pd.DataFrame()

    return pd.concat(resultats, ignore_index=True)


def selectionner_meilleur_modele(resultats_backtest: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque série (SKU, Magasin), sélectionne le modèle avec le WAPE
    le plus bas (métrique la plus robuste pour ces volumes de vente).
    """
    idx = resultats_backtest.groupby(["SKU", "Magasin"])["WAPE"].idxmin()
    return resultats_backtest.loc[idx].reset_index(drop=True)


# ------------------------------------------------------------------
# Génération de la prévision finale (sur données complètes)
# ------------------------------------------------------------------
MODELES_DISPONIBLES = {
    "Moyenne historique": lambda dt, dv, df_: forecast_moyenne_historique(dv, len(df_)),
    "Moyenne mobile (3 mois)": lambda dt, dv, df_: forecast_moyenne_mobile(dv, len(df_)),
    "Moyenne même mois (N-1)": lambda dt, dv, df_: forecast_meme_mois_annees_precedentes(dt, dv, df_),
    "Holt-Winters": lambda dt, dv, df_: forecast_holt_winters(dv, len(df_)),
    "Random Forest": lambda dt, dv, df_: forecast_random_forest(dt, dv, df_),
}


def generer_previsions(df: pd.DataFrame, meilleurs_modeles: pd.DataFrame, horizon: int = HORIZON_MAX) -> pd.DataFrame:
    """
    Pour chaque série (SKU, Magasin), réentraîne le modèle SÉLECTIONNÉ
    (celui qui a gagné le backtest) sur TOUT l'historique disponible, et
    génère la prévision pour les `horizon` prochains mois.

    Retourne un DataFrame long : SKU, Magasin, Date, Prevision, Modele_utilise.
    """
    lignes = []

    for _, ligne in meilleurs_modeles.iterrows():
        sku, magasin, modele_nom = ligne["SKU"], ligne["Magasin"], ligne["Modele"]

        serie = preparer_serie(df, sku, magasin)
        dates = serie[COL_DATE].values
        valeurs = serie[COL_QTE_VENDUE].values
        dates_futures = generer_dates_futures(dates, horizon)

        fonction_modele = MODELES_DISPONIBLES.get(modele_nom)
        previsions = fonction_modele(dates, valeurs, dates_futures) if fonction_modele else None

        # Filet de sécurité : si le modèle gagnant du backtest échoue au
        # réentraînement final (rare, ex: statsmodels absent), on retombe
        # sur la moyenne historique plutôt que de planter le programme.
        if previsions is None:
            previsions = forecast_moyenne_historique(valeurs, horizon)
            modele_nom = f"{modele_nom} (repli: Moyenne historique)"

        for d, p in zip(dates_futures, previsions):
            lignes.append({
                "SKU": sku, "Magasin": magasin, "Date": d,
                "Prevision": round(float(p), 1), "Modele_utilise": modele_nom,
            })

    return pd.DataFrame(lignes)


if __name__ == "__main__":
    from src.config import DEFAULT_DATA_FILE
    from src.data_loader import run_data_quality_report
    from src.data_cleaning import clean_data

    df_raw = run_data_quality_report(DEFAULT_DATA_FILE)
    df_clean = clean_data(df_raw)

    print("\n### ÉTAPES 4 & 5 — PRÉVISION DES VENTES ###\n")

    print("Backtest de tous les modèles sur toutes les séries...")
    resultats = backtester_toutes_series(df_clean)
    print(f"{len(resultats)} lignes de résultats de backtest.\n")

    meilleurs = selectionner_meilleur_modele(resultats)
    print("Meilleur modèle par série (SKU, Magasin) :")
    print(meilleurs[["SKU", "Magasin", "Modele", "WAPE", "MAE", "Biais"]].to_string(index=False))

    print("\nRépartition des modèles gagnants :")
    print(meilleurs["Modele"].value_counts())

    previsions = generer_previsions(df_clean, meilleurs, horizon=HORIZON_MAX)
    print(f"\n{len(previsions)} lignes de prévisions générées (horizon = {HORIZON_MAX} mois).")
    print(previsions.head(15).to_string(index=False))
