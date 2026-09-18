"""
inventory.py
--------------
ÉTAPE 6 : Gestion du stock.
ÉTAPE 7 : Recommandation de commande.

Ce module combine les prévisions de vente (forecasting.py) et l'état du
stock/des commandes fournisseur pour produire, par (SKU, Magasin) :
- la demande prévue pendant le délai fournisseur ;
- le stock de sécurité ;
- le stock projeté et le niveau de risque de rupture ;
- la quantité recommandée à commander et la date à laquelle commander.

Hypothèses importantes (à valider avec toi)
--------------------------------------------
1. "Aujourd'hui" = la date la plus récente disponible dans le fichier.
   Dans un usage réel (données mises à jour régulièrement), ce serait
   la date du jour.
2. Le stock de sécurité suit la formule classique :
       stock_securite = Z * écart-type(demande mensuelle) * racine(délai en mois)
   Plus la demande est irrégulière et le délai long, plus le stock de
   sécurité recommandé est important.
3. Si une commande est nécessaire (quantité recommandée > 0), on
   recommande de la passer IMMÉDIATEMENT (le délai fournisseur est déjà
   pris en compte dans le calcul de la demande à couvrir). Sinon, on
   recommande de revérifier au prochain cycle mensuel.
"""

import numpy as np
import pandas as pd

from src.config import (
    COL_DATE, COL_SKU, COL_PRODUIT, COL_MAGASIN, COL_QTE_VENDUE,
    COL_STOCK_DISPO, COL_QTE_COMMANDEE, COL_DATE_RECEPTION_PREVUE,
    COL_DELAI_FOURNISSEUR, SERIES_KEYS, NIVEAU_SERVICE_Z, JOURS_PAR_MOIS,
)

ORDRE_RISQUE = pd.CategoricalDtype(categories=["Faible", "Moyen", "Élevé"], ordered=True)


def _dernier_etat_connu(df: pd.DataFrame, date_reference: pd.Timestamp) -> pd.DataFrame:
    """Pour chaque (SKU, Magasin) : Produit, stock disponible et délai fournisseur les plus récents."""
    connu = df[df[COL_DATE] <= date_reference].sort_values(COL_DATE)
    dernier = connu.groupby(SERIES_KEYS).last()
    return dernier[[COL_PRODUIT, COL_STOCK_DISPO, COL_DELAI_FOURNISSEUR]]


def _stock_en_transit(df: pd.DataFrame, date_reference: pd.Timestamp) -> pd.Series:
    """
    Somme des commandes déjà passées mais pas encore réceptionnées à la
    date de référence (réception prévue dans le futur).
    """
    en_transit = df[
        (df[COL_QTE_COMMANDEE] > 0)
        & df[COL_DATE_RECEPTION_PREVUE].notna()
        & (df[COL_DATE_RECEPTION_PREVUE] > date_reference)
    ]
    return en_transit.groupby(SERIES_KEYS)[COL_QTE_COMMANDEE].sum().rename("Stock_en_transit")


def _ecart_type_demande(df: pd.DataFrame) -> pd.Series:
    """Écart-type de la demande mensuelle historique, par série (utilisé pour le stock de sécurité)."""
    return df.groupby(SERIES_KEYS)[COL_QTE_VENDUE].std().fillna(0).rename("Ecart_type_demande")


def _demande_pendant_delai(previsions: pd.DataFrame, etat: pd.DataFrame) -> pd.Series:
    """
    Pour chaque série, additionne les prévisions des ceil(délai / 30 jours)
    premiers mois à venir : c'est la demande à couvrir pendant que la
    commande est en fabrication/transport chez le fournisseur.
    """
    valeurs = {}
    for cle, ligne in etat.iterrows():
        sku, magasin = cle
        n_mois = max(1, int(np.ceil(ligne[COL_DELAI_FOURNISSEUR] / JOURS_PAR_MOIS)))
        serie_prev = previsions[
            (previsions["SKU"] == sku) & (previsions["Magasin"] == magasin)
        ].sort_values("Date")
        valeurs[cle] = serie_prev["Prevision"].head(n_mois).sum()
    return pd.Series(valeurs, name="Demande_pendant_delai")


def _prochaine_prevision(previsions: pd.DataFrame, etat: pd.DataFrame) -> pd.Series:
    """Prévision du mois suivant pour chaque série (indicateur affiché à titre informatif)."""
    valeurs = {}
    for cle in etat.index:
        sku, magasin = cle
        serie_prev = previsions[
            (previsions["SKU"] == sku) & (previsions["Magasin"] == magasin)
        ].sort_values("Date")
        valeurs[cle] = serie_prev["Prevision"].iloc[0] if not serie_prev.empty else np.nan
    return pd.Series(valeurs, name="Ventes_prevues_mois_prochain")


def _classer_risque(ligne: pd.Series) -> str:
    """
    Élevé : le stock projeté sera négatif -> rupture avant la fin du délai.
    Moyen : le stock projeté restera positif mais sous le stock de sécurité.
    Faible : le stock de sécurité est couvert.
    """
    if ligne["Stock_projete"] < 0:
        return "Élevé"
    if ligne["Stock_projete"] < ligne["Stock_securite"]:
        return "Moyen"
    return "Faible"


def construire_tableau_reapprovisionnement(
    df: pd.DataFrame, previsions: pd.DataFrame, date_reference: pd.Timestamp = None
) -> pd.DataFrame:
    """
    Fonction principale des étapes 6 et 7.

    Paramètres
    ----------
    df : DataFrame nettoyé (sortie de data_cleaning.clean_data)
    previsions : DataFrame de prévisions (sortie de forecasting.generer_previsions)
    date_reference : date considérée comme "aujourd'hui" (par défaut, la
        date la plus récente du fichier)

    Retourne un DataFrame avec une ligne par (SKU, Magasin), prêt à être
    affiché dans le dashboard (page Stock & Réapprovisionnement).
    """
    if date_reference is None:
        date_reference = df[COL_DATE].max()

    etat = _dernier_etat_connu(df, date_reference)
    en_transit = _stock_en_transit(df, date_reference)
    ecart_type = _ecart_type_demande(df)

    table = etat.join(en_transit, how="left").join(ecart_type, how="left")
    table["Stock_en_transit"] = table["Stock_en_transit"].fillna(0)
    table["Ecart_type_demande"] = table["Ecart_type_demande"].fillna(0)

    table["Ventes_prevues_mois_prochain"] = _prochaine_prevision(previsions, table)
    table["Demande_pendant_delai"] = _demande_pendant_delai(previsions, table)

    delai_mois = table[COL_DELAI_FOURNISSEUR] / JOURS_PAR_MOIS
    table["Stock_securite"] = NIVEAU_SERVICE_Z * table["Ecart_type_demande"] * np.sqrt(delai_mois)

    table["Stock_projete"] = (
        table[COL_STOCK_DISPO] + table["Stock_en_transit"] - table["Demande_pendant_delai"]
    )

    table["Risque_rupture"] = table.apply(_classer_risque, axis=1).astype(ORDRE_RISQUE)

    # Besoin futur = demande pendant le délai + stock de sécurité
    #                - stock actuel - stock déjà en transit
    table["Besoin_futur"] = (
        table["Demande_pendant_delai"] + table["Stock_securite"]
        - table[COL_STOCK_DISPO] - table["Stock_en_transit"]
    )
    table["Quantite_recommandee"] = table["Besoin_futur"].clip(lower=0).round().astype(int)

    table["Date_recommandee_commande"] = table["Quantite_recommandee"].apply(
        lambda q: date_reference if q > 0 else date_reference + pd.Timedelta(days=JOURS_PAR_MOIS)
    )

    for col in ["Ventes_prevues_mois_prochain", "Demande_pendant_delai", "Stock_securite", "Stock_projete", "Besoin_futur"]:
        table[col] = table[col].round(1)

    table = table.reset_index()
    colonnes_finales = [
        COL_SKU, COL_PRODUIT, COL_MAGASIN,
        "Ventes_prevues_mois_prochain", COL_STOCK_DISPO, "Stock_en_transit",
        COL_DELAI_FOURNISSEUR, "Demande_pendant_delai", "Stock_securite",
        "Stock_projete", "Risque_rupture", "Besoin_futur",
        "Quantite_recommandee", "Date_recommandee_commande",
    ]
    return table[colonnes_finales].sort_values("Risque_rupture", ascending=False)


if __name__ == "__main__":
    from src.config import DEFAULT_DATA_FILE
    from src.data_loader import run_data_quality_report
    from src.data_cleaning import clean_data
    from src.forecasting import backtester_toutes_series, selectionner_meilleur_modele, generer_previsions

    df_raw = run_data_quality_report(DEFAULT_DATA_FILE)
    df_clean = clean_data(df_raw)

    print("\n### ÉTAPES 4-5 (nécessaires pour le stock) — calcul des prévisions ###")
    resultats_backtest = backtester_toutes_series(df_clean)
    meilleurs_modeles = selectionner_meilleur_modele(resultats_backtest)
    previsions = generer_previsions(df_clean, meilleurs_modeles)

    print("\n### ÉTAPES 6-7 — STOCK & RECOMMANDATION DE COMMANDE ###\n")
    tableau = construire_tableau_reapprovisionnement(df_clean, previsions)
    print(tableau.to_string(index=False))

    print("\nRépartition des niveaux de risque :")
    print(tableau["Risque_rupture"].value_counts())

    print(f"\nProduits à commander maintenant (quantité > 0) : {(tableau['Quantite_recommandee'] > 0).sum()} / {len(tableau)}")
