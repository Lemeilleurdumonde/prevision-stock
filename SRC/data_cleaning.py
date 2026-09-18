"""
data_cleaning.py
------------------
ÉTAPE 2 : Nettoyage des données.

Ce module transforme le DataFrame brut (issu de data_loader.py) en un
DataFrame propre, prêt pour l'analyse et la modélisation.

Principe important : le fichier Excel original n'est JAMAIS modifié.
On travaille uniquement sur des copies en mémoire (df.copy()).

Chaque décision de nettoyage est expliquée dans les commentaires et
affichée à l'écran, pour que tu puisses vérifier que les hypothèses
prises sont raisonnables pour ton activité.
"""

import numpy as np
import pandas as pd

from src.config import (
    COL_SKU, COL_PRODUIT, COL_CATEGORIE, COL_MAGASIN,
    COL_QTE_VENDUE, COL_PRIX_UNITAIRE, COL_STOCK_DISPO, COL_QTE_COMMANDEE,
    COL_DATE, COL_DATE_CMD_FOURNISSEUR, COL_DATE_RECEPTION_PREVUE,
    COL_DELAI_FOURNISSEUR, COL_PROMOTION,
    NUMERIC_COLUMNS, TEXT_COLUMNS,
    PROMOTION_VALEUR_PAR_DEFAUT,
)
from src.data_loader import convert_date_columns


def _canonicalize_casing(series: pd.Series) -> pd.Series:
    """
    Pour une colonne à vocabulaire contrôlé (ex: noms de magasins, de
    catégories), remplace chaque valeur par sa variante de casse LA PLUS
    FRÉQUENTE parmi celles réellement présentes dans les données.

    Exemple : si on trouve 'Casablanca' (120 fois) et 'casablanca' (3 fois),
    tout est ramené à 'Casablanca'.

    On préfère cette méthode à un simple `.str.title()`, qui casserait des
    valeurs contenant des acronymes ou des unités (ex: 'TV', 'BTU', '4K'
    deviendraient 'Tv', 'Btu', '4K'... et 'Réfrigérateur 400L' deviendrait
    'Réfrigérateur 400L' → correct par hasard, mais 'BTU' → 'Btu' serait
    faux). Ici, on ne choisit qu'entre des variantes qui existent déjà
    réellement dans le fichier, donc aucun acronyme n'est inventé ni cassé.
    """
    non_null = series.dropna()
    if non_null.empty:
        return series

    counts = non_null.value_counts()  # variantes existantes, de la plus courante à la plus rare
    canonical_by_lower = {}
    for value in counts.index:
        key = value.lower()
        canonical_by_lower.setdefault(key, value)  # la 1ère rencontrée = la plus fréquente

    return series.map(lambda v: canonical_by_lower.get(v.lower(), v) if pd.notna(v) else v)


def standardize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uniformise les colonnes textuelles :
    - supprime les espaces en début/fin ;
    - réduit les espaces multiples ;
    - harmonise la casse.

    La stratégie de casse diffère selon le type de colonne :
    - SKU              : toujours en MAJUSCULES (c'est un code produit).
    - Magasin, Catégorie : vocabulaire contrôlé -> on garde la variante de
      casse la plus fréquente réellement observée (voir _canonicalize_casing).
    - Produit          : texte libre pouvant contenir des acronymes/unités
      (TV, BTU, 4K, kg, L...) -> on ne touche PAS à la casse, seulement
      aux espaces, pour ne pas déformer ces valeurs.
    """
    df = df.copy()

    # Nettoyage des espaces pour toutes les colonnes textuelles
    for col in TEXT_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    if COL_SKU in df.columns:
        df[COL_SKU] = df[COL_SKU].str.upper()

    for col in [COL_MAGASIN, COL_CATEGORIE]:
        if col in df.columns:
            df[col] = _canonicalize_casing(df[col])

    # COL_PRODUIT : espaces nettoyés ci-dessus, casse volontairement inchangée

    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les colonnes numériques (config.NUMERIC_COLUMNS) en nombres.

    Les valeurs qui ne peuvent pas être converties (texte parasite, etc.)
    deviennent NaN. On rapporte combien de valeurs sont concernées.
    """
    df = df.copy()

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            continue

        before_non_null = df[col].notna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after_non_null = df[col].notna().sum()

        lost = before_non_null - after_non_null
        if lost > 0:
            print(f"⚠️  Colonne '{col}' : {lost} valeur(s) non numérique(s) converties en manquant.")

    return df


def neutralize_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Neutralise (met à NaN) les valeurs physiquement impossibles :
    - quantité vendue négative ;
    - stock disponible négatif ;
    - prix unitaire <= 0 ;
    - délai fournisseur négatif.

    HYPOTHÈSE IMPORTANTE (à valider avec toi) : dans ce prototype, une
    quantité vendue négative est considérée comme une ERREUR DE SAISIE,
    pas comme un retour produit. Si ton activité enregistre les retours
    comme des quantités négatives, cette règle devra être adaptée avant
    de passer en production.
    """
    df = df.copy()

    checks = {}
    if COL_QTE_VENDUE in df.columns:
        checks[COL_QTE_VENDUE] = df[COL_QTE_VENDUE] < 0
    if COL_STOCK_DISPO in df.columns:
        checks[COL_STOCK_DISPO] = df[COL_STOCK_DISPO] < 0
    if COL_PRIX_UNITAIRE in df.columns:
        checks[COL_PRIX_UNITAIRE] = df[COL_PRIX_UNITAIRE] <= 0
    if COL_DELAI_FOURNISSEUR in df.columns:
        checks[COL_DELAI_FOURNISSEUR] = df[COL_DELAI_FOURNISSEUR] < 0

    for col, mask in checks.items():
        nb = int(mask.sum())
        if nb > 0:
            print(f"⚠️  Colonne '{col}' : {nb} valeur(s) impossible(s) (négative ou nulle) mise(s) à manquant.")
            df.loc[mask, col] = np.nan

    return df


def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes strictement identiques sur toutes les colonnes.

    C'est la SEULE suppression automatique de ce module : une ligne
    100% identique à une autre est très probablement une erreur
    d'export/import du fichier Excel (ligne dupliquée par accident).
    """
    df = df.copy()
    nb_before = len(df)
    df = df.drop_duplicates(keep="first")
    nb_removed = nb_before - len(df)

    if nb_removed > 0:
        print(f"🗑️  {nb_removed} doublon(s) strict(s) supprimé(s).")
    else:
        print("✅ Aucun doublon strict à supprimer.")

    return df


def drop_rows_missing_essential_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes où un champ ESSENTIEL est manquant :
    Date, SKU ou Quantité vendue.

    Pourquoi précisément ces 3 colonnes ?
    - Date              : indispensable pour construire une série temporelle.
    - SKU               : indispensable pour savoir de quel produit on parle.
    - Quantité vendue   : c'est la variable qu'on cherche à prévoir.

    Une ligne sans l'une de ces informations est inutilisable pour la
    prévision. On la retire, en affichant clairement combien de lignes
    sont concernées et pourquoi.
    """
    df = df.copy()
    essential_cols = [c for c in [COL_DATE, COL_SKU, COL_QTE_VENDUE] if c in df.columns]

    mask_incomplete = df[essential_cols].isna().any(axis=1)
    nb_dropped = int(mask_incomplete.sum())

    if nb_dropped > 0:
        print(
            f"🗑️  {nb_dropped} ligne(s) supprimée(s) car il manque au moins un champ "
            f"essentiel parmi {essential_cols} (Date, SKU ou Quantité vendue)."
        )
    else:
        print("✅ Aucune ligne à supprimer pour champ essentiel manquant.")

    return df[~mask_incomplete].copy()


def fill_missing_descriptive_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remplit Produit / Catégorie / Magasin manquants :
    1. en se basant sur d'autres lignes du même SKU (si disponible) ;
    2. sinon, avec la valeur 'Inconnu' (pour ne rien inventer).
    """
    df = df.copy()

    for col in [COL_PRODUIT, COL_CATEGORIE, COL_MAGASIN]:
        if col not in df.columns:
            continue

        nb_missing_before = int(df[col].isna().sum())
        if nb_missing_before == 0:
            continue

        nb_filled_via_lookup = 0
        if col in (COL_PRODUIT, COL_CATEGORIE) and COL_SKU in df.columns:
            lookup = df.dropna(subset=[col]).groupby(COL_SKU)[col].agg(
                lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
            )
            before = df[col].isna().sum()
            df[col] = df[col].fillna(df[COL_SKU].map(lookup))
            nb_filled_via_lookup = int(before - df[col].isna().sum())

        nb_missing_after = int(df[col].isna().sum())
        if nb_missing_after > 0:
            df[col] = df[col].fillna("Inconnu")

        if nb_filled_via_lookup > 0:
            print(f"ℹ️  Colonne '{col}' : {nb_filled_via_lookup} valeur(s) déduite(s) via le SKU.")
        if nb_missing_after > 0:
            print(f"⚠️  Colonne '{col}' : {nb_missing_after} valeur(s) restée(s) inconnue(s), remplie(s) par 'Inconnu'.")

    return df


def fill_missing_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remplit les prix unitaires manquants avec, dans l'ordre de préférence :
    1. la médiane du prix pour CE SKU (le plus fiable) ;
    2. sinon, la médiane globale du prix (dernier recours).
    """
    df = df.copy()
    if COL_PRIX_UNITAIRE not in df.columns:
        return df

    nb_missing = int(df[COL_PRIX_UNITAIRE].isna().sum())
    if nb_missing == 0:
        return df

    if COL_SKU in df.columns:
        median_by_sku = df.groupby(COL_SKU)[COL_PRIX_UNITAIRE].transform("median")
        df[COL_PRIX_UNITAIRE] = df[COL_PRIX_UNITAIRE].fillna(median_by_sku)

    global_median = df[COL_PRIX_UNITAIRE].median()
    df[COL_PRIX_UNITAIRE] = df[COL_PRIX_UNITAIRE].fillna(global_median)

    print(
        f"ℹ️  Colonne '{COL_PRIX_UNITAIRE}' : {nb_missing} valeur(s) manquante(s) remplie(s) "
        f"par la médiane du même SKU (ou médiane globale à défaut)."
    )
    return df


def standardize_promotion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uniformise la colonne Promotion (valeurs attendues : 'Oui' / 'Non').

    HYPOTHÈSE (à valider avec toi) : une ligne sans information de
    promotion est considérée comme une vente SANS promotion ('Non').
    """
    df = df.copy()
    if COL_PROMOTION not in df.columns:
        return df

    df[COL_PROMOTION] = df[COL_PROMOTION].astype("string").str.strip().str.title()

    nb_missing = int(df[COL_PROMOTION].isna().sum())
    if nb_missing > 0:
        df[COL_PROMOTION] = df[COL_PROMOTION].fillna(PROMOTION_VALEUR_PAR_DEFAUT)
        print(
            f"ℹ️  Colonne '{COL_PROMOTION}' : {nb_missing} valeur(s) manquante(s) "
            f"considérée(s) comme '{PROMOTION_VALEUR_PAR_DEFAUT}' (pas de promotion)."
        )

    valeurs_inattendues = set(df[COL_PROMOTION].unique()) - {"Oui", "Non"}
    if valeurs_inattendues:
        print(f"⚠️  Valeurs inattendues dans '{COL_PROMOTION}' : {valeurs_inattendues}")

    return df


def report_remaining_gaps(df: pd.DataFrame) -> None:
    """
    Signale les colonnes où des valeurs manquantes restent VOLONTAIREMENT
    (on ne les invente pas car elles dépendent du contexte métier) :
    - Stock disponible
    - Quantité commandée / dates de commande / réception / délai fournisseur
      (normal si aucune commande n'était en cours à cette date)
    """
    cols_to_check = [
        COL_STOCK_DISPO, COL_QTE_COMMANDEE, COL_DATE_CMD_FOURNISSEUR,
        COL_DATE_RECEPTION_PREVUE, COL_DELAI_FOURNISSEUR,
    ]
    print("\nℹ️  Valeurs manquantes volontairement conservées (non inventées) :")
    any_gap = False
    for col in cols_to_check:
        if col not in df.columns:
            continue
        nb = int(df[col].isna().sum())
        if nb > 0:
            any_gap = True
            print(f"   - {col} : {nb} valeur(s) manquante(s)")
    if not any_gap:
        print("   (aucune)")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fonction principale de l'étape 2 : orchestre tout le nettoyage.

    Prend le DataFrame brut (déjà passé par data_loader) et retourne un
    NOUVEAU DataFrame propre. Le DataFrame d'entrée n'est jamais modifié
    en place, et le fichier Excel original n'est jamais touché.
    """
    print("\n### ÉTAPE 2 — NETTOYAGE DES DONNÉES ###\n")

    df_clean = df.copy()
    nb_rows_start = len(df_clean)

    df_clean = standardize_text_columns(df_clean)
    df_clean = convert_numeric_columns(df_clean)
    df_clean = convert_date_columns(df_clean)          # idempotent si déjà fait à l'étape 1
    df_clean = neutralize_impossible_values(df_clean)
    df_clean = remove_exact_duplicates(df_clean)
    df_clean = drop_rows_missing_essential_fields(df_clean)
    df_clean = fill_missing_descriptive_fields(df_clean)
    df_clean = fill_missing_prices(df_clean)
    df_clean = standardize_promotion(df_clean)

    report_remaining_gaps(df_clean)

    nb_rows_end = len(df_clean)
    print(
        f"\n✅ Nettoyage terminé : {nb_rows_start} → {nb_rows_end} lignes "
        f"({nb_rows_start - nb_rows_end} ligne(s) supprimée(s) au total).\n"
    )

    return df_clean


if __name__ == "__main__":
    from src.config import DEFAULT_DATA_FILE
    from src.data_loader import run_data_quality_report

    df_raw = run_data_quality_report(DEFAULT_DATA_FILE)
    df_clean = clean_data(df_raw)

    print(df_clean.head())
