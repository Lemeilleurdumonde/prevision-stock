"""
data_loader.py
----------------
ÉTAPE 1 : Import et contrôle des données.

Ce module est responsable de :
- charger le fichier Excel fourni par l'utilisateur ;
- vérifier que toutes les colonnes attendues sont présentes ;
- convertir les colonnes de dates ;
- détecter (SANS les supprimer) : valeurs manquantes, doublons,
  valeurs incohérentes ;
- afficher un résumé clair du jeu de données.

IMPORTANT : aucune donnée n'est supprimée dans ce module. On se
contente de diagnostiquer et d'afficher des avertissements. Le
nettoyage effectif (avec suppressions justifiées) est fait dans
data_cleaning.py (étape 2).
"""

from pathlib import Path
import pandas as pd

from src.config import (
    REQUIRED_COLUMNS,
    DATE_COLUMNS,
    COL_QTE_VENDUE,
    COL_STOCK_DISPO,
    COL_PRIX_UNITAIRE,
    COL_DELAI_FOURNISSEUR,
    COL_DATE,
    COL_DATE_CMD_FOURNISSEUR,
    COL_DATE_RECEPTION_PREVUE,
    COL_SKU,
)


def load_excel_file(filepath: Path) -> pd.DataFrame:
    """
    Charge le fichier Excel en DataFrame pandas.

    Lève une erreur claire si le fichier est introuvable ou illisible,
    plutôt que de laisser planter le script avec une erreur pandas brute.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {filepath}\n"
            f"Vérifie que le fichier Excel est bien placé dans le dossier 'data/'."
        )

    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        raise ValueError(f"Impossible de lire le fichier Excel '{filepath}': {e}")

    print(f"✅ Fichier chargé avec succès : {filepath}")
    print(f"   → {df.shape[0]} lignes, {df.shape[1]} colonnes\n")
    return df


def check_required_columns(df: pd.DataFrame) -> list:
    """
    Vérifie que toutes les colonnes attendues (définies dans config.py)
    sont présentes dans le fichier.

    Retourne la liste des colonnes manquantes (vide si tout est OK).
    Ne modifie pas le DataFrame.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        print("❌ Colonnes manquantes détectées :")
        for col in missing:
            print(f"   - {col}")
    else:
        print("✅ Toutes les colonnes attendues sont présentes.")

    return missing


def convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit les colonnes de dates (définies dans config.DATE_COLUMNS)
    en type datetime.

    Les valeurs qui ne peuvent pas être converties deviennent NaT
    (Not a Time) plutôt que de faire planter le script. On rapporte
    combien de valeurs n'ont pas pu être converties.
    """
    df = df.copy()

    for col in DATE_COLUMNS:
        if col not in df.columns:
            continue

        before_non_null = df[col].notna().sum()
        df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        after_non_null = df[col].notna().sum()

        lost = before_non_null - after_non_null
        if lost > 0:
            print(
                f"⚠️  Colonne '{col}' : {lost} valeur(s) n'ont pas pu être "
                f"converties en date et sont devenues manquantes (NaT)."
            )

    return df


def detect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Détecte les valeurs manquantes par colonne.

    Retourne un DataFrame récapitulatif (colonne, nb manquants, % manquants),
    trié du plus problématique au moins problématique.
    """
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)

    report = pd.DataFrame({
        "colonne": missing_count.index,
        "valeurs_manquantes": missing_count.values,
        "pourcentage": missing_pct.values,
    })
    report = report[report["valeurs_manquantes"] > 0].sort_values(
        "valeurs_manquantes", ascending=False
    )

    if report.empty:
        print("✅ Aucune valeur manquante détectée.")
    else:
        print("⚠️  Valeurs manquantes détectées :")
        print(report.to_string(index=False))

    return report


def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Détecte les lignes strictement dupliquées (toutes colonnes identiques).

    Ne supprime rien : retourne les lignes dupliquées pour inspection.
    """
    duplicated_mask = df.duplicated(keep=False)
    duplicates = df[duplicated_mask]

    if duplicates.empty:
        print("✅ Aucun doublon strict détecté.")
    else:
        nb_groups = df.duplicated(keep="first").sum()
        print(
            f"⚠️  {nb_groups} ligne(s) dupliquée(s) détectée(s) "
            f"({duplicated_mask.sum()} lignes concernées au total)."
        )

    return duplicates


def detect_inconsistent_values(df: pd.DataFrame) -> dict:
    """
    Détecte les valeurs incohérentes SANS les supprimer :
    - quantités vendues négatives ;
    - stock disponible négatif ;
    - prix unitaire négatif ou nul ;
    - délai fournisseur négatif ;
    - date de réception prévue antérieure à la date de commande.

    Retourne un dictionnaire {nom_du_probleme: DataFrame des lignes concernées}.
    """
    issues = {}

    if COL_QTE_VENDUE in df.columns:
        neg_qte = df[df[COL_QTE_VENDUE] < 0]
        if not neg_qte.empty:
            issues["quantite_vendue_negative"] = neg_qte
            print(f"⚠️  {len(neg_qte)} ligne(s) avec une quantité vendue négative.")

    if COL_STOCK_DISPO in df.columns:
        neg_stock = df[df[COL_STOCK_DISPO] < 0]
        if not neg_stock.empty:
            issues["stock_disponible_negatif"] = neg_stock
            print(f"⚠️  {len(neg_stock)} ligne(s) avec un stock disponible négatif.")

    if COL_PRIX_UNITAIRE in df.columns:
        bad_price = df[df[COL_PRIX_UNITAIRE] <= 0]
        if not bad_price.empty:
            issues["prix_unitaire_invalide"] = bad_price
            print(f"⚠️  {len(bad_price)} ligne(s) avec un prix unitaire nul ou négatif.")

    if COL_DELAI_FOURNISSEUR in df.columns:
        neg_delai = df[df[COL_DELAI_FOURNISSEUR] < 0]
        if not neg_delai.empty:
            issues["delai_fournisseur_negatif"] = neg_delai
            print(f"⚠️  {len(neg_delai)} ligne(s) avec un délai fournisseur négatif.")

    if COL_DATE_CMD_FOURNISSEUR in df.columns and COL_DATE_RECEPTION_PREVUE in df.columns:
        both_dates = df[
            df[COL_DATE_CMD_FOURNISSEUR].notna() & df[COL_DATE_RECEPTION_PREVUE].notna()
        ]
        incoherent_dates = both_dates[
            both_dates[COL_DATE_RECEPTION_PREVUE] < both_dates[COL_DATE_CMD_FOURNISSEUR]
        ]
        if not incoherent_dates.empty:
            issues["reception_avant_commande"] = incoherent_dates
            print(
                f"⚠️  {len(incoherent_dates)} ligne(s) où la date de réception prévue "
                f"est antérieure à la date de commande fournisseur."
            )

    if not issues:
        print("✅ Aucune valeur incohérente détectée parmi les vérifications effectuées.")

    return issues


def generate_summary(df: pd.DataFrame) -> None:
    """
    Affiche un résumé clair et lisible du dataset.
    """
    print("\n" + "=" * 60)
    print("RÉSUMÉ DU DATASET")
    print("=" * 60)
    print(f"Nombre de lignes        : {df.shape[0]}")
    print(f"Nombre de colonnes      : {df.shape[1]}")

    if COL_SKU in df.columns:
        print(f"Nombre de SKU distincts : {df[COL_SKU].nunique()}")

    if COL_DATE in df.columns and df[COL_DATE].notna().any():
        print(f"Période couverte        : du {df[COL_DATE].min().date()} au {df[COL_DATE].max().date()}")

    print("\nTypes de colonnes :")
    print(df.dtypes.to_string())
    print("=" * 60 + "\n")


def run_data_quality_report(filepath: Path) -> pd.DataFrame:
    """
    Fonction principale de l'étape 1 : orchestre toutes les vérifications.

    Retourne le DataFrame chargé (avec les dates converties), sans
    AUCUNE suppression de données. Le nettoyage effectif se fait
    ensuite dans data_cleaning.py.
    """
    print("### ÉTAPE 1 — IMPORT ET CONTRÔLE DES DONNÉES ###\n")

    df = load_excel_file(filepath)

    missing_cols = check_required_columns(df)
    if missing_cols:
        raise ValueError(
            "Le fichier Excel ne contient pas toutes les colonnes nécessaires. "
            "Corrige le fichier ou adapte src/config.py, puis relance."
        )

    df = convert_date_columns(df)

    print()
    detect_missing_values(df)
    print()
    detect_duplicates(df)
    print()
    detect_inconsistent_values(df)

    generate_summary(df)

    return df


if __name__ == "__main__":
    from src.config import DEFAULT_DATA_FILE
    run_data_quality_report(DEFAULT_DATA_FILE)
