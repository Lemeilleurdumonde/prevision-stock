"""
config.py
-----------
Configuration centrale du prototype.

Toutes les valeurs qui pourraient changer selon l'entreprise ou le
fichier source (noms de colonnes, chemins de fichiers, paramètres
métier) sont définies ICI, une seule fois, plutôt que d'être codées
en dur dans chaque module.

Si un jour ton fichier Excel a des noms de colonnes différents, c'est
le SEUL endroit à modifier.
"""

from pathlib import Path

# ----------------------------------------------------------------------
# Chemins du projet
# ----------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = ROOT_DIR / "models"

DEFAULT_DATA_FILE = DATA_DIR / "Donnees_Prototype_IA.xlsx"

# ----------------------------------------------------------------------
# Noms des colonnes attendues dans le fichier Excel
# (centralisés ici pour éviter les fautes de frappe répétées dans le code)
# ----------------------------------------------------------------------
COL_DATE = "Date"
COL_SKU = "SKU"
COL_PRODUIT = "Produit"
COL_CATEGORIE = "Catégorie"
COL_MAGASIN = "Magasin"
COL_QTE_VENDUE = "Quantité vendue"
COL_PRIX_UNITAIRE = "Prix unitaire (MAD)"
COL_STOCK_DISPO = "Stock disponible"
COL_QTE_COMMANDEE = "Quantité commandée"
COL_DATE_CMD_FOURNISSEUR = "Date commande fournisseur"
COL_DATE_RECEPTION_PREVUE = "Date réception prévue"
COL_DELAI_FOURNISSEUR = "Délai fournisseur (jours)"
COL_PROMOTION = "Promotion"

REQUIRED_COLUMNS = [
    COL_DATE, COL_SKU, COL_PRODUIT, COL_CATEGORIE, COL_MAGASIN,
    COL_QTE_VENDUE, COL_PRIX_UNITAIRE, COL_STOCK_DISPO, COL_QTE_COMMANDEE,
    COL_DATE_CMD_FOURNISSEUR, COL_DATE_RECEPTION_PREVUE,
    COL_DELAI_FOURNISSEUR, COL_PROMOTION,
]

# Colonnes à convertir en date
DATE_COLUMNS = [COL_DATE, COL_DATE_CMD_FOURNISSEUR, COL_DATE_RECEPTION_PREVUE]

# Colonnes strictement numériques
NUMERIC_COLUMNS = [
    COL_QTE_VENDUE, COL_PRIX_UNITAIRE, COL_STOCK_DISPO,
    COL_QTE_COMMANDEE, COL_DELAI_FOURNISSEUR,
]

# Colonnes textuelles à uniformiser (espaces, casse)
TEXT_COLUMNS = [COL_SKU, COL_PRODUIT, COL_CATEGORIE, COL_MAGASIN]

# La colonne Promotion est catégorielle avec des valeurs "Oui" / "Non"
# dans le fichier fourni (et non 0/1). On la traite séparément.
PROMOTION_VALEURS_VALIDES = ["Oui", "Non"]
PROMOTION_VALEUR_PAR_DEFAUT = "Non"

# ----------------------------------------------------------------------
# Paramètres de la clé de série temporelle
# ----------------------------------------------------------------------
# Chaque série de vente est identifiée par la combinaison (SKU, Magasin) :
# c'est à ce niveau que le stock, les commandes fournisseur et les délais
# sont enregistrés dans le fichier, donc c'est aussi à ce niveau qu'on
# prévoit la demande et qu'on calcule les recommandations de commande.
SERIES_KEYS = [COL_SKU, COL_MAGASIN]

# ----------------------------------------------------------------------
# Paramètres de saisonnalité / calendrier
# ----------------------------------------------------------------------
SEASON_BY_MONTH = {
    12: "Hiver", 1: "Hiver", 2: "Hiver",
    3: "Printemps", 4: "Printemps", 5: "Printemps",
    6: "Été", 7: "Été", 8: "Été",
    9: "Automne", 10: "Automne", 11: "Automne",
}

# ----------------------------------------------------------------------
# Paramètres de prévision (ÉTAPE 4-5)
# ----------------------------------------------------------------------
# Fenêtres utilisées pour les moyennes mobiles / variables retardées
ROLLING_WINDOWS = [3, 6]

# Nombre de mois utilisés comme période de test lors du backtest
# (le reste sert à l'entraînement des modèles)
BACKTEST_HORIZON_MOIS = 3

# Nombre minimum de points historiques nécessaires pour qu'une série
# soit prévisible (en dessous, on se contente de la moyenne historique)
MIN_POINTS_POUR_PREVISION = 6

# Nombre de points par cycle saisonnier (12 = saisonnalité annuelle,
# adapté à des données mensuelles)
SAISON_PERIODES = 12

# Horizons de prévision proposés dans le dashboard (en mois)
HORIZONS_PREVISION = [1, 3, 6]

# Paramètres du modèle Random Forest
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 5
RF_RANDOM_STATE = 42

# ----------------------------------------------------------------------
# Paramètres de gestion de stock (ÉTAPE 6-7)
# ----------------------------------------------------------------------
# Niveau de service visé pour le stock de sécurité (95 % -> z = 1.65).
# Plus il est proche de 1, plus le stock de sécurité est important.
# Formule utilisée : stock_securite = Z * écart-type(demande mensuelle) * racine(délai en mois)
NIVEAU_SERVICE_Z = 1.65

# Nombre de jours dans un "mois" pour convertir le délai fournisseur
# (exprimé en jours) en mois, et l'aligner sur des prévisions mensuelles.
JOURS_PAR_MOIS = 30
