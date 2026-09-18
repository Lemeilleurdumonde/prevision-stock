"""
utils.py
----------
Petites fonctions utilitaires partagées par plusieurs modules
(formatage de nombres, calcul de saison, etc.).

Regrouper ces fonctions ici évite de dupliquer du code dans
analysis.py, forecasting.py, inventory.py et le dashboard Streamlit.
"""

import pandas as pd

from src.config import SEASON_BY_MONTH


def saison_depuis_mois(mois: int) -> str:
    """Retourne le nom de la saison (Hiver/Printemps/Été/Automne) pour un mois donné (1-12)."""
    return SEASON_BY_MONTH.get(int(mois), "Inconnue")


def format_mad(valeur: float) -> str:
    """Formate un nombre en devise MAD, avec séparateur de milliers lisible."""
    if pd.isna(valeur):
        return "N/A"
    return f"{valeur:,.0f} MAD".replace(",", " ")


def format_pourcentage(valeur: float, decimales: int = 1) -> str:
    """Formate un nombre (0-1 ou déjà en %) en pourcentage lisible."""
    if pd.isna(valeur):
        return "N/A"
    return f"{valeur:.{decimales}f}%"


def couleur_risque(niveau_risque: str) -> str:
    """
    Retourne un code couleur hexadécimal associé à un niveau de risque,
    utilisé pour l'affichage visuel dans le dashboard Streamlit.

    vert = faible risque, orange = risque moyen, rouge = risque élevé.
    """
    mapping = {
        "Faible": "#2ECC71",
        "Moyen": "#F39C12",
        "Élevé": "#E74C3C",
    }
    return mapping.get(niveau_risque, "#95A5A6")  # gris si valeur inattendue
