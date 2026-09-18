"""
app_data.py
-------------
Fonctions de chargement/calcul PARTAGÉES par les pages du dashboard
Streamlit, mises en cache avec @st.cache_data pour éviter de relancer
le nettoyage des données et le backtest des modèles à chaque clic.

C'est le SEUL module qui mélange logique métier et Streamlit ; tous les
autres modules de src/ restent utilisables indépendamment (scripts,
tests, notebooks...).
"""

import streamlit as st

from src.config import DEFAULT_DATA_FILE
from src.data_loader import run_data_quality_report
from src.data_cleaning import clean_data
from src.forecasting import backtester_toutes_series, selectionner_meilleur_modele, generer_previsions
from src.inventory import construire_tableau_reapprovisionnement


@st.cache_data(show_spinner="Chargement et nettoyage des données...")
def charger_donnees_propres():
    """Charge le fichier Excel, contrôle sa qualité (étape 1) et le nettoie (étape 2)."""
    df_raw = run_data_quality_report(DEFAULT_DATA_FILE)
    df_clean = clean_data(df_raw)
    return df_clean


@st.cache_data(show_spinner="Calcul des prévisions (backtest de plusieurs modèles par série)...")
def charger_previsions(df):
    """Backteste tous les modèles, choisit le meilleur par série, génère les prévisions (étapes 4-5)."""
    resultats_backtest = backtester_toutes_series(df)
    meilleurs_modeles = selectionner_meilleur_modele(resultats_backtest)
    previsions = generer_previsions(df, meilleurs_modeles)
    return meilleurs_modeles, previsions


@st.cache_data(show_spinner="Calcul des recommandations de stock...")
def charger_tableau_stock(df, previsions):
    """Construit le tableau de gestion de stock et de recommandation de commande (étapes 6-7)."""
    return construire_tableau_reapprovisionnement(df, previsions)
