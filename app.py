"""
app.py
--------
Point d'entrée du dashboard Streamlit — Page 1 : Vue générale.

Lancer avec (depuis la racine du projet) :
    streamlit run app.py
"""

import streamlit as st

from src.app_data import charger_donnees_propres
from src.analysis import resume_general, ventes_par_periode, ventes_par_dimension
from src.config import COL_CATEGORIE, COL_PRODUIT
from src.utils import format_mad

st.set_page_config(page_title="Prévision & Stock — Vue générale", page_icon="📊", layout="wide")

st.title("📊 Vue générale")
st.caption("Prototype IA — prévision de la demande & gestion des stocks")

df = charger_donnees_propres()
resume = resume_general(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Chiffre d'affaires total", format_mad(resume["chiffre_affaires_total"]))
col2.metric("Unités vendues", f"{resume['unites_vendues']:,}".replace(",", " "))
col3.metric("Nombre de SKU", resume["nb_sku"])
col4.metric("Magasins", resume["nb_magasins"])

st.caption(
    f"Période couverte : du {resume['date_min'].date()} au {resume['date_max'].date()}"
)

st.divider()

st.subheader("Évolution mensuelle des ventes")
evolution = ventes_par_periode(df)
st.line_chart(evolution.set_index("Periode")[["Quantite_vendue"]])

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Top produits (chiffre d'affaires)")
    par_produit = ventes_par_dimension(df, COL_PRODUIT).head(10)
    st.bar_chart(par_produit.set_index(COL_PRODUIT)["Chiffre_affaires"])

with col_b:
    st.subheader("Ventes par catégorie")
    par_categorie = ventes_par_dimension(df, COL_CATEGORIE)
    st.bar_chart(par_categorie.set_index(COL_CATEGORIE)["Chiffre_affaires"])

st.divider()
st.caption("👈 Utilise le menu à gauche pour accéder aux pages **Prévision** et **Stock & Réapprovisionnement**.")
