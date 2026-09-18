"""
1_Prevision.py
-----------------
Page 2 du dashboard : Prévision des ventes par produit / SKU / magasin,
avec l'historique, la courbe de prévision et les métriques du modèle.
"""

import pandas as pd
import streamlit as st

from src.app_data import charger_donnees_propres, charger_previsions
from src.forecasting import preparer_serie
from src.config import COL_DATE, COL_QTE_VENDUE, COL_PRODUIT, COL_SKU, COL_MAGASIN, HORIZONS_PREVISION

st.set_page_config(page_title="Prévision des ventes", page_icon="🔮", layout="wide")
st.title("🔮 Prévision des ventes")

df = charger_donnees_propres()
meilleurs_modeles, previsions = charger_previsions(df)

col1, col2, col3 = st.columns(3)

produit_liste = sorted(df[COL_PRODUIT].unique())
produit_choisi = col1.selectbox("Produit", produit_liste)

sku_liste = sorted(df[df[COL_PRODUIT] == produit_choisi][COL_SKU].unique())
sku_choisi = col2.selectbox("SKU", sku_liste)

magasin_liste = sorted(df[df[COL_SKU] == sku_choisi][COL_MAGASIN].unique())
magasin_choisi = col3.selectbox("Magasin", magasin_liste)

horizon_choisi = st.select_slider(
    "Horizon de prévision (mois)", options=HORIZONS_PREVISION, value=HORIZONS_PREVISION[0]
)

# --- Historique ---
serie = preparer_serie(df, sku_choisi, magasin_choisi)
historique = serie[[COL_DATE, COL_QTE_VENDUE]].rename(
    columns={COL_DATE: "Date", COL_QTE_VENDUE: "Ventes"}
)
historique["Type"] = "Historique"

# --- Prévision, limitée à l'horizon choisi ---
prevision_serie = (
    previsions[(previsions["SKU"] == sku_choisi) & (previsions["Magasin"] == magasin_choisi)]
    .sort_values("Date")
    .head(horizon_choisi)
)
prevision_aff = prevision_serie[["Date", "Prevision"]].rename(columns={"Prevision": "Ventes"})
prevision_aff["Type"] = "Prévision"

combine = pd.concat([historique, prevision_aff], ignore_index=True)
pivot = combine.pivot(index="Date", columns="Type", values="Ventes")

st.subheader(f"{produit_choisi} — {sku_choisi} — {magasin_choisi}")
st.line_chart(pivot)
st.caption(
    "La courbe 'Historique' montre les ventes réelles passées ; la courbe "
    "'Prévision' montre les ventes estimées pour les prochains mois."
)

# --- Métriques du modèle sélectionné pour cette série ---
ligne_modele = meilleurs_modeles[
    (meilleurs_modeles["SKU"] == sku_choisi) & (meilleurs_modeles["Magasin"] == magasin_choisi)
]

if not ligne_modele.empty:
    ligne = ligne_modele.iloc[0]
    st.subheader("Modèle sélectionné (backtest)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Modèle retenu", ligne["Modele"])
    m2.metric("WAPE (erreur %)", f"{ligne['WAPE']:.1f}%")
    m3.metric("MAE", f"{ligne['MAE']:.1f} unités")
    m4.metric("Biais", f"{ligne['Biais']:+.1f} unités")
    st.caption(
        "Le modèle est choisi automatiquement parmi plusieurs candidats "
        "(moyennes de référence, Holt-Winters, Random Forest) selon leur "
        "erreur sur les 3 derniers mois connus (backtest). WAPE = erreur "
        "absolue moyenne pondérée ; Biais > 0 = le modèle a tendance à "
        "sur-prévoir, Biais < 0 = à sous-prévoir."
    )
else:
    st.info("Pas assez d'historique pour cette série pour évaluer un modèle par backtest.")

st.subheader("Détail des prévisions")
st.dataframe(
    prevision_serie[["Date", "Prevision", "Modele_utilise"]]
    .rename(columns={"Prevision": "Ventes prévues", "Modele_utilise": "Modèle utilisé"})
    .reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
)
