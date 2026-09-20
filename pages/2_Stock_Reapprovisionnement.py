"""
2_Stock_Reapprovisionnement.py
---------------------------------
Page 3 du dashboard : tableau de gestion du stock et recommandations
de commande, avec code couleur par niveau de risque (vert/orange/rouge)
et filtre sur les produits à risque.
"""

import streamlit as st

from src.app_data import charger_donnees_propres, charger_previsions, charger_tableau_stock
from src.utils import couleur_risque

st.set_page_config(page_title="Stock & Réapprovisionnement", page_icon="📦", layout="wide")
st.title("📦 Stock & Réapprovisionnement")

df = charger_donnees_propres()
_, previsions = charger_previsions(df)
tableau = charger_tableau_stock(df, previsions)

afficher_uniquement_risque = st.checkbox(
    "Afficher uniquement les produits à risque (moyen ou élevé)", value=False
)

tableau_affiche = tableau.copy()
if afficher_uniquement_risque:
    tableau_affiche = tableau_affiche[tableau_affiche["Risque_rupture"] != "Faible"]

noms_colonnes_affiches = {
    "SKU": "SKU", "Produit": "Produit", "Magasin": "Magasin",
    "Ventes_prevues_mois_prochain": "Ventes prévues (M+1)",
    "Stock disponible": "Stock actuel", "Stock_en_transit": "Stock en transit",
    "Délai fournisseur (jours)": "Délai fournisseur (j)",
    "Demande_pendant_delai": "Demande pendant délai",
    "Stock_securite": "Stock de sécurité", "Stock_projete": "Stock projeté",
    "Risque_rupture": "Risque de rupture", "Besoin_futur": "Besoin futur",
    "Quantite_recommandee": "Qté recommandée",
    "Date_recommandee_commande": "Date recommandée de commande",
}
tableau_affiche = tableau_affiche.rename(columns=noms_colonnes_affiches)


def _style_risque(valeur):
    couleur = couleur_risque(valeur)
    return f"background-color: {couleur}; color: white; font-weight: 600;"


# NB: Styler.applymap est marqué "deprecated" dans les versions très
# récentes de pandas (au profit de .map) mais reste fonctionnel ; on le
# garde ici pour la compatibilité avec un plus large éventail de versions.
st.dataframe(
    tableau_affiche.style.map(_style_risque, subset=["Risque de rupture"]),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "🟢 Risque faible — le stock de sécurité est couvert. "
    "🟠 Risque moyen — le stock projeté restera positif mais sous le stock de sécurité. "
    "🔴 Risque élevé — rupture de stock anticipée avant la fin du délai fournisseur."
)

nb_a_commander = int((tableau["Quantite_recommandee"] > 0).sum())
if nb_a_commander > 0:
    st.warning(f"⚠️ {nb_a_commander} produit(s)/magasin(s) nécessitent une commande dès maintenant.")
else:
    st.success("✅ Aucune commande urgente à passer actuellement.")

with st.expander("Comment sont calculées ces recommandations ?"):
    st.markdown(
        """
        - **Demande pendant le délai** : somme des ventes prévues sur les mois couverts par le délai fournisseur.
        - **Stock de sécurité** = 1,65 × écart-type de la demande mensuelle × racine(délai en mois) — une marge pour absorber les imprévus (niveau de service ≈ 95 %).
        - **Stock projeté** = stock actuel + stock en transit − demande pendant le délai.
        - **Besoin futur** = demande pendant le délai + stock de sécurité − stock actuel − stock en transit.
        - **Quantité recommandée** = max(0, besoin futur).
        - Si une quantité est recommandée, on suggère de commander **dès maintenant** (le délai fournisseur est déjà pris en compte dans le calcul de la demande à couvrir).
        """
    )
