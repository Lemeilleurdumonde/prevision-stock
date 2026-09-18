"""
test_full_pipeline.py
------------------------
Script de test bout-en-bout : exécute toutes les étapes du pipeline
(sauf le dashboard Streamlit, qui se lance séparément) et vérifie que
tout s'enchaîne sans erreur.

Utilisation (depuis la racine du projet) :
    python -m tests.test_full_pipeline
"""

from src.config import DEFAULT_DATA_FILE, OUTPUTS_DIR
from src.data_loader import run_data_quality_report
from src.data_cleaning import clean_data
from src.analysis import resume_general, generate_exploratory_charts
from src.forecasting import backtester_toutes_series, selectionner_meilleur_modele, generer_previsions
from src.inventory import construire_tableau_reapprovisionnement


def main():
    print("############################################")
    print("# TEST BOUT-EN-BOUT DU PIPELINE")
    print("############################################\n")

    # Étapes 1-2
    df_raw = run_data_quality_report(DEFAULT_DATA_FILE)
    df_clean = clean_data(df_raw)

    # Étape 3
    print("\n### ÉTAPE 3 — Analyse exploratoire ###")
    resume = resume_general(df_clean)
    print("Résumé général :", resume)
    generate_exploratory_charts(df_clean, OUTPUTS_DIR / "graphiques")

    # Étapes 4-5
    print("\n### ÉTAPES 4-5 — Prévision ###")
    resultats_backtest = backtester_toutes_series(df_clean)
    meilleurs_modeles = selectionner_meilleur_modele(resultats_backtest)
    previsions = generer_previsions(df_clean, meilleurs_modeles)
    print(f"Backtest : {len(resultats_backtest)} lignes. Meilleurs modèles : {len(meilleurs_modeles)} séries.")
    print("Répartition des modèles gagnants :")
    print(meilleurs_modeles["Modele"].value_counts().to_string())

    # Étapes 6-7
    print("\n### ÉTAPES 6-7 — Stock & recommandation de commande ###")
    tableau_stock = construire_tableau_reapprovisionnement(df_clean, previsions)
    print(tableau_stock.to_string(index=False))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    chemin_sortie = OUTPUTS_DIR / "recommandations_reapprovisionnement.csv"
    tableau_stock.to_csv(chemin_sortie, index=False)
    print(f"\n💾 Tableau de recommandations sauvegardé dans : {chemin_sortie}")

    print("\n✅ Pipeline complet exécuté sans erreur (hors dashboard Streamlit).")


if __name__ == "__main__":
    main()
