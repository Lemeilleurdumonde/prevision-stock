"""
test_step1_2.py
-----------------
Script de test manuel pour les étapes 1 et 2 (import/contrôle + nettoyage).

Utilisation (depuis la racine du projet) :
    python -m tests.test_step1_2

Ce script :
1. charge le fichier Excel (data/Donnees_Prototype_IA.xlsx) ;
2. lance les contrôles qualité (étape 1, aucune suppression) ;
3. lance le nettoyage (étape 2) ;
4. affiche un aperçu du résultat ;
5. sauvegarde le DataFrame nettoyé dans outputs/ (CSV), pour inspection,
   SANS jamais toucher au fichier Excel original.
"""

from src.config import DEFAULT_DATA_FILE, OUTPUTS_DIR
from src.data_loader import run_data_quality_report
from src.data_cleaning import clean_data


def main():
    if not DEFAULT_DATA_FILE.exists():
        print(
            f"❌ Fichier introuvable : {DEFAULT_DATA_FILE}\n"
            f"Place ton fichier Excel à cet emplacement exact, puis relance ce script."
        )
        return

    # Étape 1 — import et contrôle (pas de suppression)
    df_raw = run_data_quality_report(DEFAULT_DATA_FILE)

    # Étape 2 — nettoyage (retourne un NOUVEAU DataFrame propre)
    df_clean = clean_data(df_raw)

    print("Aperçu des données nettoyées (10 premières lignes) :")
    print(df_clean.head(10).to_string())

    # Sauvegarde pour inspection — le fichier Excel original n'est jamais modifié
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / "donnees_nettoyees.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"\n💾 Données nettoyées sauvegardées dans : {output_path}")


if __name__ == "__main__":
    main()
