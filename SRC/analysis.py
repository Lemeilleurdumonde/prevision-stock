"""
analysis.py
-------------
ÉTAPE 3 : Analyse exploratoire.

Ce module prend le DataFrame NETTOYÉ (sortie de data_cleaning.clean_data)
et calcule les indicateurs demandés : chiffre d'affaires, quantités
vendues, évolution, top/flop produits, saisonnalité, tendance, impact
des promotions — par mois, produit, SKU, catégorie et magasin.

Chaque fonction retourne un DataFrame simple, réutilisable aussi bien
dans un script que dans le dashboard Streamlit (étape 8).
"""

import pandas as pd

from src.config import (
    COL_DATE, COL_SKU, COL_PRODUIT, COL_CATEGORIE, COL_MAGASIN,
    COL_QTE_VENDUE, COL_PRIX_UNITAIRE, COL_PROMOTION,
)
from src.utils import saison_depuis_mois


def add_ca_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute une colonne 'Chiffre_affaires' = Quantité vendue x Prix unitaire."""
    df = df.copy()
    df["Chiffre_affaires"] = df[COL_QTE_VENDUE] * df[COL_PRIX_UNITAIRE]
    return df


def add_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute Mois, Trimestre, Année, Saison à partir de la colonne Date."""
    df = df.copy()
    df["Année"] = df[COL_DATE].dt.year
    df["Mois"] = df[COL_DATE].dt.month
    df["Trimestre"] = df[COL_DATE].dt.quarter
    df["Saison"] = df["Mois"].apply(saison_depuis_mois)
    return df


def ventes_par_periode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Évolution des ventes par mois calendaire (toutes séries confondues) :
    chiffre d'affaires et quantité vendue.
    """
    df = add_ca_column(df)
    df["Periode"] = df[COL_DATE].dt.to_period("M").dt.to_timestamp()

    resume = (
        df.groupby("Periode")
        .agg(
            Chiffre_affaires=("Chiffre_affaires", "sum"),
            Quantite_vendue=(COL_QTE_VENDUE, "sum"),
        )
        .reset_index()
        .sort_values("Periode")
    )
    return resume


def ventes_par_dimension(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """
    Agrège le chiffre d'affaires et la quantité vendue par une dimension
    donnée (ex: COL_PRODUIT, COL_SKU, COL_CATEGORIE, COL_MAGASIN).

    Trié par chiffre d'affaires décroissant.
    """
    df = add_ca_column(df)
    resume = (
        df.groupby(dimension)
        .agg(
            Chiffre_affaires=("Chiffre_affaires", "sum"),
            Quantite_vendue=(COL_QTE_VENDUE, "sum"),
            Prix_moyen=(COL_PRIX_UNITAIRE, "mean"),
        )
        .reset_index()
        .sort_values("Chiffre_affaires", ascending=False)
    )
    return resume


def top_flop_produits(df: pd.DataFrame, n: int = 5) -> tuple:
    """
    Retourne (top_n, flop_n) : les n produits les plus vendus et les
    moins vendus, en QUANTITÉ (plus parlant que le CA pour repérer les
    produits qui tournent peu, indépendamment de leur prix).
    """
    par_produit = ventes_par_dimension(df, COL_PRODUIT).sort_values(
        "Quantite_vendue", ascending=False
    )
    top = par_produit.head(n).reset_index(drop=True)
    flop = par_produit.tail(n).sort_values("Quantite_vendue").reset_index(drop=True)
    return top, flop


def saisonnalite_mensuelle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Moyenne des ventes par mois CALENDAIRE (1 à 12), toutes années
    confondues, pour repérer un éventuel motif saisonnier récurrent
    (ex: climatiseurs qui vendent plus en été).
    """
    df = add_calendar_columns(df)
    resume = (
        df.groupby("Mois")
        .agg(Quantite_vendue_moyenne=(COL_QTE_VENDUE, "mean"))
        .reindex(range(1, 13))  # garantit les 12 mois dans l'ordre, même si l'un est vide
        .reset_index()
    )
    return resume


def tendance_avec_moyenne_mobile(df: pd.DataFrame, fenetre: int = 3) -> pd.DataFrame:
    """
    Ventes mensuelles totales + moyenne mobile sur `fenetre` mois, pour
    visualiser la tendance générale en lissant les variations ponctuelles.
    """
    resume = ventes_par_periode(df).sort_values("Periode")
    resume["Moyenne_mobile"] = resume["Quantite_vendue"].rolling(
        window=fenetre, min_periods=1
    ).mean()
    return resume


def impact_promotion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare les ventes moyennes AVEC et SANS promotion, par produit.

    Donne une estimation simple de l'effet promo : un ratio > 1 signifie
    que le produit se vend mieux en promotion.
    """
    df = df.copy()
    resume = (
        df.groupby([COL_PRODUIT, COL_PROMOTION])
        .agg(Quantite_vendue_moyenne=(COL_QTE_VENDUE, "mean"))
        .reset_index()
        .pivot(index=COL_PRODUIT, columns=COL_PROMOTION, values="Quantite_vendue_moyenne")
    )

    for col in ["Oui", "Non"]:
        if col not in resume.columns:
            resume[col] = float("nan")

    resume = resume.rename(columns={"Oui": "Ventes_moy_avec_promo", "Non": "Ventes_moy_sans_promo"})
    resume["Effet_promo_ratio"] = (
        resume["Ventes_moy_avec_promo"] / resume["Ventes_moy_sans_promo"]
    ).round(2)

    return resume.reset_index().sort_values("Effet_promo_ratio", ascending=False)


def resume_general(df: pd.DataFrame) -> dict:
    """
    Indicateurs clés pour la page 1 du dashboard ('Vue générale') :
    CA total, unités vendues, nombre de SKU, période couverte.
    """
    df = add_ca_column(df)
    return {
        "chiffre_affaires_total": float(df["Chiffre_affaires"].sum()),
        "unites_vendues": int(df[COL_QTE_VENDUE].sum()),
        "nb_sku": int(df[COL_SKU].nunique()),
        "nb_magasins": int(df[COL_MAGASIN].nunique()),
        "date_min": df[COL_DATE].min(),
        "date_max": df[COL_DATE].max(),
    }


def generate_exploratory_charts(df: pd.DataFrame, output_dir) -> list:
    """
    Génère quelques graphiques simples avec matplotlib et les sauvegarde
    en PNG dans `output_dir`. Retourne la liste des fichiers créés.

    Ces graphiques sont un aperçu rapide en dehors du dashboard ; le
    dashboard Streamlit (étape 8) propose des versions interactives.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fichiers = []

    # 1. Évolution mensuelle des ventes
    evolution = ventes_par_periode(df)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(evolution["Periode"], evolution["Quantite_vendue"], marker="o")
    ax.set_title("Évolution mensuelle des quantités vendues")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Unités vendues")
    fig.autofmt_xdate()
    fig.tight_layout()
    path1 = output_dir / "01_evolution_mensuelle.png"
    fig.savefig(path1)
    plt.close(fig)
    fichiers.append(path1)

    # 2. Top produits (par quantité vendue)
    top, _ = top_flop_produits(df, n=10)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(top[COL_PRODUIT], top["Quantite_vendue"], color="#2E86C1")
    ax.invert_yaxis()
    ax.set_title("Top produits (quantité vendue)")
    ax.set_xlabel("Unités vendues")
    fig.tight_layout()
    path2 = output_dir / "02_top_produits.png"
    fig.savefig(path2)
    plt.close(fig)
    fichiers.append(path2)

    # 3. Saisonnalité mensuelle
    saison = saisonnalite_mensuelle(df)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(saison["Mois"], saison["Quantite_vendue_moyenne"], color="#28B463")
    ax.set_title("Saisonnalité : ventes moyennes par mois calendaire")
    ax.set_xlabel("Mois (1=Janvier ... 12=Décembre)")
    ax.set_ylabel("Unités vendues (moyenne)")
    ax.set_xticks(range(1, 13))
    fig.tight_layout()
    path3 = output_dir / "03_saisonnalite.png"
    fig.savefig(path3)
    plt.close(fig)
    fichiers.append(path3)

    # 4. Impact des promotions
    promo = impact_promotion(df)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(len(promo))
    ax.bar([i - 0.2 for i in x], promo["Ventes_moy_sans_promo"], width=0.4, label="Sans promo")
    ax.bar([i + 0.2 for i in x], promo["Ventes_moy_avec_promo"], width=0.4, label="Avec promo")
    ax.set_xticks(list(x))
    ax.set_xticklabels(promo[COL_PRODUIT], rotation=45, ha="right")
    ax.set_title("Impact des promotions sur les ventes moyennes")
    ax.set_ylabel("Unités vendues (moyenne)")
    ax.legend()
    fig.tight_layout()
    path4 = output_dir / "04_impact_promotion.png"
    fig.savefig(path4)
    plt.close(fig)
    fichiers.append(path4)

    print(f"✅ {len(fichiers)} graphiques générés dans {output_dir}")
    return fichiers


if __name__ == "__main__":
    from src.config import DEFAULT_DATA_FILE, OUTPUTS_DIR
    from src.data_loader import run_data_quality_report
    from src.data_cleaning import clean_data

    df_raw = run_data_quality_report(DEFAULT_DATA_FILE)
    df_clean = clean_data(df_raw)

    print("\n### ÉTAPE 3 — ANALYSE EXPLORATOIRE ###\n")
    print("Résumé général :", resume_general(df_clean))
    print("\nVentes par catégorie :\n", ventes_par_dimension(df_clean, COL_CATEGORIE))
    print("\nVentes par magasin :\n", ventes_par_dimension(df_clean, COL_MAGASIN))

    top, flop = top_flop_produits(df_clean)
    print("\nTop produits :\n", top)
    print("\nFlop produits :\n", flop)

    print("\nSaisonnalité mensuelle :\n", saisonnalite_mensuelle(df_clean))
    print("\nImpact promotion :\n", impact_promotion(df_clean))

    generate_exploratory_charts(df_clean, OUTPUTS_DIR / "graphiques")
