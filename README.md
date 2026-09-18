# Prototype IA — Prévision de la demande & gestion des stocks

Prototype fonctionnel (MVP) qui analyse les ventes historiques pour :
prévoir la demande par produit, détecter la saisonnalité, anticiper
les risques de rupture de stock, et recommander quoi/quand commander.

## 1. Installer Python

Le prototype nécessite **Python 3.10 ou plus récent**.

- **Windows** : télécharge Python sur https://www.python.org/downloads/
  et coche bien la case "Add python.exe to PATH" pendant l'installation.
- **macOS** : `brew install python3` (avec [Homebrew](https://brew.sh)),
  ou télécharge l'installeur sur python.org.
- **Linux** : Python est généralement déjà installé (`python3 --version`
  pour vérifier) ; sinon `sudo apt install python3 python3-venv`.

Vérifie l'installation :
```bash
python3 --version
```

## 2. Créer un environnement virtuel

Depuis le dossier racine du projet (`project/`) :

```bash
python3 -m venv venv
```

Puis active-le :
```bash
# macOS / Linux
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# Windows (cmd)
venv\Scripts\activate.bat
```

Ton terminal doit maintenant afficher `(venv)` au début de la ligne.
**Pense à réactiver cet environnement à chaque nouvelle session de
travail**, avec la même commande.

## 3. Installer les dépendances

Toujours avec l'environnement virtuel activé :
```bash
pip install -r requirements.txt
```

## 4. Où placer le fichier Excel

Place ton fichier de données dans le dossier `data/`, avec exactement
ce nom :
```
project/data/Donnees_Prototype_IA.xlsx
```
(Un fichier d'exemple, reconstruit à partir de tes données, y est déjà présent.)

Les colonnes attendues sont définies dans `src/config.py` — si le nom
d'une colonne change dans ton fichier, c'est le seul endroit à modifier.

## 5. Lancer et tester le prototype (sans le dashboard)

Chaque étape peut être testée indépendamment, depuis la racine du projet :

```bash
# Étape 1-2 : import, contrôle qualité et nettoyage des données
python -m tests.test_step1_2

# Étape 3 : analyse exploratoire (génère des graphiques dans outputs/graphiques)
python -m src.analysis

# Étapes 4-5 : prévision des ventes (backtest + choix du meilleur modèle par série)
python -m src.forecasting

# Étapes 6-7 : gestion du stock et recommandations de commande
python -m src.inventory

# Tout le pipeline d'un coup (recommandé pour une vérification rapide)
python -m tests.test_full_pipeline
```

Le dernier script sauvegarde aussi le tableau de recommandations dans
`outputs/recommandations_reapprovisionnement.csv`.

## 6. Lancer le dashboard Streamlit

```bash
streamlit run app.py
```

Ton navigateur s'ouvre automatiquement sur `http://localhost:8501`. Le
menu à gauche donne accès aux 3 pages :
- **Vue générale** (`app.py`) : indicateurs clés, évolution des ventes, top produits/catégories.
- **Prévision** (`pages/1_Prevision.py`) : sélection produit/SKU/magasin/horizon, courbe historique + prévision, métriques du modèle.
- **Stock & Réapprovisionnement** (`pages/2_Stock_Reapprovisionnement.py`) : tableau des recommandations avec code couleur (🟢🟠🔴) et filtre sur les produits à risque.

La première ouverture de chaque page peut prendre quelques secondes
(calcul des modèles) ; les exécutions suivantes sont mises en cache.

## Architecture du projet

```
project/
├── app.py                              # Dashboard Streamlit — page 1 (Vue générale)
├── pages/
│   ├── 1_Prevision.py                  # Dashboard — page 2 (Prévision)
│   └── 2_Stock_Reapprovisionnement.py  # Dashboard — page 3 (Stock & réappro)
├── data/
│   └── Donnees_Prototype_IA.xlsx
├── src/
│   ├── config.py          # Toute la configuration centralisée (colonnes, chemins, paramètres)
│   ├── utils.py            # Fonctions utilitaires partagées
│   ├── data_loader.py      # Étape 1 — import et contrôle qualité (aucune suppression)
│   ├── data_cleaning.py    # Étape 2 — nettoyage (retourne un nouveau DataFrame)
│   ├── analysis.py         # Étape 3 — analyse exploratoire + graphiques
│   ├── forecasting.py      # Étapes 4-5 — prévision (baselines, Holt-Winters, Random Forest, backtest)
│   ├── inventory.py        # Étapes 6-7 — stock de sécurité et recommandations de commande
│   └── app_data.py         # Fonctions mises en cache pour le dashboard Streamlit
├── tests/
│   ├── test_step1_2.py         # Test étapes 1-2
│   └── test_full_pipeline.py   # Test bout-en-bout (étapes 1 à 7)
├── outputs/                # Graphiques et tableaux générés (créé automatiquement)
├── models/                 # Réservé pour une future sauvegarde de modèles entraînés
├── requirements.txt
└── README.md
```

## Hypothèses importantes du prototype (MVP)

Ces choix sont documentés dans le code (commentaires) et peuvent être
ajustés dans `src/config.py` :

- **Granularité de prévision** : chaque combinaison (SKU, Magasin) est
  traitée comme une série indépendante, car c'est à ce niveau que le
  stock et les commandes fournisseur sont gérés dans le fichier.
- **Quantités/stocks négatifs** : considérés comme des erreurs de
  saisie (pas des retours produits) et neutralisés lors du nettoyage.
- **Promotions futures** : on suppose qu'aucune promotion n'est prévue
  sur l'horizon de prévision (hypothèse simplificatrice).
- **Stock de sécurité** : formule `Z × écart-type(demande mensuelle) ×
  racine(délai en mois)`, avec Z = 1,65 (niveau de service ≈ 95 %).
- **Historique limité (24 points/série)** : avec seulement 2 ans de
  données mensuelles, les baselines simples gagnent souvent le
  backtest face à des modèles plus complexes — c'est un comportement
  normal et volontaire (le système ne choisit pas automatiquement le
  modèle le plus sophistiqué).
- **Holt-Winters** nécessite la librairie `statsmodels` ; si elle
  n'est pas installée, ce modèle est simplement exclu de la
  comparaison (le reste du pipeline continue de fonctionner).

## Limites connues (prototype, pas un produit fini)

- Peu de données par série (24 mois) : à surveiller à mesure que
  l'historique s'allonge.
- Le stock en transit ne prend en compte que les commandes présentes
  dans le fichier historique fourni.
- Pas d'authentification ni de base de données : conçu pour tourner
  en local, sur un seul poste.
