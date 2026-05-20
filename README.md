# 🧠 Conception d’un système décisionnel intelligent pour POS

## Prédiction des ventes et du stock par l’IA et visualisation des données

> Projet de Master - Système POS (Point Of Sale)

## 📌 Contexte et problématique

Les systèmes de caisse (POS) cloud génèrent quotidiennement un volume important de données liées aux ventes, aux stocks et aux comportements des clients. Cependant, ces données sont souvent :

- exploitées de manière **descriptive uniquement**
- faiblement utilisées pour la **prévision et l’aide à la décision**
- peu intégrées dans des outils de **visualisation intelligents**

**Problématique :**
> Comment concevoir un système décisionnel intelligent permettant d’exploiter les données issues d’un POS afin de prédire les ventes et les niveaux de stock à l’aide de l’IA, tout en offrant une visualisation claire et interactive des résultats ?

## 🎯 Objectif général

Concevoir et implémenter un **système décisionnel intelligent** intégré à un POS cloud, combinant :

- Des **modèles d’intelligence artificielle** pour la prédiction des ventes et du stock
- Des **tableaux de bord interactifs** pour la visualisation

## 📊 Objectifs spécifiques

Objectif :
1. Analyser l’historique des ventes 
2.Développer des modèles de prévision de la demande
3.Anticiper les ruptures de stock  
4.Générer des recommandations décisionnelles 
5. Visualiser les résultats dans un dashboard 

## 🔄 Démarche méthodologique

### 1. Génération des données (2 ans)

 *Problème initial : données insuffisantes pour l’apprentissage*

Une étape de **génération de données synthétiques** a été réalisée pour couvrir une période de **2 ans** à partir des données existantes.

**Fichiers :**
- `data_generator/generator.py` - Génération intelligente avec saisonnalité
- `complete_missing.py` - Complétion pour atteindre 300K transactions

| Élément              | Valeur |
| Période générée      | 2024-2026 |
| Transactions finales | 300 000+ |
| Produits             | 6 875 |
| Patterns intégrés    | Ramadan, weekends, saisons, fêtes |

### 2. Prétraitement des données (ETL)

**Pipeline ETL** (`etl/pipeline.py`) :

| Étape          | Action |
|-------         |--------|
| Extraction     | Jointure des tables (transactions, produits, paiements) |
| Nettoyage      | Suppression doublons, outliers, valeurs manquantes |
| Enrichissement | Ajout de 30+ features temporelles |
| Chargement     | Création table `pos_analytics` (770 106 lignes) |

**Features ajoutées :**
- Date, heure, jour de semaine, weekend
- Périodes spéciales : Ramadan, Aïd, rentrée scolaire
- Marge brute, indicateurs de vente

### 3. Analyse exploratoire des données (EDA)

**Principales analyses réalisées :**

```sql
-- Impact du Ramadan sur les ventes
SELECT is_ramadan, AVG(line_total) FROM pos_analytics GROUP BY is_ramadan;
-- Résultat : Baisse de 3.9% du panier moyen pendant le Ramadan
sql
-- Top 10 produits les plus vendus
-- 1. Huile d'olive extra vierge (10 914 unités, 355 809 TND)
-- 2. Thon à l'huile végétale (5 468 unités, 109 701 TND)
Découverte	Impact
Ramadan → baisse panier moyen	Modèles doivent capturer saisonnalité
Produits alimentaires dominent	Priorité gestion stock
Forte variabilité journalière	Nécessité modèles temporels (LSTM, GRU)

### 4. Feature Engineering pour le stock
Script : stock/1-stock_features.py

Indicateurs calculés par produit :

Feature	Signification
velocity_7d / velocity_30d	Vitesse de vente
days_to_stockout	Jours avant rupture
rupture_score (0 à 1)	Score de risque
stock_status	CRITIQUE / ALERTE / FAIBLE / NORMAL
reorder_qty	Quantité recommandée
Résultats :

5 759 121 lignes de features générées

6 875 produits analysés

5. Modèles de Machine Learning
Modèles entraînés et comparés :

Modèle            	R²	     MAPE
Régression Linéaire	0.555	   4.36%
Ridge             	0.558	   4.38%
Random Forest	      0.801	   4.52%
LightGBM	          0.824	   3.98%
XGBoost	            0.848	   3.05%

6. Modèles de Deep Learning
Modèle	R²    	MAPE
MLP	    0.656	  9.99%
RNN	    0.416   32%
LSTM	  0.496	  15.09%
GRU	    0.484	  15.97%
Conclusion : XGBoost reste le plus performant pour ce cas d'usage.

7. Prédiction et gestion du stock
Script : stock/2-demand_forecast_xgboost.py

Processus :

Prédiction du CA total sur 14 jours

Répartition du CA par produit (selon poids historique)

Conversion CA → Quantités vendues

Calcul du stock projeté

Détection des ruptures

Génération d'alertes prioritaires

Résultats obtenus :

Indicateur	Valeur
CA prévu 14 jours	865 022 TND
Produits à risque	596
🔴 CRITIQUE (rupture < 3j)	59
🟠 URGENT (rupture 4-7j)	89
🟡 ATTENTION (rupture 8-14j)	448
Exemple d’alerte générée :

text
🔴 CRITIQUE - Boisson lactée tropicale
   Stock actuel : 0 unité (rupture totale)
   Rupture dans : 1 jour
   Demande quotidienne : 0.76 unité
   Commande recommandée : 1 006 unités
8. Visualisation des résultats (Dashboard)
Dashboard Power BI :

Visualisation	Source
KPI - CA prévu 14 jours	sales_forecast.csv
KPI - Produits CRITIQUE	stock_alerts.csv
KPI - Produits URGENT	stock_alerts.csv
Évolution du CA prévu	sales_forecast.csv
Tableaux des alertes	stock_alerts.csv

📁 Structure du dépôt
projet_prevision_ventes/
│
├── config/
│   └── settings.py              # Configuration PostgreSQL
│
├── data_generator/
│   ├── generator.py             # Génération intelligente des données
│   └── complete_missing.py      # Complétion des transactions
│
├── etl/
│   └── pipeline.py              # Pipeline ETL (pos_analytics)
│
├── stock/
│   ├── 1-stock_features.py      # Feature engineering stock
│   └── 2-demand_forecast_xgboost.py  # Prédiction + alertes
│
├── ml_models/
│   ├── 1-linear_regression.py
│   ├── 2-ridge.py
│   ├── 3-random_forest.py
│   ├── 4-xgboost.py
│   └── 5-lightgbm.py
│
├── dl_models/
│   ├── 0-RNN.py
│   ├── 1-MLP.py
│   ├── 2-LSTM.py
│   └── 3-GRU.py
│
├── results/
│   ├── sales_forecast.csv       # CA prévu 14 jours
│   ├── stock_alerts.csv         # Alertes stock
│   └── stock_status_latest.csv  # État des stocks
│
└── README.md

🚀 Installation et exécution

# 1. Cloner le dépôt
git clone https://github.com/bouchraa7/projet_prevision_ventes.git
cd projet_prevision_ventes

# 2. Installer les dépendances
pip install pandas numpy psycopg2 xgboost lightgbm scikit-learn tensorflow matplotlib

# 3. Configurer la base de données (modifier config/settings.py)

# 4. Générer les données (si nécessaire)
python data_generator/generator.py

# 5. Lancer le pipeline ETL
python etl/pipeline.py

# 6. Feature engineering stock
python stock/1-stock_features.py

# 7. Prédiction + alertes
python stock/2-demand_forecast_xgboost.py

# 8. Importer les CSV dans Power BI

📊 Fichiers générés pour le dashboard
Fichier	Contenu	Utilisation
sales_forecast.csv	CA prévu J+1 à J+14	Graphique évolution
stock_alerts.csv	Produits CRITIQUE, URGENT, ATTENTION	Tableaux alertes
stock_status_latest.csv	État stock 6 875 produits	Analyse globale

🛠️ Technologies utilisées
Catégorie                 	Technologies
Base de données	            PostgreSQL
Machine Learning           	XGBoost, LightGBM, Scikit-learn
Deep Learning             	TensorFlow, Keras (LSTM, GRU, RNN, MLP)
Visualisation	              Power BI, Apache Superset
Langage	                    Python 3.11
Versioning                 	Git, GitHub
