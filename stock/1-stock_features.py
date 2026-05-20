"""
STOCK PREDICTION - Feature Engineering
=======================================================
Prépare les données par produit pour la prédiction de stock :
  - Vélocité de vente (daily, 7j, 30j)
  - Jours avant rupture
  - Score de risque de rupture
  - Features temporelles enrichies
  - Sauvegarde dans stock_analytics (PostgreSQL)
"""

import psycopg2
import pandas as pd
import numpy as np
from psycopg2.extras import execute_batch
import os
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

os.makedirs('results/stock', exist_ok=True)
os.makedirs('figures/stock', exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "caissatndb",
    "user": "postgres",
    "password": "Bouchra1234"
}
SCHEMA = "s5831082f95ef4a1eac9a6a8c484faf0a"

LEAD_TIME_DAYS    = 3     # Délai moyen de réapprovisionnement (jours)
SAFETY_STOCK_MULT = 1.5   # Multiplicateur stock de sécurité
RUPTURE_THRESHOLD = 7     # Jours → alerte critique
FORECAST_HORIZON  = 30    # Jours à prédire

# ============================================================================
# EXTRACTION
# ============================================================================
print("=" * 65)
print("📦 STOCK PREDICTION — Feature Engineering")
print("=" * 65)

log.info("Connexion à la base de données...")
conn = psycopg2.connect(**DB_CONFIG)

# Ventes journalières par produit
sql_sales = f"""
SELECT
    DATE(transaction_date)          AS date,
    product_id,
    product_name,
    brand_name,
    SUM(qty)                        AS qty_sold,
    SUM(line_total)                 AS revenue,
    AVG(sold_price)                 AS avg_price,
    MAX(current_stock)              AS stock_snapshot,
    MAX(stock_min_limit)            AS stock_min,
    MAX(stock_max_limit)            AS stock_max,
    MAX(cost_price)                 AS cost_price,
    MAX(is_ramadan)                 AS is_ramadan,
    MAX(is_eid_al_fitr)             AS is_eid,
    MAX(is_rentree_scolaire)        AS is_rentree,
    EXTRACT(DOW  FROM DATE(transaction_date)) AS day_of_week,
    EXTRACT(MONTH FROM DATE(transaction_date)) AS month,
    EXTRACT(WEEK  FROM DATE(transaction_date)) AS week_of_year
FROM {SCHEMA}.pos_analytics
WHERE is_refund = false
GROUP BY DATE(transaction_date), product_id, product_name, brand_name
ORDER BY product_id, date
"""

log.info("Chargement des ventes journalières par produit...")
df = pd.read_sql(sql_sales, conn)
df['date'] = pd.to_datetime(df['date'])

# Catalogue produits
sql_products = f"""
SELECT id, name, sale_unit_price, purchase_unit_price,
       COALESCE(stock_qty, 0)       AS current_stock,
       COALESCE(stock_min_limit, 0) AS stock_min,
       COALESCE(stock_max_limit, 0) AS stock_max
FROM {SCHEMA}.products
WHERE sale_unit_price > 0
"""
products = pd.read_sql(sql_products, conn)
conn.close()

log.info(f"Données chargées : {len(df):,} lignes | {df['product_id'].nunique()} produits")

# ============================================================================
# FEATURE ENGINEERING PAR PRODUIT
# ============================================================================
log.info("Calcul des features de stock par produit...")

all_features = []
product_ids = df['product_id'].unique()

for pid in product_ids:
    sub = df[df['product_id'] == pid].copy().sort_values('date')

    if len(sub) < 7:
        continue  # Pas assez d'historique

    # Compléter les jours manquants (0 vente = jour sans activité)
    full_range = pd.date_range(sub['date'].min(), sub['date'].max(), freq='D')
    sub = sub.set_index('date').reindex(full_range).reset_index()
    sub.rename(columns={'index': 'date'}, inplace=True)

    # Remplir les métadonnées produit (CORRECTION ICI)
    for col in ['product_id', 'product_name', 'brand_name', 'stock_min', 'stock_max',
                'avg_price', 'cost_price']:
        if col in sub.columns:
            sub[col] = sub[col].ffill().bfill()  # ← CORRECTION
  
    sub['qty_sold'] = sub['qty_sold'].fillna(0)
    sub['revenue']  = sub['revenue'].fillna(0)

    # ── Vélocités ──────────────────────────────────────────────────────────
    sub['velocity_1d']  = sub['qty_sold']
    sub['velocity_7d']  = sub['qty_sold'].rolling(7,  min_periods=1).mean()
    sub['velocity_14d'] = sub['qty_sold'].rolling(14, min_periods=1).mean()
    sub['velocity_30d'] = sub['qty_sold'].rolling(30, min_periods=1).mean()
    sub['velocity_std_7d'] = sub['qty_sold'].rolling(7, min_periods=2).std().fillna(0)

    # ── Lags (demande passée) ───────────────────────────────────────────────
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        sub[f'qty_lag_{lag}'] = sub['qty_sold'].shift(lag)
        sub[f'rev_lag_{lag}'] = sub['revenue'].shift(lag)

    # ── Rolling revenue ────────────────────────────────────────────────────
    for w in [7, 14, 30]:
        sub[f'rev_roll_mean_{w}'] = sub['revenue'].rolling(w, min_periods=1).mean()
        sub[f'qty_roll_mean_{w}'] = sub['qty_sold'].rolling(w, min_periods=1).mean()

    # ── Features temporelles ───────────────────────────────────────────────
    sub['day_of_week']  = sub['date'].dt.dayofweek
    sub['month']        = sub['date'].dt.month
    sub['week_of_year'] = sub['date'].dt.isocalendar().week.astype(int)
    sub['is_weekend']   = (sub['date'].dt.dayofweek >= 4).astype(int)
    sub['quarter']      = sub['date'].dt.quarter

    # ── Tendance ───────────────────────────────────────────────────────────
    sub['qty_pct_7d']  = sub['qty_sold'].pct_change(7).replace([np.inf, -np.inf], 0).fillna(0)
    sub['qty_pct_30d'] = sub['qty_sold'].pct_change(30).replace([np.inf, -np.inf], 0).fillna(0)

    # ── Stock simulé (décroissant) ─────────────────────────────────────────
    # Récupérer le stock actuel du produit
    prod_row = products[products['id'] == pid]
    current_stock = int(prod_row['current_stock'].values[0]) if len(prod_row) > 0 else 0
    stock_min     = int(prod_row['stock_min'].values[0])     if len(prod_row) > 0 else 5
    stock_max     = int(prod_row['stock_max'].values[0])     if len(prod_row) > 0 else 100

    sub['current_stock'] = current_stock
    sub['stock_min']     = stock_min
    sub['stock_max']     = stock_max

    # Jours avant rupture (basé sur vélocité 30j)
    v30 = max(sub['velocity_30d'].iloc[-1], 0.01)
    sub['days_to_stockout'] = np.clip(current_stock / v30, 0, 365)

    # Stock de sécurité recommandé
    v_std = sub['velocity_std_7d'].iloc[-1]
    sub['safety_stock'] = np.ceil(
        v30 * LEAD_TIME_DAYS + SAFETY_STOCK_MULT * v_std * np.sqrt(LEAD_TIME_DAYS)
    )

    # Point de commande (reorder point)
    sub['reorder_point'] = np.ceil(v30 * LEAD_TIME_DAYS + sub['safety_stock'])

    # Score de risque de rupture [0-1]
    sub['rupture_score'] = np.clip(
        1 - sub['days_to_stockout'] / 30, 0, 1
    )

    # Statut stock
    def classify_stock(row):
        if row['current_stock'] <= row['stock_min'] or row['days_to_stockout'] < 3:
            return 'CRITIQUE'
        elif row['days_to_stockout'] < RUPTURE_THRESHOLD:
            return 'ALERTE'
        elif row['days_to_stockout'] < 14:
            return 'FAIBLE'
        elif row['current_stock'] >= row['stock_max'] * 0.9:
            return 'SURSTOCK'
        else:
            return 'NORMAL'

    sub['stock_status'] = sub.apply(classify_stock, axis=1)

    # ── Quantité de réapprovisionnement recommandée ────────────────────────
    sub['reorder_qty'] = np.maximum(
        0,
        np.ceil(stock_max - current_stock + v30 * FORECAST_HORIZON)
    )

    all_features.append(sub)

# Consolidation
df_features = pd.concat(all_features, ignore_index=True)
df_features = df_features.replace([np.inf, -np.inf], 0).fillna(0)

log.info(f"Features calculées : {len(df_features):,} lignes | {df_features['product_id'].nunique()} produits")

# ============================================================================
# RÉSUMÉ ÉTAT DU STOCK ACTUEL
# ============================================================================
print("\n" + "=" * 65)
print("📊 ÉTAT ACTUEL DU STOCK (dernière observation)")
print("=" * 65)

# Dernière observation par produit
latest = df_features.sort_values('date').groupby('product_id').last().reset_index()

status_counts = latest['stock_status'].value_counts()
print("\n   Distribution des statuts :")
for status, count in status_counts.items():
    pct = count / len(latest) * 100
    icon = {'CRITIQUE': '🔴', 'ALERTE': '🟠', 'FAIBLE': '🟡',
            'NORMAL': '🟢', 'SURSTOCK': '🔵'}.get(status, '⚪')
    print(f"   {icon} {status:<12}: {count:3d} produits ({pct:.1f}%)")

print(f"\n   Jours moyens avant rupture : {latest['days_to_stockout'].mean():.1f}")
print(f"   Score rupture moyen        : {latest['rupture_score'].mean():.3f}")

# Top 10 produits à risque
print("\n" + "=" * 65)
print("⚠️  TOP 10 PRODUITS À RISQUE DE RUPTURE")
print("=" * 65)
at_risk = latest.sort_values('rupture_score', ascending=False).head(10)
for _, row in at_risk.iterrows():
    icon = '🔴' if row['rupture_score'] > 0.8 else '🟠'
    print(f"   {icon} {str(row['product_name'])[:35]:<35} | "
          f"Stock: {int(row['current_stock']):4d} | "
          f"J-rupture: {row['days_to_stockout']:5.1f} | "
          f"Score: {row['rupture_score']:.3f}")

# ============================================================================
# SAUVEGARDE
# ============================================================================
# Sauvegarde en CSV (pas besoin de pyarrow)
df_features.to_csv('results/stock/stock_features.csv', index=False)
latest.to_csv('results/stock/stock_status_latest.csv', index=False)

print(f"\n💾 Features sauvegardées : results/stock/stock_features.csv")
print(f"💾 Statut actuel        : results/stock/stock_status_latest.csv")
print(f"\n✅ Feature Engineering terminé !")
print(f"   → {len(df_features):,} lignes | {df_features['product_id'].nunique()} produits")