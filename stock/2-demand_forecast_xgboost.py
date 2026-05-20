"""
STOCK PREDICTION — Utilisation du modèle XGBoost 
=================================================================
Transforme les prédictions de CA → Quantités → Stock
"""
import psycopg2
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "caissatndb",
    "user": "postgres",
    "password": "Bouchra1234"
}
SCHEMA = "s5831082f95ef4a1eac9a6a8c484faf0a"

PREDICTION_DAYS = 14

print("=" * 70)
print("STOCK PREDICTION — Avec XGBoost ")
print("=" * 70)

# ============================================================================
# 1. CHARGER LE MODÈLE EXISTANT 
# ============================================================================
print("\n Chargement du modèle XGBoost existant...")

with open('results/xgboost_8f_auto.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']

print(f" Modèle chargé : R² = {model_data['r2_test']:.4f}, MAPE = {model_data['mape_test']:.2f}%")

# ============================================================================
# 2. CHARGER LES DONNÉES
# ============================================================================
print("\n Chargement des données...")

conn = psycopg2.connect(**DB_CONFIG)

# Données journalières agrégées (comme dans l'entraînement original)
sql = f"""
SELECT 
    DATE(transaction_date) as date,
    COUNT(DISTINCT transaction_id) as nb_transactions,
    SUM(line_total) as revenue,
    AVG(sold_price) as avg_price,
    EXTRACT(DOW FROM DATE(transaction_date)) as day_of_week,
    EXTRACT(MONTH FROM DATE(transaction_date)) as month,
    MAX(is_ramadan) as is_ramadan
FROM {SCHEMA}.pos_analytics
WHERE is_refund = false
GROUP BY DATE(transaction_date)
ORDER BY date
"""

df = pd.read_sql(sql, conn)
df['date'] = pd.to_datetime(df['date'])

print(f" {len(df)} jours chargés")

# ============================================================================
# 3. PRÉDIRE LE CA FUTUR
# ============================================================================
print(f"\n Prédiction du CA pour les {PREDICTION_DAYS} prochains jours...")

# Préparer les features
df['is_weekend'] = (df['day_of_week'] >= 4).astype(int)
df['sales_per_transaction'] = df['revenue'] / df['nb_transactions']
df['revenue_lag_1'] = df['revenue'].shift(1)
df['revenue_ma_7'] = df['revenue'].rolling(7).mean()
df = df.fillna(0).replace([np.inf, -np.inf], 0)

# Dernières données
last_row = df.iloc[-1].copy()
last_date = df['date'].iloc[-1]

predictions = []
current_df = df.copy()

for day in range(1, PREDICTION_DAYS + 1):
    pred_date = last_date + timedelta(days=day)
    
    # Créer la ligne de prédiction
    new_row = pd.DataFrame({
        'date': [pred_date],
        'nb_transactions': [last_row['nb_transactions']],
        'revenue': [0],
        'avg_price': [last_row['avg_price']],
        'day_of_week': [pred_date.weekday()],
        'month': [pred_date.month],
        'is_ramadan': [1 if (pred_date.month == 3 and pred_date.day > 10) else 0],
        'is_weekend': [1 if pred_date.weekday() >= 4 else 0],
        'sales_per_transaction': [last_row['sales_per_transaction']],
        'revenue_lag_1': [current_df['revenue'].iloc[-1]],
        'revenue_ma_7': [current_df['revenue'].iloc[-7:].mean()]
    })
    
    current_df = pd.concat([current_df, new_row], ignore_index=True)
    
    # Prédire
    X_pred = current_df[features].iloc[-1:].values
    X_pred_scaled = scaler.transform(X_pred)
    pred_revenue = model.predict(X_pred_scaled)[0]
    
    predictions.append({
        'date': pred_date,
        'predicted_revenue': round(pred_revenue, 2),
        'day_of_week': pred_date.weekday(),
        'is_weekend': 1 if pred_date.weekday() >= 4 else 0
    })
    
    # Mettre à jour pour le prochain jour
    current_df.loc[current_df.index[-1], 'revenue'] = pred_revenue
    if len(current_df) > 7:
        current_df.loc[current_df.index[-1], 'revenue_ma_7'] = current_df['revenue'].iloc[-7:].mean()

predictions_df = pd.DataFrame(predictions)

print(f"\n PRÉVISIONS CA TOTAUX :")
print(predictions_df[['date', 'predicted_revenue']].to_string(index=False))
print(f"\n   Total CA prévu : {predictions_df['predicted_revenue'].sum():,.2f} TND")
print(f"   Moyenne/jour : {predictions_df['predicted_revenue'].mean():,.2f} TND")

# ============================================================================
# 4. CHARGER LES STOCKS ET PRODUITS
# ============================================================================
print("\n  Chargement des stocks par produit...")

stock_sql = f"""
SELECT 
    id as product_id,
    name as product_name,
    COALESCE(stock_qty, 0) as current_stock,
    COALESCE(stock_min_limit, 10) as min_stock,
    COALESCE(stock_max_limit, 500) as max_stock,
    sale_unit_price,
    COALESCE(purchase_unit_price, sale_unit_price * 0.6) as cost_price
FROM {SCHEMA}.products
WHERE sale_unit_price > 0
"""

stocks = pd.read_sql(stock_sql, conn)
conn.close()

print(f" {len(stocks)} produits chargés")

# ============================================================================
# 5. CALCULER LE POIDS DE CHAQUE PRODUIT DANS LES VENTES
# ============================================================================
print("\n Calcul de la répartition des ventes par produit...")

# Récupérer l'historique des ventes par produit
conn = psycopg2.connect(**DB_CONFIG)
sales_by_product_sql = f"""
SELECT 
    product_id,
    SUM(qty) as total_qty_sold,
    SUM(line_total) as total_revenue
FROM {SCHEMA}.pos_analytics
WHERE is_refund = false
GROUP BY product_id
"""
sales_by_product = pd.read_sql(sales_by_product_sql, conn)
conn.close()

# Calculer le poids de chaque produit
total_revenue_all = sales_by_product['total_revenue'].sum()
sales_by_product['revenue_share'] = sales_by_product['total_revenue'] / total_revenue_all

# Fusionner avec stocks
stocks = stocks.merge(sales_by_product[['product_id', 'revenue_share', 'total_qty_sold']], 
                       on='product_id', how='left')
stocks['revenue_share'] = stocks['revenue_share'].fillna(0)
stocks['total_qty_sold'] = stocks['total_qty_sold'].fillna(0)

# ============================================================================
# 6. PRÉDIRE LA QUANTITÉ VENDUE PAR PRODUIT
# ============================================================================
print("\n Conversion CA → Quantités par produit...")

# Prix moyen pondéré
avg_price = (stocks['sale_unit_price'] * stocks['revenue_share']).sum()

# Pour chaque jour, répartir le CA total entre les produits
daily_results = []

for _, pred in predictions_df.iterrows():
    date = pred['date']
    total_revenue = pred['predicted_revenue']
    
    for _, product in stocks.iterrows():
        if product['revenue_share'] > 0 and product['sale_unit_price'] > 0:
            # Quantité prédite = (part du CA) / prix unitaire
            predicted_qty = (total_revenue * product['revenue_share']) / product['sale_unit_price']
        else:
            predicted_qty = 0
        
        daily_results.append({
            'date': date,
            'product_id': product['product_id'],
            'product_name': product['product_name'],
            'predicted_revenue_share': total_revenue * product['revenue_share'],
            'predicted_qty': round(predicted_qty, 2),
            'current_stock': product['current_stock'],
            'min_stock': product['min_stock'],
            'max_stock': product['max_stock'],
            'sale_price': product['sale_unit_price']
        })

results_df = pd.DataFrame(daily_results)

# ============================================================================
# 7. CALCULER LE STOCK PROJETÉ ET LES ALERTES
# ============================================================================
print("\n  Calcul du stock projeté et des alertes...")

# Grouper par produit et calculer le stock cumulé
alert_list = []

for product_id in results_df['product_id'].unique():
    prod_df = results_df[results_df['product_id'] == product_id].copy()
    prod_df = prod_df.sort_values('date')
    
    current_stock = prod_df['current_stock'].iloc[0]
    min_stock = prod_df['min_stock'].iloc[0]
    max_stock = prod_df['max_stock'].iloc[0]
    product_name = prod_df['product_name'].iloc[0]
    
    # Calcul du stock projeté
    stock_projected = current_stock
    for idx, row in prod_df.iterrows():
        stock_projected -= row['predicted_qty']
        
        # Détection de rupture
        if stock_projected <= min_stock:
            alert_list.append({
                'product_id': product_id,
                'product_name': product_name,
                'current_stock': current_stock,
                'min_stock': min_stock,
                'max_stock': max_stock,
                'rupture_day': row['date'],
                'days_to_rupture': (row['date'] - prod_df['date'].min()).days + 1,
                'predicted_daily_demand': row['predicted_qty'],
                'recommended_order': max(0, int(max_stock - stock_projected + row['predicted_qty'] * 7))
            })
            break

alerts_df = pd.DataFrame(alert_list)

# ============================================================================
# 8. CLASSIFICATION DES ALERTES
# ============================================================================
def get_priority(days):
    if days <= 3:
        return '🔴 CRITIQUE'
    elif days <= 7:
        return '🟠 URGENT'
    elif days <= 14:
        return '🟡 ATTENTION'
    else:
        return '🟢 OK'

if len(alerts_df) > 0:
    alerts_df['priority'] = alerts_df['days_to_rupture'].apply(get_priority)
    alerts_df = alerts_df.sort_values('days_to_rupture')
else:
    alerts_df = pd.DataFrame(columns=['product_id', 'product_name', 'current_stock', 'priority'])

# ============================================================================
# 9. AFFICHAGE DES RÉSULTATS
# ============================================================================
print("\n" + "=" * 70)
print(" RÉSULTATS FINAUX")
print("=" * 70)

print(f"\n PRÉVISIONS (CA) :")
print(f"   R² modèle original : 0.8483")
print(f"   Total CA prévu 14j : {predictions_df['predicted_revenue'].sum():,.2f} TND")
print(f"   Moyenne journalière : {predictions_df['predicted_revenue'].mean():,.2f} TND")

print(f"\n ALERTES STOCK :")
print(f"   Produits analysés : {len(stocks)}")
print(f"   Produits à risque : {len(alerts_df)}")

if len(alerts_df) > 0:
    print(f"\n 🔴 CRITIQUE (0-3j) : {len(alerts_df[alerts_df['priority']=='🔴 CRITIQUE'])}")
    print(f"   🟠 URGENT (4-7j)   : {len(alerts_df[alerts_df['priority']=='🟠 URGENT'])}")
    print(f"   🟡 ATTENTION (8-14j): {len(alerts_df[alerts_df['priority']=='🟡 ATTENTION'])}")

# ============================================================================
# 10. TOP PRODUITS À COMMANDER
# ============================================================================
if len(alerts_df) > 0:
    print("\n" + "=" * 70)
    print("🛒 TOP PRODUITS À RÉAPPROVISIONNER")
    print("=" * 70)
    
    for _, row in alerts_df.head(15).iterrows():
        print(f"\n{row['priority']} {row['product_name'][:45]}")
        print(f"   📦 Stock actuel : {row['current_stock']:.0f} unités")
        print(f"   📅 Rupture dans : {row['days_to_rupture']} jours")
        print(f"   📊 Demande quotidienne : {row['predicted_daily_demand']:.2f}")
        print(f"   🛒 Commande recommandée : {row['recommended_order']:.0f} unités")

# ============================================================================
# 11. EXPORT POUR SUPERSET
# ============================================================================
print("\n  Export des fichiers pour Superset...")

# Prévisions CA
predictions_df.to_csv('results/sales_forecast.csv', index=False)
print(" sales_forecast.csv")

# Alertes stock
alerts_df.to_csv('results/stock_alerts.csv', index=False)
print(" stock_alerts.csv")

# Détail des prédictions par produit
results_df.to_csv('results/stock_predictions_detail.csv', index=False)
print(" stock_predictions_detail.csv")

# ============================================================================
# 12. RÉSUMÉ
# ============================================================================
print("\n" + "=" * 70)
print(" TERMINÉ — Dashboard prêt ")
print("=" * 70)
print("""
📋 FICHIERS À IMPORTER DANS SUPERSET :
   1. sales_forecast.csv → Graphique des prévisions CA
   2. stock_alerts.csv → Tableau des alertes stock
   3. stock_predictions_detail.csv → Détail par produit

📊 DASHBOARDS À CRÉER :
   - KPI : CA prévu J+7, J+14
   - Graphique : Évolution du CA réel vs prédit
   - Tableau : Produits 🔴 CRITIQUE (commande immédiate)
   - Tableau : Produits 🟠 URGENT (commande cette semaine)
""")
print("=" * 70)