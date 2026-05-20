"""
MLP
"""
import numpy as np
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
import pickle
import os

os.makedirs('results/dl_results', exist_ok=True)

print("="*70)
print("🧠 MLP - Version anti-overfitting")
print("="*70)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================
print("\n📂 Chargement des données...")

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="caissatndb",
    user="postgres",
    password="Bouchra1234"
)

sql = """
SELECT 
    DATE(transaction_date) as date,
    COUNT(DISTINCT transaction_id) as nb_transactions,
    SUM(line_total) as revenue,
    AVG(sold_price) as avg_price,
    EXTRACT(DOW FROM DATE(transaction_date)) as day_of_week,
    EXTRACT(MONTH FROM DATE(transaction_date)) as month,
    MAX(is_ramadan) as is_ramadan
FROM s5831082f95ef4a1eac9a6a8c484faf0a.pos_analytics
WHERE is_refund = false
GROUP BY DATE(transaction_date)
ORDER BY date
"""

df = pd.read_sql(sql, conn)
conn.close()

df['date'] = pd.to_datetime(df['date'])
df['is_weekend'] = (df['day_of_week'] >= 4).astype(int)

print(f"✅ {len(df)} jours chargés")

# ============================================================================
# 2. CRÉATION DES FEATURES (8 features)
# ============================================================================
print("\n📊 Création des features...")

df['sales_per_transaction'] = df['revenue'] / df['nb_transactions']
df['revenue_lag_1'] = df['revenue'].shift(1)
df['revenue_ma_7'] = df['revenue'].rolling(7).mean()
df = df.replace([np.inf, -np.inf], 0).fillna(0)

features = [
    'nb_transactions', 'sales_per_transaction', 'avg_price',
    'day_of_week', 'is_weekend', 'month',
    'revenue_lag_1', 'revenue_ma_7'
]

X = df[features].values
y = df['revenue'].values

# Split
split = int(len(X) * 0.8)       # 851 × 0.8 = 680
X_train, X_test = X[:split], X[split:]   # 680 jours pour APPRENDRE
y_train, y_test = y[:split], y[split:]   # 171 jours pour TESTER

scaler = StandardScaler()    # moyenne=0, écart-type=1
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"\n📊 Split: Train={X_train_s.shape[0]} jours, Test={X_test_s.shape[0]} jours")
print(f"   Features: {len(features)}")

# ============================================================================
# 3. CONFIGURATIONS À TESTER
# ============================================================================
print("\n Test de plusieurs architectures ...")

configurations = [
    {"name": "Très simple", "layers": (16,), "neurones": 16},
    {"name": "Simple", "layers": (32, 16), "neurones": 48},
    {"name": "Medium", "layers": (32, 16, 8), "neurones": 56},
    {"name": "Petit", "layers": (16, 8), "neurones": 24},
]

results = []

for cfg in configurations:
    print(f"\n   Test: {cfg['name']} ({cfg['neurones']} neurones)")
    
    model = MLPRegressor(
        hidden_layer_sizes=cfg['layers'],    # 2 couches : 32 puis 16 neurones
        activation='relu',                   # Fonction d'activation
        solver='adam',                       # Optimiseur
        alpha=0.01,                          # Régularisation plus forte(évite overfitting))
        batch_size=32,
        learning_rate='adaptive',            # Taux adaptatif
        learning_rate_init=0.001,
        max_iter=200,                        # Maximum 200 itérations
        early_stopping=True,                 # Arrêt précoce
        validation_fraction=0.1,
        random_state=42,
        verbose=False
    )
    
    model.fit(X_train_s, y_train)
    
    y_pred_train = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    
    results.append({
        'name': cfg['name'],
        'neurones': cfg['neurones'],
        'layers': cfg['layers'],
        'r2_train': r2_train,
        'r2_test': r2_test,
        'mape': mape_test,
        'overfit': r2_train - r2_test
    })
    
    print(f" R² Train: {r2_train:.4f} | R² Test: {r2_test:.4f} | MAPE: {mape_test:.2f}% | Overfit: {r2_train - r2_test:.4f}")

# ============================================================================
# 4. MEILLEURE CONFIGURATION
# ============================================================================
print("\n" + "="*70)
print(" MEILLEURE CONFIGURATION ")
print("="*70)

# Choisir celle avec le meilleur R² test
best = min(results, key=lambda x: x['overfit'])  # Moins d'overfit

print(f"\n Architecture retenue: {best['name']}")
print(f"   Couches: {best['layers']}")
print(f"   Neurones: {best['neurones']}")
print(f"   R² Train: {best['r2_train']:.4f}")
print(f"   R² Test: {best['r2_test']:.4f}")
print(f"   MAPE: {best['mape']:.2f}%")
print(f"   Surapprentissage: {best['overfit']:.4f}")

# ============================================================================
# 5. MODÈLE FINAL AVEC MEILLEURE CONFIGURATION
# ============================================================================
print("\n Entraînement du modèle final...")

best_model = MLPRegressor(
    hidden_layer_sizes=best['layers'],
    activation='relu',
    solver='adam',
    alpha=0.01,
    batch_size=32,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
    verbose=True
)

best_model.fit(X_train_s, y_train)
y_pred = best_model.predict(X_test_s)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(((y_test - y_pred) ** 2).mean())
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("\n" + "="*70)
print(" RÉSULTATS MLP ")
print("="*70)
print(f"   R²   = {r2:.4f}")
print(f"   MAE  = {mae:.2f} TND")
print(f"   RMSE = {rmse:.2f} TND")
print(f"   MAPE = {mape:.2f}%")
print(f"   Architecture: {best['layers']} ({best['neurones']} neurones)")

# ============================================================================
# 6. COMPARAISON AVEC XGBOOST
# ============================================================================
print("\n" + "="*70)
print("📊 COMPARAISON AVEC XGBOOST")
print("="*70)
print(f"   XGBoost : R² = 0.8483 | MAPE = 3.05%")
print(f"   MLP     : R² = {r2:.4f} | MAPE = {mape:.2f}%")

if r2 > 0.8483 and mape < 3.05:
    print("\n   MLP meilleur que XGBoost!")
else:
    print("\n  XGBoost reste meilleur")

# ============================================================================
# 7. SAUVEGARDE
# ============================================================================
with open('results/dl_results/mlp_antioverfit.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'scaler': scaler,
        'best_layers': best['layers'],
        'neurones': best['neurones'],
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }, f)

print(f"\n💾 Modèle sauvegardé: results/dl_results/mlp_antioverfit.pkl")
print("\n✅ MLP anti-overfitting terminé!")