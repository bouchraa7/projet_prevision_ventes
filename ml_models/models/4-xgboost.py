"""
XGBOOST - RandomizedSearchCV (8 features)
Recherche automatique des meilleurs paramètres
"""

import psycopg2
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import xgboost as xgb
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)

print("="*60)
print("⚡ XGBOOST - RandomizedSearchCV (8 features)")
print("="*60)

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

# ============================================================================
# CHARGEMENT
# ============================================================================
print("\n📂 Chargement des données...")
conn = psycopg2.connect(**DB_CONFIG)

sql = f"""
SELECT 
    DATE(transaction_date) AS date,
    COUNT(DISTINCT transaction_id) AS nb_transactions,
    SUM(line_total) AS revenue,
    AVG(sold_price) AS avg_price,
    EXTRACT(DOW FROM DATE(transaction_date)) AS day_of_week,
    EXTRACT(MONTH FROM DATE(transaction_date)) AS month
FROM {SCHEMA}.pos_analytics
WHERE is_refund = false
GROUP BY DATE(transaction_date)
ORDER BY date
"""

df = pd.read_sql(sql, conn)
conn.close()

print(f"✅ {len(df)} jours chargés")

# ============================================================================
# FEATURES
# ============================================================================
print("\n⚙️ Création des features...")

df["date"] = pd.to_datetime(df["date"])
df["is_weekend"] = (df["day_of_week"] >= 4).astype(int)
df["sales_per_transaction"] = df["revenue"] / df["nb_transactions"]
df["revenue_lag_1"] = df["revenue"].shift(1)
df["revenue_ma_7"] = df["revenue"].rolling(window=7).mean()

df = df.replace([np.inf, -np.inf], 0).fillna(0)

features = [
    'nb_transactions',
    'sales_per_transaction',
    'avg_price',
    'day_of_week',
    'is_weekend',
    'month',
    'revenue_lag_1',
    'revenue_ma_7'
]

print(f"\n✅ {len(features)} features")
for i, f in enumerate(features, 1):
    print(f"   {i}. {f}")

# ============================================================================
# SPLIT
# ============================================================================
X = df[features].values
y = df["revenue"].values

split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"\n📊 Split temporel:")
print(f"   Train : {len(X_train)} jours")
print(f"   Test  : {len(X_test)} jours")

# ============================================================================
# NORMALISATION
# ============================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# PARAMÈTRES FIXES
# ============================================================================
fixed_params = {
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

# ============================================================================
# RANDOMIZED SEARCH
# ============================================================================
print("\n🔍 Recherche automatique des paramètres...")

param_dist = {
    'n_estimators': [100, 150, 200, 250, 300],
    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
    'max_depth': [3, 4, 5, 6, 7],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [0.5, 1, 1.5, 2]
}

print("\n   Paramètres testés:")
for param, values in param_dist.items():
    print(f"   • {param}: {len(values)} valeurs")

tscv = TimeSeriesSplit(n_splits=3)

xgb_base = xgb.XGBRegressor(**fixed_params)

random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=60,
    cv=tscv,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("\n⏳ Recherche en cours (environ 5-8 minutes)...")
random_search.fit(X_train_scaled, y_train)

# ============================================================================
# MEILLEURS PARAMÈTRES
# ============================================================================
best_params = random_search.best_params_
best_score = random_search.best_score_

print("\n" + "="*60)
print("🏆 MEILLEURS PARAMÈTRES")
print("="*60)
for param, value in best_params.items():
    print(f"   {param}: {value}")
print(f"\n   Meilleur score CV: {best_score:.4f}")

# ============================================================================
# MODÈLE FINAL
# ============================================================================
print("\n🚀 Entraînement...")

all_params = {**fixed_params, **best_params}
model = xgb.XGBRegressor(**all_params)
model.fit(X_train_scaled, y_train)

# ============================================================================
# PRÉDICTIONS
# ============================================================================
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# ============================================================================
# ÉVALUATION
# ============================================================================
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

print("\n" + "="*60)
print("📊 RÉSULTATS XGBOOST")
print("="*60)
print(f"   R² Entraînement : {r2_train:.4f}")
print(f"   R² Test         : {r2_test:.4f}")
print(f"   MAE Test        : {mae_test:.2f} TND")
print(f"   RMSE Test       : {rmse_test:.2f} TND")
print(f"   MAPE Test       : {mape_test:.2f}%")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n📊 IMPORTANCE DES FEATURES")
print("-"*50)

importance = dict(zip(features, model.feature_importances_))
total = sum(importance.values())

for f, v in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    bar = "█" * int(v / max(importance.values()) * 50)
    pct = v / total * 100
    print(f"   {f:<22}: {v:.4f} {bar} ({pct:.1f}%)")

# ============================================================================
# SAUVEGARDE
# ============================================================================
with open('results/xgboost_8f_auto.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'scaler': scaler,
        'features': features,
        'best_params': best_params,
        'best_cv_score': best_score,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'mae_test': mae_test,
        'rmse_test': rmse_test,
        'mape_test': mape_test,
        'feature_importance': importance
    }, f)

print("\n💾 Modèle sauvegardé: results/xgboost_8f_auto.pkl")
print("\n✅ XGBoost terminé !")