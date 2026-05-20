"""
RECHERCHE SYSTÉMATIQUE DES PARAMÈTRES
GridSearchCV + TimeSeriesSplit pour Random Forest, XGBoost, LightGBM
"""

import psycopg2
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔬 RECHERCHE SYSTÉMATIQUE DES PARAMÈTRES")
print("GridSearchCV + TimeSeriesSplit (5 folds)")
print("="*80)

# ============================================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================================
print("\n📂 Chargement des données...")

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "caissatndb",
    "user": "postgres",
    "password": "Bouchra1234"
}
SCHEMA = "s5831082f95ef4a1eac9a6a8c484faf0a"

conn = psycopg2.connect(**DB_CONFIG)
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
conn.close()

df['date'] = pd.to_datetime(df['date'])
df['is_weekend'] = (df['day_of_week'] >= 4).astype(int)

print(f"✅ {len(df)} jours chargés")

# ============================================================================
# 2. CRÉATION DES FEATURES (8 features)
# ============================================================================
print("\n⚙️ Création des features...")

df['sales_per_transaction'] = df['revenue'] / df['nb_transactions']
df['revenue_lag_1'] = df['revenue'].shift(1)
df['revenue_ma_7'] = df['revenue'].rolling(7).mean()
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

X = df[features].values
y = df['revenue'].values

# Split temporel (80% train, 20% test)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"✅ {len(features)} features")
print(f"   Train: {len(X_train)} jours")
print(f"   Test:  {len(X_test)} jours")

# Normalisation
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ============================================================================
# 3. TIME SERIES SPLIT (respecte l'ordre chronologique)
# ============================================================================
tscv = TimeSeriesSplit(n_splits=5)
print(f"\n📊 TimeSeriesSplit: 5 folds temporels")

# ============================================================================
# 4. RANDOM FOREST - GridSearchCV
# ============================================================================
print("\n" + "="*80)
print("🌲 RANDOM FOREST - GridSearchCV")
print("="*80)

from sklearn.ensemble import RandomForestRegressor

# Grille de paramètres à tester
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

print("\nParamètres testés:")
for param, values in rf_param_grid.items():
    print(f"   • {param}: {values}")

print(f"\nNombre total de combinaisons: {np.prod([len(v) for v in rf_param_grid.values()])}")

# GridSearch
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=tscv,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

print("\n⏳ Recherche en cours...")
rf_grid.fit(X_train_s, y_train)

print(f"\n✅ MEILLEURS PARAMÈTRES RF:")
for param, value in rf_grid.best_params_.items():
    print(f"   {param}: {value}")
print(f"   Score CV (R²): {rf_grid.best_score_:.4f}")

# Évaluation sur test
rf_best = rf_grid.best_estimator_
y_pred = rf_best.predict(X_test_s)
r2_test = r2_score(y_test, y_pred)
mape_test = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
mae_test = mean_absolute_error(y_test, y_pred)

print(f"\n📊 RÉSULTATS RF SUR TEST:")
print(f"   R²   = {r2_test:.4f}")
print(f"   MAPE = {mape_test:.2f}%")
print(f"   MAE  = {mae_test:.2f} TND")

# ============================================================================
# 5. XGBOOST - GridSearchCV
# ============================================================================
print("\n" + "="*80)
print("⚡ XGBOOST - GridSearchCV")
print("="*80)

import xgboost as xgb

# Grille de paramètres (plus petite pour des raisons de temps)
xgb_param_grid = {
    'n_estimators': [200, 300],
    'learning_rate': [0.03, 0.05, 0.07],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.5, 1, 2]
}

print("\nParamètres testés:")
for param, values in xgb_param_grid.items():
    print(f"   • {param}: {values}")

print(f"\nNombre total de combinaisons: {np.prod([len(v) for v in xgb_param_grid.values()])}")

# GridSearch
xgb_grid = GridSearchCV(
    xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),
    xgb_param_grid,
    cv=tscv,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

print("\n⏳ Recherche en cours...")
xgb_grid.fit(X_train_s, y_train)

print(f"\n✅ MEILLEURS PARAMÈTRES XGB:")
for param, value in xgb_grid.best_params_.items():
    print(f"   {param}: {value}")
print(f"   Score CV (R²): {xgb_grid.best_score_:.4f}")

# Évaluation sur test
xgb_best = xgb_grid.best_estimator_
y_pred = xgb_best.predict(X_test_s)
r2_test = r2_score(y_test, y_pred)
mape_test = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
mae_test = mean_absolute_error(y_test, y_pred)

print(f"\n📊 RÉSULTATS XGB SUR TEST:")
print(f"   R²   = {r2_test:.4f}")
print(f"   MAPE = {mape_test:.2f}%")
print(f"   MAE  = {mae_test:.2f} TND")

# ============================================================================
# 6. LIGHTGBM - GridSearchCV
# ============================================================================
print("\n" + "="*80)
print("⚡ LIGHTGBM - GridSearchCV")
print("="*80)

import lightgbm as lgb

# Grille de paramètres
lgb_param_grid = {
    'n_estimators': [200, 300],
    'learning_rate': [0.03, 0.05, 0.07],
    'num_leaves': [15, 31, 63],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0.5, 1, 2]
}

print("\nParamètres testés:")
for param, values in lgb_param_grid.items():
    print(f"   • {param}: {values}")

print(f"\nNombre total de combinaisons: {np.prod([len(v) for v in lgb_param_grid.values()])}")

# GridSearch
lgb_grid = GridSearchCV(
    lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1),
    lgb_param_grid,
    cv=tscv,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

print("\n⏳ Recherche en cours...")
lgb_grid.fit(X_train_s, y_train)

print(f"\n✅ MEILLEURS PARAMÈTRES LGB:")
for param, value in lgb_grid.best_params_.items():
    print(f"   {param}: {value}")
print(f"   Score CV (R²): {lgb_grid.best_score_:.4f}")

# Évaluation sur test
lgb_best = lgb_grid.best_estimator_
y_pred = lgb_best.predict(X_test_s)
r2_test = r2_score(y_test, y_pred)
mape_test = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
mae_test = mean_absolute_error(y_test, y_pred)

print(f"\n📊 RÉSULTATS LGB SUR TEST:")
print(f"   R²   = {r2_test:.4f}")
print(f"   MAPE = {mape_test:.2f}%")
print(f"   MAE  = {mae_test:.2f} TND")

# ============================================================================
# 7. TABLEAU COMPARATIF FINAL
# ============================================================================
print("\n" + "="*80)
print("📊 TABLEAU COMPARATIF FINAL")
print("="*80)

results = [
    {
        'Modèle': 'Random Forest',
        'R²': rf_grid.best_score_,
        'R² Test': r2_score(y_test, rf_best.predict(X_test_s)),
        'MAPE Test': np.mean(np.abs((y_test - rf_best.predict(X_test_s)) / y_test)) * 100,
        'MAE Test': mean_absolute_error(y_test, rf_best.predict(X_test_s))
    },
    {
        'Modèle': 'XGBoost',
        'R²': xgb_grid.best_score_,
        'R² Test': r2_score(y_test, xgb_best.predict(X_test_s)),
        'MAPE Test': np.mean(np.abs((y_test - xgb_best.predict(X_test_s)) / y_test)) * 100,
        'MAE Test': mean_absolute_error(y_test, xgb_best.predict(X_test_s))
    },
    {
        'Modèle': 'LightGBM',
        'R²': lgb_grid.best_score_,
        'R² Test': r2_score(y_test, lgb_best.predict(X_test_s)),
        'MAPE Test': np.mean(np.abs((y_test - lgb_best.predict(X_test_s)) / y_test)) * 100,
        'MAE Test': mean_absolute_error(y_test, lgb_best.predict(X_test_s))
    }
]

print("\n{:15} {:10} {:10} {:10} {:12}".format('Modèle', 'R² CV', 'R² Test', 'MAPE(%)', 'MAE(TND)'))
print("-"*70)
for r in results:
    print(f"{r['Modèle']:15} {r['R²']:.4f}     {r['R² Test']:.4f}     {r['MAPE Test']:.2f}       {r['MAE Test']:.0f}")

# ============================================================================
# 8. MEILLEUR MODÈLE
# ============================================================================
print("\n" + "="*80)
print("🏆 MEILLEUR MODÈLE")
print("="*80)

# Trouver le meilleur modèle selon MAPE (priorité à la précision)
best = min(results, key=lambda x: x['MAPE Test'])
print(f"\n✅ MODÈLE RECOMMANDÉ: {best['Modèle']}")
print(f"   R²   = {best['R² Test']:.4f}")
print(f"   MAPE = {best['MAPE Test']:.2f}%")
print(f"   MAE  = {best['MAE Test']:.0f} TND")

# ============================================================================
# 9. SAUVEGARDE
# ============================================================================
print("\n💾 Sauvegarde des résultats...")

# Sauvegarder les meilleurs paramètres
import pickle
with open('results/best_params.pkl', 'wb') as f:
    pickle.dump({
        'rf': rf_grid.best_params_,
        'xgb': xgb_grid.best_params_,
        'lgb': lgb_grid.best_params_,
        'best_model': best['Modèle'],
        'rf_score': rf_grid.best_score_,
        'xgb_score': xgb_grid.best_score_,
        'lgb_score': lgb_grid.best_score_
    }, f)

print("   ✅ results/best_params.pkl")

print("\n" + "="*80)
print("✅ RECHERCHE TERMINÉE")
print("="*80)