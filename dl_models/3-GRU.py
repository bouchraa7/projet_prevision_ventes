"""
GRU - RandomizedSearchCV (Version DÉTERMINISTE)
Recherche aléatoire des meilleurs paramètres
Résultats reproductibles à 100%
"""

# ============================================================================
# 1. FORCER LE DÉTERMINISME
# ============================================================================
import os
import random
import numpy as np

os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

random.seed(42)
np.random.seed(42)

# ============================================================================
# 2. IMPORTER TENSORFLOW
# ============================================================================
import tensorflow as tf
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# ============================================================================
# 3. LE RESTE DES IMPORTS
# ============================================================================
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import ParameterSampler
import json

os.makedirs('results/dl_results', exist_ok=True)

print("="*70)
print("🧠 GRU - RandomizedSearchCV (Version DÉTERMINISTE)")
print("="*70)

# ============================================================================
# 4. CHARGEMENT DES DONNÉES
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
    SUM(line_total) as revenue
FROM s5831082f95ef4a1eac9a6a8c484faf0a.pos_analytics
WHERE is_refund = false
GROUP BY DATE(transaction_date)
ORDER BY date
"""

df = pd.read_sql(sql, conn)
conn.close()

print(f"✅ {len(df)} jours chargés")

# ============================================================================
# 5. PRÉPARATION DES DONNÉES
# ============================================================================
scaler = MinMaxScaler()
df['revenue_scaled'] = scaler.fit_transform(df[['revenue']])

SEQ_LEN = 30
X, y = [], []

for i in range(len(df) - SEQ_LEN):
    X.append(df['revenue_scaled'].values[i:i+SEQ_LEN])
    y.append(df['revenue_scaled'].values[i+SEQ_LEN])

X = np.array(X).reshape(-1, SEQ_LEN, 1)
y = np.array(y)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Train: {X_train.shape[0]} séquences")
print(f"Test:  {X_test.shape[0]} séquences")

# ============================================================================
# 6. DISTRIBUTION DES PARAMÈTRES
# ============================================================================
print("\n🔍 RandomizedSearchCV - Test aléatoire des combinaisons...")

param_dist = {
    'units': [32, 64, 128],
    'layers': [1, 2],
    'dropout': [0.1, 0.2, 0.3],
    'batch_size': [16, 32]
}

n_iter = 20
print(f"\n📊 Nombre de combinaisons aléatoires: {n_iter}")

random_params = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))

# ============================================================================
# 7. TEST DES COMBINAISONS
# ============================================================================
results = []
best_r2 = -np.inf
best_config = None
counter = 0

for params in random_params:
    counter += 1
    units = params['units']
    layers = params['layers']
    dropout = params['dropout']
    batch_size = params['batch_size']
    
    print(f"\n[{counter}/{n_iter}] Test: units={units}, layers={layers}, dropout={dropout}, batch={batch_size}")
    
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    model = Sequential()
    for i in range(layers):
        return_seq = (i < layers - 1)
        if i == 0:
            model.add(GRU(units, return_sequences=return_seq, input_shape=(SEQ_LEN, 1)))
        else:
            model.add(GRU(units, return_sequences=return_seq))
        model.add(Dropout(dropout))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0
    )
    
    y_pred = model.predict(X_test, verbose=0)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_real = scaler.inverse_transform(y_pred)
    
    r2 = r2_score(y_test_real, y_pred_real)
    mae = mean_absolute_error(y_test_real, y_pred_real)
    mape = np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100
    
    results.append({
        'units': units,
        'layers': layers,
        'dropout': dropout,
        'batch_size': batch_size,
        'r2': r2,
        'mae': mae,
        'mape': mape
    })
    
    print(f"      R²={r2:.4f}, MAPE={mape:.2f}%")
    
    if r2 > best_r2:
        best_r2 = r2
        best_config = results[-1]
        print(f"      ⭐ NOUVEAU BEST! R²={r2:.4f}")

# ============================================================================
# 8. RÉSULTATS
# ============================================================================
print("\n" + "="*70)
print("🏆 MEILLEURE CONFIGURATION GRU (RandomizedSearch)")
print("="*70)
print(f"   Units: {best_config['units']}")
print(f"   Layers: {best_config['layers']}")
print(f"   Dropout: {best_config['dropout']}")
print(f"   Batch Size: {best_config['batch_size']}")
print(f"   R²: {best_config['r2']:.4f}")
print(f"   MAPE: {best_config['mape']:.2f}%")

# ============================================================================
# 9. TOP 10
# ============================================================================
print("\n" + "="*70)
print("📊 TOP 10 CONFIGURATIONS GRU")
print("="*70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('r2', ascending=False)
print(results_df.head(10).to_string(index=False))

# ============================================================================
# 10. MODÈLE FINAL
# ============================================================================
print("\n🚀 Entraînement du modèle final (150 epochs)...")

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

model = Sequential()
for i in range(best_config['layers']):
    return_seq = (i < best_config['layers'] - 1)
    if i == 0:
        model.add(GRU(best_config['units'], return_sequences=return_seq, input_shape=(SEQ_LEN, 1)))
    else:
        model.add(GRU(best_config['units'], return_sequences=return_seq))
    model.add(Dropout(best_config['dropout']))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=best_config['batch_size'],
    validation_split=0.1,
    shuffle=False,
    callbacks=[early_stop],
    verbose=1
)

y_pred = model.predict(X_test)
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_real = scaler.inverse_transform(y_pred)

r2_final = r2_score(y_test_real, y_pred_real)
mae_final = mean_absolute_error(y_test_real, y_pred_real)
mape_final = np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100

print("\n" + "="*70)
print("📊 RÉSULTATS FINAUX GRU (RandomizedSearch)")
print("="*70)
print(f"   R² = {r2_final:.4f}")
print(f"   MAPE = {mape_final:.2f}%")
print(f"   MAE = {mae_final:.2f} TND")

# Sauvegarde
model.save('results/dl_results/gru_randomized_final.h5')
results_df.to_csv('results/dl_results/gru_randomized_results.csv', index=False)

with open('results/dl_results/gru_randomized_best.json', 'w') as f:
    json.dump(best_config, f, indent=4)

print("\n💾 Fichiers sauvegardés dans results/dl_results/")
print("\n✅ GRU RandomizedSearchCV terminé!")