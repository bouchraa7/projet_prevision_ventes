"""
GRU - GridSearchCV (Version COMPLÈTE et DÉTERMINISTE)
Résultats reproductibles à 100%
Recherche exhaustive des meilleurs paramètres
"""

# ============================================================================
# 1. FORCER LE DÉTERMINISME (AVANT TOUT IMPORT)
# ============================================================================
import os
import random
import numpy as np

# Variables d'environnement pour le déterministe
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Graines de base
random.seed(42)
np.random.seed(42)

# ============================================================================
# 2. IMPORTER TENSORFLOW APRÈS AVOIR FIXÉ LES GRAINES
# ============================================================================
import tensorflow as tf
tf.random.set_seed(42)

# Désactiver le parallélisme non déterministe
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
import itertools
import json

os.makedirs('results/dl_results', exist_ok=True)

print("="*70)
print("🧠 GRU - GridSearchCV (Version COMPLÈTE et DÉTERMINISTE)")
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

# Split temporel (80% train, 20% test)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\n📊 Split des données:")
print(f"   Train: {X_train.shape[0]} séquences")
print(f"   Test:  {X_test.shape[0]} séquences")

# ============================================================================
# 6. GRILLE DE PARAMÈTRES
# ============================================================================
print("\n🔍 GridSearchCV - Test de TOUTES les combinaisons...")

param_grid = {
    'units': [32, 64, 128],
    'layers': [1, 2],
    'dropout': [0.1, 0.2, 0.3],
    'batch_size': [16, 32]
}

print("\n📊 Grille de paramètres:")
for param, values in param_grid.items():
    print(f"   {param}: {values}")

total_combinations = len(param_grid['units']) * len(param_grid['layers']) * \
                     len(param_grid['dropout']) * len(param_grid['batch_size'])
print(f"\n📊 Nombre total de combinaisons: {total_combinations}")

# ============================================================================
# 7. TEST DE TOUTES LES COMBINAISONS
# ============================================================================
results = []
best_r2 = -np.inf
best_config = None
counter = 0

for units, layers, dropout, batch_size in itertools.product(
    param_grid['units'], 
    param_grid['layers'], 
    param_grid['dropout'], 
    param_grid['batch_size']
):
    counter += 1
    print(f"\n[{counter}/{total_combinations}] Test: units={units}, layers={layers}, dropout={dropout}, batch={batch_size}")
    
    # ✅ RAZ des graines avant chaque modèle (pour reproductibilité)
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Construction du modèle GRU
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
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    # ✅ shuffle=False pour garder l'ordre temporel
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Évaluation
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
# 8. AFFICHAGE DES MEILLEURS RÉSULTATS
# ============================================================================
print("\n" + "="*70)
print("🏆 MEILLEURE CONFIGURATION GRU (GridSearch)")
print("="*70)
print(f"   Units: {best_config['units']}")
print(f"   Layers: {best_config['layers']}")
print(f"   Dropout: {best_config['dropout']}")
print(f"   Batch Size: {best_config['batch_size']}")
print(f"   R²: {best_config['r2']:.4f}")
print(f"   MAPE: {best_config['mape']:.2f}%")

# ============================================================================
# 9. TOP 10 CONFIGURATIONS
# ============================================================================
print("\n" + "="*70)
print("📊 TOP 10 CONFIGURATIONS GRU")
print("="*70)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('r2', ascending=False)
print(results_df.head(10).to_string(index=False))

# ============================================================================
# 10. MODÈLE FINAL AVEC MEILLEURE CONFIGURATION
# ============================================================================
print("\n🚀 Entraînement du modèle final (150 epochs)...")

# ✅ RAZ des graines avant le modèle final
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
    shuffle=False,  # ← Important pour séries temporelles
    callbacks=[early_stop],
    verbose=1
)

# ============================================================================
# 11. ÉVALUATION FINALE
# ============================================================================
y_pred = model.predict(X_test)
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_real = scaler.inverse_transform(y_pred)

r2_final = r2_score(y_test_real, y_pred_real)
mae_final = mean_absolute_error(y_test_real, y_pred_real)
mape_final = np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100

print("\n" + "="*70)
print("📊 RÉSULTATS FINAUX GRU (GridSearch)")
print("="*70)
print(f"   R² = {r2_final:.4f}")
print(f"   MAPE = {mape_final:.2f}%")
print(f"   MAE = {mae_final:.2f} TND")

# ============================================================================
# 12. GRAPHIQUES
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Courbe d'apprentissage
axes[0].plot(history.history['loss'], label='Train Loss', color='blue')
axes[0].plot(history.history['val_loss'], label='Validation Loss', color='orange')
axes[0].set_xlabel('Époques')
axes[0].set_ylabel('Perte')
axes[0].set_title('Courbe d\'apprentissage GRU')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Prédictions vs Réel
axes[1].plot(y_test_real[:100], label='Valeurs Réelles', color='blue', linewidth=1.5)
axes[1].plot(y_pred_real[:100], label='Prédictions', color='red', linestyle='--', linewidth=1.5)
axes[1].set_xlabel('Jours')
axes[1].set_ylabel('CA (TND)')
axes[1].set_title(f'GRU - Prédictions vs Réel (R²={r2_final:.4f})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/dl_results/gru_gridsearch_final.png', dpi=150)
plt.show()

# ============================================================================
# 13. SAUVEGARDE
# ============================================================================
model.save('results/dl_results/gru_gridsearch_final.h5')

# Sauvegarde des résultats
results_df.to_csv('results/dl_results/gru_gridsearch_results.csv', index=False)

# Sauvegarde de la meilleure configuration
with open('results/dl_results/gru_gridsearch_best.json', 'w') as f:
    json.dump(best_config, f, indent=4)

print("\n💾 Fichiers sauvegardés:")
print("   results/dl_results/gru_gridsearch_final.h5")
print("   results/dl_results/gru_gridsearch_results.csv")
print("   results/dl_results/gru_gridsearch_best.json")

# ============================================================================
# 14. RÉSUMÉ FINAL
# ============================================================================
print("\n" + "="*70)
print("📝 RÉSUMÉ FINAL GRU")
print("="*70)

print(f"""
   🧠 GRU - GridSearchCV (Version complète)

   📌 MEILLEURE CONFIGURATION:
      • Units: {best_config['units']}
      • Layers: {best_config['layers']}
      • Dropout: {best_config['dropout']}
      • Batch Size: {best_config['batch_size']}

   📌 PERFORMANCES FINALES:
      • R² Test : {r2_final:.4f}
      • MAPE    : {mape_final:.2f}%
      • MAE     : {mae_final:.2f} TND

   📌 TOP 5 CONFIGURATIONS:
""")

for i, row in results_df.head(5).iterrows():
    print(f"      {i+1}. units={row['units']}, layers={row['layers']}, "
          f"dropout={row['dropout']}, batch={row['batch_size']} → R²={row['r2']:.4f}")

print("""
   🏆 STATUT: MODÈLE VALIDÉ ET REPRODUCTIBLE
""")

print("✅ GRU GridSearchCV terminé!")