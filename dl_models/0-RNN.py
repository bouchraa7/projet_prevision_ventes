"""
RNN (Simple RNN) - Prédiction des ventes
Avec optimisation des paramètres (RandomizedSearchCV via scikeras)
"""

import psycopg2
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

print("="*60)
print("🧠 RNN (SIMPLE RNN) - Recherche automatique des paramètres")
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

# Paramètres RNN à tester
SEQ_LENGTH = 7  # Séquence de 7 jours pour prédire le jour suivant

# ============================================================================
# CHARGEMENT DES DONNÉES
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
# FEATURE ENGINEERING
# ============================================================================
print("\n⚙️ Création des features...")

df["date"] = pd.to_datetime(df["date"])
df["is_weekend"] = (df["day_of_week"] >= 4).astype(int)
df["sales_per_transaction"] = df["revenue"] / df["nb_transactions"]
df["revenue_lag_1"] = df["revenue"].shift(1)
df["revenue_ma_7"] = df["revenue"].rolling(window=7).mean()

# ============================================================================
# NETTOYAGE
# ============================================================================
df = df.replace([np.inf, -np.inf], 0).fillna(0)

# ============================================================================
# 8 FEATURES
# ============================================================================
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
# PRÉPARATION DES DONNÉES POUR RNN (séquence temporelle)
# ============================================================================
print(f"\n🔄 Création des séquences temporelles (seq_length={SEQ_LENGTH})...")

def create_sequences(X, y, seq_length):
    """Crée des séquences pour RNN"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

X = df[features].values
y = df["revenue"].values

# Split temporel (80% entraînement, 20% test)
split_index = int(len(X) * 0.8)
X_train_raw, X_test_raw = X[:split_index], X[split_index:]
y_train_raw, y_test_raw = y[:split_index], y[split_index:]

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Création des séquences
X_train, y_train = create_sequences(X_train_scaled, y_train_raw, SEQ_LENGTH)
X_test, y_test = create_sequences(X_test_scaled, y_test_raw, SEQ_LENGTH)

print(f"   Train: {X_train.shape[0]} séquences")
print(f"   Test : {X_test.shape[0]} séquences")
print(f"   Chaque séquence: {SEQ_LENGTH} jours × {len(features)} features")

# ============================================================================
# FONCTION DE CRÉATION DU MODÈLE RNN
# ============================================================================
def create_rnn_model(units=50, dropout=0.2, learning_rate=0.001):
    """Crée un modèle RNN simple"""
    model = Sequential()
    model.add(SimpleRNN(units=units, activation='tanh', input_shape=(SEQ_LENGTH, len(features))))
    model.add(Dropout(dropout))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# ============================================================================
# RECHERCHE DES MEILLEURS PARAMÈTRES (RandomizedSearchCV)
# ============================================================================
print("\n🔍 RandomizedSearchCV - Recherche automatique des paramètres...")

# Distribution des paramètres à tester
param_dist = {
    'units': [32, 64, 128],           # Taille de la mémoire RNN
    'dropout': [0.1, 0.2, 0.3],       # Anti-surapprentissage
    'learning_rate': [0.01, 0.001, 0.0005],  # Vitesse d'apprentissage
    'batch_size': [16, 32],           # Taille des lots
    'epochs': [50, 100]               # Nombre d'époques
}

print("\n   Paramètres testés automatiquement:")
for param, values in param_dist.items():
    print(f"   • {param}: {values}")

# Création du wrapper Keras pour sklearn
keras_regressor = KerasRegressor(
    model=create_rnn_model,
    units=64,
    dropout=0.2,
    learning_rate=0.001,
    batch_size=16,
    epochs=50,
    verbose=0
)

# TimeSeriesSplit (respecte l'ordre chronologique)
tscv = TimeSeriesSplit(n_splits=3)

# RandomizedSearch
random_search = RandomizedSearchCV(
    estimator=keras_regressor,
    param_distributions=param_dist,
    n_iter=30,  # Teste 30 combinaisons aléatoires
    cv=tscv,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("\n⏳ Recherche en cours (environ 5-10 minutes)...")
random_search.fit(X_train, y_train)

# ============================================================================
# MEILLEURS PARAMÈTRES TROUVÉS AUTOMATIQUEMENT
# ============================================================================
best_params = random_search.best_params_
best_score = random_search.best_score_

print("\n" + "="*60)
print("🏆 MEILLEURS PARAMÈTRES RNN (trouvés automatiquement)")
print("="*60)
for param, value in best_params.items():
    print(f"   {param}: {value}")
print(f"\n   Meilleur score CV: {best_score:.4f}")

# ============================================================================
# MODÈLE FINAL
# ============================================================================
print("\n🚀 Entraînement du modèle final...")

final_model = create_rnn_model(
    units=best_params['units'],
    dropout=best_params['dropout'],
    learning_rate=best_params['learning_rate']
)

history = final_model.fit(
    X_train, y_train,
    epochs=best_params['epochs'],
    batch_size=best_params['batch_size'],
    validation_split=0.1,
    verbose=1
)

# ============================================================================
# PRÉDICTIONS
# ============================================================================
y_pred_train = final_model.predict(X_train, verbose=0)
y_pred_test = final_model.predict(X_test, verbose=0)

# ============================================================================
# ÉVALUATION
# ============================================================================
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

print("\n" + "="*60)
print("📊 RÉSULTATS RNN")
print("="*60)
print(f"   R² Entraînement : {r2_train:.4f}")
print(f"   R² Test         : {r2_test:.4f}")
print(f"   MAE Test        : {mae_test:.2f} TND")
print(f"   RMSE Test       : {rmse_test:.2f} TND")
print(f"   MAPE Test       : {mape_test:.2f}%")

# ============================================================================
# DIAGNOSTIC
# ============================================================================
print("\n" + "="*60)
print("🔍 DIAGNOSTIC RNN")
print("="*60)

overfit = r2_train - r2_test
if overfit > 0.1:
    print(f"   ⚠️ Surapprentissage (écart = {overfit:.4f})")
elif overfit > 0.05:
    print(f"   ✅ Léger surapprentissage acceptable (écart = {overfit:.4f})")
else:
    print(f"   ✅ Pas de surapprentissage (écart = {overfit:.4f})")

# ============================================================================
# COMPARAISON FINALE
# ============================================================================
print("\n" + "="*60)
print("📊 COMPARAISON RNN vs LSTM vs GRU vs MLP")
print("="*60)
print(f"   RNN  : R² = {r2_test:.4f} | MAPE = {mape_test:.2f}%")
print(f"   LSTM : R² = 0.4958 | MAPE = 15.09%")
print(f"   GRU  : R² = 0.4835 | MAPE = 15.97%")
print(f"   MLP  : R² = 0.6560 | MAPE = 9.99%")

# ============================================================================
# CALCUL DU NOMBRE DE PARAMÈTRES
# ============================================================================
total_params = final_model.count_params()
print(f"\n🔢 Nombre total de paramètres RNN: {total_params:,}")

print("\n✅ RNN terminé !")