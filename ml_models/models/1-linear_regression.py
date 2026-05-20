"""
LINEAR REGRESSION - 8 features
R² = 0.5552 | MAPE = 4.36%
"""

import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

# ============================================================================
# CONFIGURATION
# ============================================================================

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dossiers
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Base de données
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "caissatndb",
    "user": "postgres",
    "password": "Bouchra1234"
}
SCHEMA = "s5831082f95ef4a1eac9a6a8c484faf0a"

# Features finales
FEATURES = [
    'nb_transactions',
    'sales_per_transaction',
    'avg_price',
    'day_of_week',
    'is_weekend',
    'month',
    'revenue_lag_1',
    'revenue_ma_7'
]

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ============================================================================
# FONCTIONS
# ============================================================================

def load_data():
    """Charge et prépare les données depuis PostgreSQL."""
    logging.info("Chargement des données...")
    
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
    
    # Feature engineering
    df['date'] = pd.to_datetime(df['date'])
    df['is_weekend'] = (df['day_of_week'] >= 4).astype(int)
    df['sales_per_transaction'] = df['revenue'] / df['nb_transactions']
    df['revenue_lag_1'] = df['revenue'].shift(1)
    df['revenue_ma_7'] = df['revenue'].rolling(7).mean()
    
    # Nettoyage
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    logging.info(f"Données chargées : {len(df)} jours")
    return df


def split_data(df, features, target='revenue', test_size=TEST_SIZE):
    """Split chronologique (pas aléatoire)."""
    X = df[features].values
    y = df[target].values
    
    split_idx = int(len(X) * (1 - test_size))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logging.info(f"Split: Train={len(X_train)} jours, Test={len(X_test)} jours")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Entraîne un modèle de régression linéaire avec normalisation."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    return model, scaler


def evaluate_model(model, scaler, X_test, y_test, features):
    """Évalue le modèle et retourne les métriques + importance des features."""
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    # Métriques
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Importance des features (coefficients standardisés)
    coeffs = pd.DataFrame({
        'feature': features,
        'coefficient': model.coef_
    })
    coeffs['abs_coefficient'] = coeffs['coefficient'].abs()
    coeffs['importance_pct'] = (coeffs['abs_coefficient'] / coeffs['abs_coefficient'].sum()) * 100
    coeffs = coeffs.sort_values('abs_coefficient', ascending=False).reset_index(drop=True)
    
    metrics = {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'y_pred': y_pred,
        'y_test': y_test,
        'coeffs': coeffs
    }
    
    return metrics


def plot_results(metrics, save_path='figures/linear_regression_result.png'):
    """Génère les graphiques : prédictions vs réalité + importance des features."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Graphique 1 : Prédictions vs Valeurs réelles
    ax1 = axes[0]
    ax1.scatter(metrics['y_test'], metrics['y_pred'], alpha=0.6, 
                color='steelblue', edgecolors='black', linewidth=0.5)
    
    min_val = min(metrics['y_test'].min(), metrics['y_pred'].min())
    max_val = max(metrics['y_test'].max(), metrics['y_pred'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prédiction parfaite')
    
    ax1.set_xlabel('Valeurs Réelles (TND)', fontsize=12)
    ax1.set_ylabel('Prédictions (TND)', fontsize=12)
    ax1.set_title(f'Régression Linéaire\nR² = {metrics["r2"]:.4f} | MAPE = {metrics["mape"]:.2f}%', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Graphique 2 : Top features (importance en pourcentage)
    ax2 = axes[1]
    top5 = metrics['coeffs'].head(5)
    colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in top5['coefficient']]
    bars = ax2.barh(top5['feature'], top5['coefficient'], color=colors, edgecolor='black')
    ax2.axvline(x=0, color='black', linewidth=0.5)
    ax2.set_xlabel('Coefficient standardisé', fontsize=12)
    ax2.set_title('Top 5 : Importance des features', fontsize=14)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Ajout des pourcentages
    for i, (idx, row) in enumerate(top5.iterrows()):
        ax2.text(row['coefficient'] + (5 if row['coefficient'] > 0 else -50), 
                 i, f"  {row['importance_pct']:.1f}%", 
                 va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path.replace('.png', '.pdf'))
    plt.close()
    
    logging.info(f"Graphiques sauvegardés : {save_path}")


def save_model(model, scaler, metrics, features, path='results/linear_regression_8features.pkl'):
    """Sauvegarde le modèle et ses métadonnées."""
    with open(path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'features': features,
            'r2': metrics['r2'],
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'mape': metrics['mape'],
            'coeffs': metrics['coeffs']
        }, f)
    logging.info(f"Modèle sauvegardé : {path}")


def print_results(metrics):
    """Affiche les résultats de manière claire."""
    print("\n" + "="*70)
    print("📊 RÉSULTATS LINEAR REGRESSION")
    print("="*70)
    print(f"   R²   = {metrics['r2']:.4f}")
    print(f"   MAE  = {metrics['mae']:.2f} TND")
    print(f"   RMSE = {metrics['rmse']:.2f} TND")
    print(f"   MAPE = {metrics['mape']:.2f}%")
    
    print("\n📊 IMPORTANCE DES FEATURES")
    print("-"*70)
    for _, row in metrics['coeffs'].iterrows():
        arrow = "▲" if row['coefficient'] > 0 else "▼"
        print(f"   {arrow} {row['feature']:<25}: {row['coefficient']:10.2f}  ({row['importance_pct']:.1f}%)")
    
    print("\n💡 INTERPRÉTATION :")
    print("   Les coefficients sont standardisés (même échelle).")
    print("   Plus |coeff| est grand, plus la feature impacte la prédiction.")
    print(f"   Feature la plus importante : {metrics['coeffs'].iloc[0]['feature']} ({metrics['coeffs'].iloc[0]['importance_pct']:.1f}%)")
    print("="*70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    logging.info("="*70)
    logging.info("LINEAR REGRESSION - 8 features")
    logging.info("="*70)
    
    # 1. Chargement
    df = load_data()
    
    # 2. Split
    X_train, X_test, y_train, y_test = split_data(df, FEATURES)
    
    # 3. Entraînement
    model, scaler = train_model(X_train, y_train)
    
    # 4. Évaluation
    metrics = evaluate_model(model, scaler, X_test, y_test, FEATURES)
    
    # 5. Affichage
    print_results(metrics)
    
    # 6. Graphiques
    plot_results(metrics)
    
    # 7. Sauvegarde
    save_model(model, scaler, metrics, FEATURES)
    
    logging.info("✅ Linear Regression terminé !")


if __name__ == "__main__":
    main()