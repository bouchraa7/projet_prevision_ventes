"""
Préparation des données pour les modèles ML et séries temporelles
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import DB_CONFIG, POS_SCHEMA

class DataPreparator:
    """Classe pour préparer les données pour l'analyse"""
    
    def __init__(self):
        self.SCHEMA = POS_SCHEMA
        self.conn = None
        self.df_daily = None
        self.df_hourly = None
        
    def load_daily_data(self):
        """Charge les données journalières pour ML"""
        self.conn = psycopg2.connect(**DB_CONFIG)
        
        sql = f"""
        SELECT 
            DATE(transaction_date) as date,
            COUNT(DISTINCT transaction_id) as nb_transactions,
            COUNT(*) as nb_items,
            SUM(line_total) as revenue,
            SUM(qty) as qty_sold,
            COUNT(DISTINCT product_id) as nb_products,
            AVG(sold_price) as avg_price,
            SUM(gross_margin) as gross_margin,
            EXTRACT(DOW FROM DATE(transaction_date)) as day_of_week,
            EXTRACT(MONTH FROM DATE(transaction_date)) as month,
            EXTRACT(QUARTER FROM DATE(transaction_date)) as quarter,
            EXTRACT(YEAR FROM DATE(transaction_date)) as year,
            MAX(is_ramadan) as is_ramadan,
            MAX(is_eid_al_fitr) as is_eid_al_fitr,
            MAX(is_eid_al_adha) as is_eid_al_adha,
            MAX(is_rentree_scolaire) as is_rentree_scolaire,
            MAX(is_nouvel_an) as is_nouvel_an,
            MAX(is_ete_peak) as is_ete_peak,
            MAX(is_soldes_hiver) as is_soldes_hiver
        FROM {self.SCHEMA}.pos_analytics
        WHERE is_refund = false
        GROUP BY DATE(transaction_date)
        ORDER BY date
        """
        
        self.df_daily = pd.read_sql(sql, self.conn)
        self.conn.close()
        
        # ✅ CORRECTION: Convertir la colonne date en datetime
        self.df_daily['date'] = pd.to_datetime(self.df_daily['date'])
        
        # Features supplémentaires
        self.df_daily['is_weekend'] = (self.df_daily['day_of_week'] >= 4).astype(int)
        self.df_daily['is_month_start'] = (self.df_daily['date'].dt.day <= 5).astype(int)
        self.df_daily['is_month_end'] = (self.df_daily['date'].dt.day >= 25).astype(int)
        self.df_daily['day_of_year'] = self.df_daily['date'].dt.dayofyear
        
        # Lags (pour les séries temporelles)
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            self.df_daily[f'revenue_lag_{lag}'] = self.df_daily['revenue'].shift(lag)
            self.df_daily[f'transactions_lag_{lag}'] = self.df_daily['nb_transactions'].shift(lag)
        
        # Rolling windows
        for window in [7, 14, 30]:
            self.df_daily[f'revenue_ma_{window}'] = self.df_daily['revenue'].rolling(window).mean()
            self.df_daily[f'revenue_std_{window}'] = self.df_daily['revenue'].rolling(window).std()
        
        print(f"✅ Données chargées: {len(self.df_daily)} jours")
        print(f"📅 Période: {self.df_daily['date'].min()} → {self.df_daily['date'].max()}")
        print(f"💰 CA total: {self.df_daily['revenue'].sum():,.2f} TND")
        
        return self.df_daily
    
    def get_features(self):
        """Retourne la liste des features disponibles"""
        return [
            'nb_transactions', 'nb_items', 'qty_sold', 'nb_products',
            'day_of_week', 'is_weekend', 'month', 'quarter', 'year',
            'is_month_start', 'is_month_end', 'is_ramadan', 'is_eid_al_fitr',
            'is_eid_al_adha', 'is_rentree_scolaire', 'is_nouvel_an',
            'is_ete_peak', 'is_soldes_hiver'
        ]
    
    def prepare_ml_data(self, target='revenue', test_size=0.2):
        """Prépare les données pour les modèles ML"""
        # Supprimer les lignes avec des NaN
        df_clean = self.df_daily.dropna()
        
        features = self.get_features()
        # Ne garder que les features qui existent
        available_features = [f for f in features if f in df_clean.columns]
        
        X = df_clean[available_features].values
        y = df_clean[target].values
        dates = df_clean['date'].values
        
        # Split temporel
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        dates_train, dates_test = dates[:split_idx], dates[split_idx:]
        
        # Normalisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"📊 Train: {len(X_train)} jours | Test: {len(X_test)} jours")
        print(f"📊 Features utilisées: {len(available_features)}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'dates_train': dates_train,
            'dates_test': dates_test,
            'scaler': scaler,
            'features': available_features
        }
    
    def prepare_time_series(self, target='revenue'):
        """Prépare les données pour les modèles de séries temporelles"""
        ts_data = self.df_daily[['date', target]].copy()
        ts_data.columns = ['ds', 'y']  # Convention Prophet
        ts_data = ts_data.dropna()
        
        print(f"📈 Série temporelle: {len(ts_data)} points")
        
        return ts_data


if __name__ == "__main__":
    prep = DataPreparator()
    df = prep.load_daily_data()
    print("\n✅ Test réussi!")
    print(f"📊 Aperçu des données:\n{df.head()}")