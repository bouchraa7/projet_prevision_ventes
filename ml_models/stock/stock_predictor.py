"""
STOCK PREDICTOR - Prédiction des ruptures de stock et alertes
Basé sur les prévisions de ventes LightGBM
"""

import pandas as pd
import numpy as np
import psycopg2
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    """
    Prédiction des ruptures de stock et génération d'alertes
    """
    
    def __init__(self, schema, db_config):
        self.schema = schema
        self.db_config = db_config
        self.model = None
        self.scaler = None
        
    def load_model(self, model_path='results/best_ml_model.pkl'):
        """Charge le modèle LightGBM entraîné"""
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
            print(f"✅ Modèle chargé: {model_path}")
        except:
            print(f"⚠️ Modèle non trouvé: {model_path}")
    
    def get_current_stock(self):
        """Récupère les niveaux de stock actuels"""
        conn = psycopg2.connect(**self.db_config)
        query = f"""
        SELECT 
            id,
            name,
            COALESCE(stock_qty, 0) as current_stock,
            COALESCE(stock_min_limit, 5) as min_stock,
            COALESCE(stock_max_limit, 500) as max_stock,
            sale_unit_price
        FROM {self.schema}.products
        WHERE sale_unit_price > 0
        ORDER BY name
        """
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    
    def predict_daily_sales(self):
        """Prédit les ventes moyennes journalières basées sur l'historique"""
        conn = psycopg2.connect(**self.db_config)
        # ✅ CORRECTION: transaction_date est dans la table transactions
        query = f"""
        SELECT 
            DATE(t.date) as date,
            SUM(ti.qty) as qty_sold
        FROM {self.schema}.transaction_items ti
        JOIN {self.schema}.transactions t ON ti.transaction_id = t.id
        WHERE t.refund_id IS NULL
          AND t.date IS NOT NULL
        GROUP BY DATE(t.date)
        ORDER BY date DESC
        LIMIT 30
        """
        df = pd.read_sql(query, conn)
        conn.close()
        
        if len(df) > 0:
            avg_daily_sales = df['qty_sold'].mean()
        else:
            avg_daily_sales = 10  # Valeur par défaut
        
        return avg_daily_sales
    
    def calculate_risk_score(self, current_stock, avg_daily_sales, days_to_reorder=7):
        """Calcule le score de risque de rupture (0-100)"""
        if avg_daily_sales <= 0:
            avg_daily_sales = 1
        
        days_of_stock = current_stock / avg_daily_sales
        
        if days_of_stock <= 0:
            return 100  # Critique
        elif days_of_stock <= days_to_reorder:
            return 80  # Urgent
        elif days_of_stock <= days_to_reorder * 2:
            return 50  # Attention
        elif days_of_stock <= days_to_reorder * 3:
            return 20  # Surveiller
        else:
            return 0   # Safe
    
    def get_stock_alerts(self, risk_threshold=50):
        """Génère les alertes de stock"""
        print("\n" + "="*70)
        print("📊 ANALYSE DES STOCKS - ALERTES")
        print("="*70)
        
        # Récupérer les stocks
        stocks = self.get_current_stock()
        
        # Prédire les ventes moyennes
        avg_daily_sales = self.predict_daily_sales()
        print(f"📈 Ventes moyennes journalières (30j): {avg_daily_sales:.1f} unités")
        
        # Calculer les indicateurs
        stocks['avg_daily_sales'] = avg_daily_sales
        stocks['days_of_stock'] = stocks['current_stock'] / max(avg_daily_sales, 0.01)
        stocks['days_of_stock'] = stocks['days_of_stock'].round(1)
        stocks['risk_score'] = stocks.apply(
            lambda x: self.calculate_risk_score(x['current_stock'], avg_daily_sales), 
            axis=1
        )
        stocks['status'] = stocks['risk_score'].apply(
            lambda x: '🔴 CRITIQUE' if x >= 80 else ('🟠 URGENT' if x >= 50 else ('🟡 ATTENTION' if x >= 20 else '🟢 OK'))
        )
        
        # Quantité à recommander (réapprovisionnement à 50% du max)
        stocks['recommended_order'] = stocks.apply(
            lambda x: max(0, int((x['max_stock'] - x['current_stock']) * 0.6)) if x['risk_score'] > 20 else 0,
            axis=1
        )
        
        # Filtrer les alertes
        alerts = stocks[stocks['risk_score'] >= risk_threshold].sort_values('risk_score', ascending=False)
        
        return stocks, alerts
    
    def display_alerts(self, alerts):
        """Affiche les alertes de manière claire"""
        if len(alerts) == 0:
            print("\n✅ Aucune alerte de stock à signaler")
            return
        
        print(f"\n⚠️ {len(alerts)} PRODUITS NÉCESSITENT UNE ATTENTION")
        print("-"*80)
        
        for _, row in alerts.head(10).iterrows():
            status_icon = "🔴" if row['risk_score'] >= 80 else ("🟠" if row['risk_score'] >= 50 else "🟡")
            print(f"\n{status_icon} {row['name'][:50]}")
            print(f"   📦 Stock actuel : {row['current_stock']:.0f} unités")
            print(f"   📅 Jours restants : {row['days_of_stock']:.0f} jours")
            print(f"   📊 Score risque : {row['risk_score']:.0f}/100")
            if row['recommended_order'] > 0:
                print(f"   🛒 Commande recommandée : {row['recommended_order']:.0f} unités")
        
        print("\n" + "="*70)
        
        # Résumé par niveau
        print("\n📊 RÉSUMÉ DES ALERTES")
        print("-"*40)
        print(f"🔴 CRITIQUE (< 7 jours) : {len(alerts[alerts['risk_score'] >= 80])}")
        print(f"🟠 URGENT (7-14 jours)  : {len(alerts[(alerts['risk_score'] >= 50) & (alerts['risk_score'] < 80)])}")
        print(f"🟡 ATTENTION (14-21 j) : {len(alerts[(alerts['risk_score'] >= 20) & (alerts['risk_score'] < 50)])}")
        print("="*70)
    
    def generate_report(self):
        """Génère un rapport complet des stocks"""
        stocks, alerts = self.get_stock_alerts(risk_threshold=20)
        
        print("\n" + "="*70)
        print("📊 RAPPORT DE STOCK COMPLET")
        print("="*70)
        
        # Top 10 des stocks les plus bas
        print("\n📉 TOP 10 PRODUITS AVEC STOCK LE PLUS BAS")
        print("-"*55)
        low_stocks = stocks.nsmallest(10, 'current_stock')[['name', 'current_stock', 'days_of_stock', 'status']]
        for _, row in low_stocks.iterrows():
            print(f"   {row['name'][:45]:45s} : {row['current_stock']:.0f} u ({row['days_of_stock']:.0f} j) {row['status']}")
        
        # Statistiques globales
        print("\n📊 STATISTIQUES GLOBALES")
        print("-"*50)
        print(f"   Total produits analysés : {len(stocks)}")
        print(f"   Produits en stock bas (< 14 jours) : {len(stocks[stocks['days_of_stock'] < 14])}")
        print(f"   Produits en stock critique (< 7 jours) : {len(stocks[stocks['days_of_stock'] < 7])}")
        print(f"   Ventes moyennes journalières : {stocks['avg_daily_sales'].iloc[0]:.1f} unités")
        
        # Stock total
        total_stock = stocks['current_stock'].sum()
        print(f"\n💰 VALEUR TOTALE DU STOCK")
        print("-"*50)
        if 'sale_unit_price' in stocks.columns:
            total_value = (stocks['current_stock'] * stocks['sale_unit_price']).sum()
            print(f"   Stock total : {total_stock:,.0f} unités")
            print(f"   Valeur totale : {total_value:,.2f} TND")
        
        print("="*70)
        
        return stocks, alerts
    
    def export_alerts_to_csv(self, alerts, filename='stock_alerts.csv'):
        """Exporte les alertes vers un fichier CSV"""
        if len(alerts) > 0:
            alerts.to_csv(filename, index=False)
            print(f"\n💾 Alertes exportées vers {filename}")
        else:
            print("\n✅ Aucune alerte à exporter")


# ============================================================================
# EXÉCUTION PRINCIPALE
# ============================================================================
if __name__ == "__main__":
    
    # Configuration
    DB_CONFIG = {
        "host": "localhost",
        "port": 5432,
        "database": "caissatndb",
        "user": "postgres",
        "password": "Bouchra1234"
    }
    SCHEMA = "s5831082f95ef4a1eac9a6a8c484faf0a"
    
    print("="*70)
    print("📦 STOCK PREDICTOR - Analyse des risques de rupture")
    print("="*70)
    
    # Initialisation
    predictor = StockPredictor(SCHEMA, DB_CONFIG)
    
    # 1. Générer le rapport complet
    stocks, alerts = predictor.generate_report()
    
    # 2. Afficher les alertes
    predictor.display_alerts(alerts)
    
    # 3. Exporter les alertes
    if len(alerts) > 0:
        predictor.export_alerts_to_csv(alerts, 'stock_alerts.csv')