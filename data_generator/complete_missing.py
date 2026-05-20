# complete_missing.py
"""Ajoute les transactions manquantes pour atteindre 300K"""

import psycopg2
import random
import uuid
from datetime import datetime, timedelta

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "caissatndb",
    "user": "postgres",
    "password": "Bouchra1234"
}
SCHEMA = "s5831082f95ef4a1eac9a6a8c484faf0a"

def add_missing_transactions():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Compter les transactions actuelles
    cur.execute(f"SELECT COUNT(*) FROM {SCHEMA}.transactions")
    current = cur.fetchone()[0]
    
    missing = 300000 - current
    
    if missing <= 0:
        print(f"✅ Objectif déjà atteint ! {current:,} transactions")
        conn.close()
        return
    
    print(f"📊 Transactions actuelles: {current:,}")
    print(f"📊 Manquantes: {missing:,}")
    print(f"🚀 Ajout de {missing:,} transactions...")
    
    # Récupérer données nécessaires
    cur.execute(f"SELECT id FROM {SCHEMA}.registers LIMIT 1")
    register_id = cur.fetchone()[0]
    
    cur.execute(f"SELECT id, sale_unit_price FROM {SCHEMA}.products WHERE sale_unit_price > 0")
    products = cur.fetchall()
    
    cur.execute(f"SELECT id FROM {SCHEMA}.customers LIMIT 1")
    customer_row = cur.fetchone()
    customer_id = customer_row[0] if customer_row else None
    
    tx_data = []
    items_data = []
    pay_data = []
    
    # Générer les transactions manquantes
    for i in range(missing):
        tx_id = str(uuid.uuid4())
        
        # Date entre 2025-07-01 et 2026-04-30 (pour combler le gap)
        random_days = random.randint(0, 300)
        tx_date = datetime(2025, 7, 1) + timedelta(days=random_days)
        tx_date = tx_date.replace(
            hour=random.randint(10, 20),
            minute=random.randint(0, 59),
            second=random.randint(0, 59)
        )
        timestamp = tx_date.strftime("%Y-%m-%d %H:%M:%S")
        
        # Produit aléatoire
        product = random.choice(products)
        qty = random.randint(1, 3)
        price = float(product[1])
        total = round(price * qty, 2)
        
        # Transaction
        tx_data.append((
            tx_id, register_id, total, timestamp, customer_id,
            "completion", timestamp, "completion", timestamp
        ))
        
        # Item
        items_data.append((
            str(uuid.uuid4()), tx_id, product[0], price, qty,
            "completion", timestamp
        ))
        
        # Paiement
        pay_type = random.choice(["CASH", "CREDIT"])
        pay_data.append((
            str(uuid.uuid4()), tx_id, total, pay_type,
            "completion", timestamp
        ))
        
        if (i + 1) % 1000 == 0:
            print(f"   ⏳ {i+1}/{missing}...")
            conn.commit()
    
    # Insertion
    from psycopg2.extras import execute_values
    
    print("💾 Insertion des transactions...")
    execute_values(cur, f"""
        INSERT INTO {SCHEMA}.transactions
        (id, register_id, total, date, customer_id,
         created_by, created_date, last_modified_by, last_modified_date)
        VALUES %s
        ON CONFLICT (id) DO NOTHING
    """, tx_data, page_size=500)
    
    print("💾 Insertion des items...")
    execute_values(cur, f"""
        INSERT INTO {SCHEMA}.transaction_items
        (id, transaction_id, product_id, price, qty, created_by, created_date)
        VALUES %s
        ON CONFLICT (id) DO NOTHING
    """, items_data, page_size=500)
    
    print("💾 Insertion des paiements...")
    execute_values(cur, f"""
        INSERT INTO {SCHEMA}.payments
        (id, transaction_id, amount, type, created_by, created_date)
        VALUES %s
        ON CONFLICT (id) DO NOTHING
    """, pay_data, page_size=500)
    
    conn.commit()
    
    # Vérification finale
    cur.execute(f"SELECT COUNT(*) FROM {SCHEMA}.transactions")
    final = cur.fetchone()[0]
    
    cur.close()
    conn.close()
    
    print("\n" + "=" * 50)
    print("✅ AJOUT TERMINÉ !")
    print(f"   Avant: {current:,}")
    print(f"   Ajout: {missing:,}")
    print(f"   Total: {final:,} / 300,000")
    print("=" * 50)

if __name__ == "__main__":
    add_missing_transactions()