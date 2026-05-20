# data_generator/generator.py
"""
Générateur intelligent de transactions - Version 300K
- Garde les transactions réelles existantes
- Ajoute 296 942 transactions synthétiques
- Période: 2024-01-01 → 2026-04-30
- Patterns: saisonnalité, Ramadan, heures de pointe, weekends, promotions
"""

import uuid
import random
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import psycopg2
from psycopg2.extras import execute_batch
import numpy as np
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    DB_CONFIG, POS_SCHEMA, GENERATOR_CONFIG,
    SEASONAL_EVENTS, RUSH_HOURS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCHEMA = POS_SCHEMA
CFG = GENERATOR_CONFIG

# ✅ Paiements : uniquement ESPÈCE ou CRÉDIT
PAYMENT_TYPES = ["CASH", "CREDIT"]
PAYMENT_WEIGHTS = [65, 35]

# Distribution des produits par transaction
ITEM_COUNTS = [1, 2, 3, 4, 5, 6]
ITEM_WEIGHTS = [25, 30, 20, 12, 8, 5]

# Distribution des quantités
QUANTITIES = [1, 2, 3, 4, 5]
QTY_WEIGHTS = [40, 30, 15, 10, 5]

# Périodes Ramadan spécifiques
RAMADAN_PERIODS = [
    (datetime(2024, 3, 11), datetime(2024, 4, 9)),
    (datetime(2025, 3, 1), datetime(2025, 3, 29)),
    (datetime(2026, 2, 18), datetime(2026, 3, 19)),
]

# Mots-clés pour promotions saisonnières
SUMMER_KEYWORDS = ["glace", "eau", "coca", "fanta", "sprite", "jus", "punch", "ice"]
WINTER_KEYWORDS = ["chocolat", "café", "thé", "lait", "vanille", "caramel"]
RAMADAN_KEYWORDS = ["datte", "jus", "lait", "soupe", "harira"]


# ─────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────

def is_ramadan(date: datetime) -> bool:
    """Vérifie si une date est pendant le Ramadan"""
    for start, end in RAMADAN_PERIODS:
        if start <= date <= end:
            return True
    return False


def get_event_boost(date: datetime) -> float:
    """Calcule le boost pour un événement spécial"""
    month, day = date.month, date.day
    
    # Nouvel An
    if month == 1 and day <= 2:
        return 1.8
    # Aïd Fitr (approximatif)
    if (month == 4 and 10 <= day <= 14) or (month == 3 and 30 <= day <= 31):
        return 2.5
    # Aïd Adha
    if (month == 6 and 17 <= day <= 24) or (month == 5 and 27 <= day <= 31):
        return 2.2
    # Rentrée scolaire
    if month == 9 and day <= 15:
        return 1.6
    # Noël
    if month == 12 and day >= 24:
        return 1.5
    # Saint-Sylvestre
    if month == 12 and day == 31:
        return 1.9
    
    return 1.0


def daily_multiplier(date: datetime) -> float:
    """Calcule le multiplicateur pour un jour donné (saisonnalité + tendance)"""
    mult = 1.0
    m, d = date.month, date.day
    
    # Boost saisonnier
    if m in [6, 7, 8]:  # Été
        mult *= 1.4
    elif m in [12, 1, 2]:  # Hiver
        mult *= 1.2
    elif m in [3, 4, 5]:  # Printemps
        mult *= 1.1
    
    # Événements spéciaux
    mult *= get_event_boost(date)
    
    # Ramadan (réduction d'activité)
    if is_ramadan(date):
        mult *= 0.65
    
    # Week-end (vendredi, samedi, dimanche)
    if date.weekday() >= 4:
        mult *= 1.35
    
    # Tendance de croissance (de 2024 à 2026)
    start_ref = datetime(2024, 1, 1)
    days_elapsed = (date - start_ref).days
    growth = (1 + CFG["growth_rate_annual"]) ** (days_elapsed / 365.0)
    mult *= growth
    
    # Bruit aléatoire (±15%)
    mult *= np.random.normal(1.0, 0.08)
    
    return max(0.4, mult)


def get_product_seasonal_boost(product_name: str, date: datetime) -> float:
    """Calcule le boost pour un produit selon la saison"""
    name_lower = product_name.lower()
    boost = 1.0
    
    # Boost été
    if date.month in [6, 7, 8]:
        if any(k in name_lower for k in SUMMER_KEYWORDS):
            boost = 2.2
    
    # Boost hiver
    elif date.month in [12, 1, 2]:
        if any(k in name_lower for k in WINTER_KEYWORDS):
            boost = 1.9
    
    # Boost Ramadan
    if is_ramadan(date):
        if any(k in name_lower for k in RAMADAN_KEYWORDS):
            boost *= 1.7
    
    return boost


def pick_transaction_hour(date: datetime) -> datetime:
    """Choisit une heure réaliste selon les heures de pointe"""
    hours = list(RUSH_HOURS.keys())
    weights = list(RUSH_HOURS.values())
    
    # Pendant Ramadan, heures décalées
    if is_ramadan(date):
        ramadan_hours = [17, 18, 19, 20, 21, 22]
        ramadan_weights = [1.5, 2.5, 2.0, 1.5, 1.0, 0.8]
        hour = random.choices(ramadan_hours, weights=ramadan_weights)[0]
    else:
        hour = random.choices(hours, weights=weights)[0]
    
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    
    return date.replace(hour=hour, minute=minute, second=second)


# ─────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES EXISTANTES
# ─────────────────────────────────────────────────────────────────────────

def load_existing_data(conn) -> Dict:
    """Charge les données de référence depuis la base"""
    cur = conn.cursor()
    
    # Produits
    cur.execute(f"""
        SELECT id, name, sale_unit_price, stock_qty
        FROM {SCHEMA}.products
        WHERE sale_unit_price > 0
    """)
    products = [
        {"id": r[0], "name": r[1], "price": float(r[2]), "stock": r[3] or 100}
        for r in cur.fetchall()
    ]
    
    # Clients
    cur.execute(f"SELECT id FROM {SCHEMA}.customers")
    customers = [r[0] for r in cur.fetchall()]
    
    # Registre
    cur.execute(f"SELECT id FROM {SCHEMA}.registers LIMIT 1")
    register_id = cur.fetchone()[0]
    
    # Compter les transactions réelles
    cur.execute(f"""
        SELECT COUNT(*) FROM {SCHEMA}.transactions 
        WHERE created_by IS NULL OR created_by != 'generator'
    """)
    real_count = cur.fetchone()[0]
    
    cur.close()
    
    log.info(f"📦 Produits: {len(products)}")
    log.info(f"👥 Clients: {len(customers)}")
    log.info(f"📀 Transactions réelles: {real_count:,}")
    
    return {
        "products": products,
        "customers": customers,
        "register_id": register_id,
        "real_count": real_count
    }


# ─────────────────────────────────────────────────────────────────────────
# GÉNÉRATION D'UNE TRANSACTION
# ─────────────────────────────────────────────────────────────────────────

def generate_transaction(
    date: datetime,
    ref: Dict,
    daily_boost: float,
) -> Optional[Dict]:
    """Génère une transaction synthétique"""
    
    # Annulation aléatoire
    if random.random() < CFG["cancellation_rate"]:
        return None
    
    # Nombre d'articles
    n_items = random.choices(ITEM_COUNTS, weights=ITEM_WEIGHTS)[0]
    
    # Sélection des produits
    selected_products = random.sample(ref["products"], min(n_items, len(ref["products"])))
    
    items = []
    total = 0.0
    
    for product in selected_products:
        qty = random.choices(QUANTITIES, weights=QTY_WEIGHTS)[0]
        price = product["price"]
        
        # Boost saisonnier
        seasonal_boost = get_product_seasonal_boost(product["name"], date)
        price *= seasonal_boost
        
        # Promotion (20% des cas en période de forte affluence)
        if daily_boost > 1.3 and random.random() < 0.18:
            price *= random.uniform(0.80, 0.92)
        
        # Bruit sur le prix
        price *= random.uniform(0.95, 1.05)
        price = round(max(0.01, price), 3)
        
        line_total = price * qty
        items.append({
            "product_id": product["id"],
            "price": price,
            "qty": qty
        })
        total += line_total
    
    # Outlier (panier exceptionnel)
    if random.random() < CFG["outlier_rate"]:
        total *= random.uniform(3.0, 7.0)
        total = round(total, 2)
    
    # Client (60% identifiés)
    customer_id = (
        random.choice(ref["customers"])
        if ref["customers"] and random.random() < 0.60
        else None
    )
    
    # Paiement
    pay_type = random.choices(PAYMENT_TYPES, weights=PAYMENT_WEIGHTS)[0]
    tx_time = pick_transaction_hour(date)
    
    return {
        "id": str(uuid.uuid4()),
        "register_id": ref["register_id"],
        "total": round(total, 2),
        "date": tx_time,
        "customer_id": customer_id,
        "items": items,
        "payment": {
            "id": str(uuid.uuid4()),
            "amount": round(total, 2),
            "type": pay_type
        }
    }


# ─────────────────────────────────────────────────────────────────────────
# INSERTION BATCH
# ─────────────────────────────────────────────────────────────────────────

def insert_batch(conn, transactions: List[Dict]):
    """Insertion batch optimisée"""
    if not transactions:
        return
    
    cur = conn.cursor()
    
    tx_rows = []
    item_rows = []
    pay_rows = []
    
    for tx in transactions:
        tx_rows.append((
            tx["id"], tx["register_id"], tx["total"],
            tx["date"], tx.get("customer_id"),
            "generator", tx["date"], "generator", tx["date"],
        ))
        
        for it in tx["items"]:
            item_rows.append((
                str(uuid.uuid4()), tx["id"],
                it["product_id"], it["price"], it["qty"],
                "generator", tx["date"],
            ))
        
        p = tx["payment"]
        pay_rows.append((
            p["id"], tx["id"], p["amount"], p["type"],
            "generator", tx["date"],
        ))
    
    execute_batch(cur, f"""
        INSERT INTO {SCHEMA}.transactions
            (id, register_id, total, date, customer_id,
             created_by, created_date, last_modified_by, last_modified_date)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (id) DO NOTHING
    """, tx_rows, page_size=500)
    
    execute_batch(cur, f"""
        INSERT INTO {SCHEMA}.transaction_items
            (id, transaction_id, product_id, price, qty,
             created_by, created_date)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (id) DO NOTHING
    """, item_rows, page_size=500)
    
    execute_batch(cur, f"""
        INSERT INTO {SCHEMA}.payments
            (id, transaction_id, amount, type, created_by, created_date)
        VALUES (%s,%s,%s,%s,%s,%s)
        ON CONFLICT (id) DO NOTHING
    """, pay_rows, page_size=500)
    
    conn.commit()
    cur.close()


# ─────────────────────────────────────────────────────────────────────────
# GÉNÉRATION PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "=" * 70)
    print("🚀 GÉNÉRATEUR INTELLIGENT - VERSION 300K")
    print("=" * 70)
    print(f"   📅 Période: {CFG['start_date']} → {CFG['end_date']}")
    print(f"   🎯 Objectif: {CFG['target_transactions']:,} transactions")
    print(f"   💳 Paiements: CASH (65%) / CREDIT (35%)")
    print("=" * 70)
    
    # Connexion
    conn = psycopg2.connect(**DB_CONFIG)
    log.info("✅ Connexion PostgreSQL établie")
    
    # Charger les données existantes
    ref = load_existing_data(conn)
    
    # Calculer le nombre à générer
    to_generate = CFG["target_transactions"] - ref["real_count"]
    
    if to_generate <= 0:
        log.info(f"✅ Objectif déjà atteint !")
        conn.close()
        return
    
    log.info(f"📊 À générer: {to_generate:,} transactions")
    
    # Préparer les dates
    start_date = datetime.strptime(CFG["start_date"], "%Y-%m-%d")
    end_date = datetime.strptime(CFG["end_date"], "%Y-%m-%d")
    total_days = (end_date - start_date).days
    
    all_dates = [start_date + timedelta(days=i) for i in range(total_days + 1)]
    
    # Calculer les boosts quotidiens
    log.info("📊 Calcul des boosts saisonniers...")
    daily_boosts = [daily_multiplier(d) for d in all_dates]
    total_boost = sum(daily_boosts)
    
    # Distribuer les transactions par jour
    transactions_per_day = []
    for boost in daily_boosts:
        n = int(round(to_generate * boost / total_boost))
        transactions_per_day.append(max(1, n))
    
    # Ajustement
    diff = to_generate - sum(transactions_per_day)
    for i in range(abs(diff)):
        transactions_per_day[i % total_days] += 1 if diff > 0 else -1
    
    # Génération
    batch = []
    total_gen = 0
    batch_size = CFG["batch_size"]
    
    log.info(f"🚀 Début génération sur {total_days} jours...")
    
    for day_idx, (date, daily_tx) in enumerate(tqdm(zip(all_dates, transactions_per_day), 
                                                      total=total_days, 
                                                      desc="Génération")):
        boost = daily_boosts[day_idx]
        
        for _ in range(daily_tx):
            tx = generate_transaction(date, ref, boost)
            if tx:
                batch.append(tx)
                total_gen += 1
            
            if len(batch) >= batch_size:
                insert_batch(conn, batch)
                batch = []
                
                if total_gen % 50000 == 0:
                    log.info(f"   ✓ {total_gen:,}/{to_generate:,} générées")
            
            if total_gen >= to_generate:
                break
        
        if total_gen >= to_generate:
            break
    
    # Dernier batch
    if batch:
        insert_batch(conn, batch)
    
    # Vérification finale
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {SCHEMA}.transactions")
    final_count = cur.fetchone()[0]
    cur.close()
    conn.close()
    
    print("\n" + "=" * 70)
    print("🎉 GÉNÉRATION TERMINÉE !")
    print("=" * 70)
    print(f"   📀 Transactions réelles: {ref['real_count']:,}")
    print(f"   🤖 Transactions générées: {total_gen:,}")
    print(f"   ✅ TOTAL FINAL: {final_count:,} / {CFG['target_transactions']:,}")
    print("=" * 70)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    run()