# config/settings.py
"""Configuration centrale - Version finale 300K"""

import os
from dotenv import load_dotenv
load_dotenv()

# ── PostgreSQL ────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("PG_HOST",     "localhost"),
    "port":     int(os.getenv("PG_PORT", "5432")),
    "database": os.getenv("PG_DB",       "caissatndb"),
    "user":     os.getenv("PG_USER",     "postgres"),
    "password": os.getenv("PG_PASS",     "Bouchra1234"),
}
POS_SCHEMA = os.getenv("POS_SCHEMA", "s5831082f95ef4a1eac9a6a8c484faf0a")

# ── Générateur ────────────────────────────────────────────────────────────
TARGET_TOTAL = 300_000
GEN_START = "2024-01-01"
GEN_END = "2026-04-30"
BATCH_SIZE = 1000
CANCELLATION_RATE = 0.025
OUTLIER_RATE = 0.018
GROWTH_RATE = 0.15

# Configuration pour le générateur
GENERATOR_CONFIG = {
    "start_date": GEN_START,
    "end_date": GEN_END,
    "target_transactions": TARGET_TOTAL,
    "batch_size": BATCH_SIZE,
    "cancellation_rate": CANCELLATION_RATE,
    "outlier_rate": OUTLIER_RATE,
    "growth_rate_annual": GROWTH_RATE,
}

# Événements saisonniers
SEASONAL_EVENTS = [
    ("ramadan",          3, 10,  4,  9, 1.45),
    ("eid_al_fitr",      4, 10,  4, 14, 1.90),
    ("eid_al_adha",      6, 20,  6, 24, 1.70),
    ("rentree_scolaire", 9,  1,  9, 15, 1.35),
    ("nouvel_an",       12, 25,  1,  5, 1.30),
    ("ete_peak",         7,  1,  8, 31, 1.20),
    ("soldes_hiver",     1, 15,  2, 15, 1.18),
]

RUSH_HOURS = {
    8: 1.15, 9: 1.40, 10: 1.30, 12: 1.65, 13: 1.55,
    17: 1.80, 18: 2.00, 19: 1.85, 20: 1.50,
}

# ── ML / DL ───────────────────────────────────────────────────────────────
ML_CONFIG = {
    "forecast_horizons": [7, 14, 30],
    "train_ratio": 0.75,
    "val_ratio": 0.10,
    "test_ratio": 0.15,
    "random_seed": 42,
}

DL_CONFIG = {
    "sequence_length": 30,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "batch_size": 32,
    "epochs": 60,
    "learning_rate": 1e-3,
    "patience": 12,
    "device": "cpu",
}

STOCK_CRITICAL_DAYS = 3
STOCK_WARNING_DAYS = 7

LLM_CONFIG = {
    "provider": os.getenv("LLM_PROVIDER", "openai"),
    "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
    "embedding_model": "text-embedding-3-small",
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "temperature": 0.1,
}

SUPERSET_CONFIG = {
    "url": os.getenv("SUPERSET_URL", "http://localhost:8088"),
    "user": os.getenv("SUPERSET_USER", "admin"),
    "password": os.getenv("SUPERSET_PASS", "admin"),
}