# etl/pipeline.py
"""
Pipeline ETL complet.

Étapes :
  1. Extraction depuis PostgreSQL (transactions + items + produits + paiements)
  2. Nettoyage & validation (doublons, outliers, valeurs manquantes)
  3. Feature Engineering (temporel, événements, lags, rolling, KPI)
  4. Chargement dans la table analytique `pos_analytics`
     → Source unique pour ML, DL, Séries temporelles et Superset
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import DB_CONFIG, POS_SCHEMA, SEASONAL_EVENTS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
S = POS_SCHEMA


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _season(month: int) -> str:
    if month in [12, 1, 2]: return "hiver"
    if month in [3, 4, 5]:  return "printemps"
    if month in [6, 7, 8]:  return "ete"
    return "automne"


def _event(month: int, day: int) -> str:
    for label, m1, d1, m2, d2, _ in SEASONAL_EVENTS:
        if (month > m1 or (month == m1 and day >= d1)) and \
           (month < m2 or (month == m2 and day <= d2)):
            return label
    return "normal"


# ══════════════════════════════════════════════════════════════════════════
# 1. EXTRACTION
# ══════════════════════════════════════════════════════════════════════════

def extract_sales(conn) -> pd.DataFrame:
    sql = f"""
        SELECT
            t.id                                                 AS transaction_id,
            t.date                                               AS transaction_date,
            t.total                                              AS transaction_total,
            t.customer_id,
            t.refund_id,
            ti.id                                                AS item_id,
            ti.product_id,
            p.name                                               AS product_name,
            p.sale_unit_price                                    AS unit_price,
            ti.price                                             AS sold_price,
            ti.qty,
            (ti.qty * ti.price)                                  AS line_total,
            COALESCE(p.purchase_unit_price, ti.price * 0.60)     AS cost_price,
            COALESCE(p.stock_qty, 0)                             AS current_stock,
            COALESCE(p.stock_min_limit, 0)                       AS stock_min_limit,
            COALESCE(p.stock_max_limit, 0)                       AS stock_max_limit,
            COALESCE(pay.type, 'INCONNU')                        AS payment_type,
            COALESCE(pay.amount, t.total)                        AS payment_amount,
            COALESCE(b.name, 'Sans marque')                      AS brand_name
        FROM {S}.transactions t
        JOIN {S}.transaction_items ti ON ti.transaction_id = t.id
        JOIN {S}.products p           ON p.id = ti.product_id
        LEFT JOIN {S}.payments pay    ON pay.transaction_id = t.id
        LEFT JOIN {S}.brands b        ON b.id = p.brand_id
        ORDER BY t.date
    """
    df = pd.read_sql(sql, conn)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    log.info(f"[ETL] Extraction : {len(df):,} lignes | "
             f"{df['transaction_id'].nunique():,} transactions | "
             f"{df['product_id'].nunique()} produits")
    return df


def extract_products(conn) -> pd.DataFrame:
    return pd.read_sql(f"""
        SELECT p.id, p.name, p.sale_unit_price, p.purchase_unit_price,
               COALESCE(p.stock_qty,0) AS stock_qty,
               COALESCE(p.stock_min_limit,0) AS stock_min_limit,
               COALESCE(p.stock_max_limit,0) AS stock_max_limit,
               COALESCE(b.name,'Sans marque') AS brand_name
        FROM {S}.products p
        LEFT JOIN {S}.brands b ON b.id = p.brand_id
    """, conn)


# ══════════════════════════════════════════════════════════════════════════
# 2. NETTOYAGE
# ══════════════════════════════════════════════════════════════════════════

def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    df = df.drop_duplicates(subset=["item_id"])
    df = df.dropna(subset=["transaction_date", "product_id", "qty", "sold_price"])
    df = df[(df["qty"] > 0) & (df["sold_price"].abs() > 0)]

    # Outliers IQR ×3 (sur les ventes positives seulement)
    sales_mask = df["line_total"] > 0
    q1 = df.loc[sales_mask, "line_total"].quantile(0.005)
    q3 = df.loc[sales_mask, "line_total"].quantile(0.995)
    iqr = q3 - q1
    ok  = (~sales_mask) | ((df["line_total"] >= q1 - 3*iqr) &
                            (df["line_total"] <= q3 + 3*iqr))
    df  = df[ok]

    df["payment_type"] = df["payment_type"].str.upper().fillna("INCONNU")
    df["line_total"]   = (df["sold_price"] * df["qty"]).round(3)
    df["cost_price"]   = df["cost_price"].fillna(df["sold_price"].abs() * 0.60)
    df["gross_margin"] = ((df["sold_price"].abs() - df["cost_price"]) * df["qty"]).round(3)
    df["is_refund"]    = df["refund_id"].notna()

    log.info(f"[ETL] Nettoyage : {before:,} → {len(df):,} "
             f"(supprimés : {before - len(df):,})")
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["transaction_date"]
    df = df.copy()
    df["year"]           = dt.dt.year
    df["month"]          = dt.dt.month
    df["day"]            = dt.dt.day
    df["hour"]           = dt.dt.hour
    df["day_of_week"]    = dt.dt.dayofweek
    df["week_of_year"]   = dt.dt.isocalendar().week.astype(int)
    df["quarter"]        = dt.dt.quarter
    df["is_weekend"]     = (dt.dt.dayofweek >= 4).astype(int)
    df["is_month_start"] = (dt.dt.day <= 5).astype(int)
    df["is_month_end"]   = (dt.dt.day >= 25).astype(int)
    df["season"]         = dt.dt.month.map(_season)
    df["special_event"]  = [_event(m, d) for m, d in zip(dt.dt.month, dt.dt.day)]
    EVENTS = ["ramadan","eid_al_fitr","eid_al_adha","rentree_scolaire",
              "nouvel_an","ete_peak","soldes_hiver"]
    for ev in EVENTS:
        df[f"is_{ev}"] = (df["special_event"] == ev).astype(int)
    return df


def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Série journalière agrégée — source principale pour ML/DL/TS.
    Revenue en colonne 0 (convention Deep Learning).
    """
    sales = df[~df["is_refund"]].copy()
    sales["date"] = sales["transaction_date"].dt.normalize()

    agg = (
        sales.groupby("date")
        .agg(
            nb_transactions =("transaction_id",  "nunique"),
            nb_items        =("item_id",          "count"),
            revenue         =("line_total",        "sum"),
            qty_sold        =("qty",               "sum"),
            gross_margin    =("gross_margin",       "sum"),
            avg_basket      =("transaction_total",
                              lambda x: x.drop_duplicates().mean()),
            nb_products     =("product_id",         "nunique"),
        )
        .reset_index()
    )
    # Jours manquants → 0
    full = pd.date_range(agg["date"].min(), agg["date"].max(), freq="D")
    agg  = agg.set_index("date").reindex(full, fill_value=0).reset_index()
    agg.rename(columns={"index": "date"}, inplace=True)
    agg["date"] = pd.to_datetime(agg["date"])

    # Features temporelles
    dt = agg["date"]
    agg["year"]           = dt.dt.year
    agg["month"]          = dt.dt.month
    agg["day"]            = dt.dt.day
    agg["day_of_week"]    = dt.dt.dayofweek
    agg["week_of_year"]   = dt.dt.isocalendar().week.astype(int)
    agg["quarter"]        = dt.dt.quarter
    agg["is_weekend"]     = (dt.dt.dayofweek >= 4).astype(int)
    agg["is_month_start"] = (dt.dt.day <= 5).astype(int)
    agg["is_month_end"]   = (dt.dt.day >= 25).astype(int)
    agg["season"]         = dt.dt.month.map(_season)
    agg["special_event"]  = [_event(m, d) for m, d in zip(dt.dt.month, dt.dt.day)]
    for ev in ["ramadan","eid_al_fitr","eid_al_adha","rentree_scolaire",
               "nouvel_an","ete_peak","soldes_hiver"]:
        agg[f"is_{ev}"] = (agg["special_event"] == ev).astype(int)

    # Lags (variables autorégressives)
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        agg[f"rev_lag_{lag}"] = agg["revenue"].shift(lag)
        agg[f"qty_lag_{lag}"] = agg["qty_sold"].shift(lag)

    # Rolling statistics
    for w in [7, 14, 30]:
        agg[f"rev_roll_mean_{w}"] = agg["revenue"].rolling(w, min_periods=1).mean()
        agg[f"rev_roll_std_{w}"]  = agg["revenue"].rolling(w, min_periods=1).std().fillna(0)
        agg[f"qty_roll_mean_{w}"] = agg["qty_sold"].rolling(w, min_periods=1).mean()

    # Variation hebdomadaire
    agg["rev_pct_7d"] = agg["revenue"].pct_change(7).replace([np.inf,-np.inf], 0).fillna(0)

    # Vitesse de vente (consommation moyenne sur 30j)
    agg["sales_velocity_30d"] = agg["qty_sold"].rolling(30, min_periods=1).mean()

    log.info(f"[ETL] Série journalière : {len(agg)} jours | "
             f"revenue total = {agg['revenue'].sum():,.0f}")
    return agg


# ══════════════════════════════════════════════════════════════════════════
# 4. TABLE ANALYTIQUE
# ══════════════════════════════════════════════════════════════════════════

DDL_ANALYTICS = f"""
CREATE TABLE IF NOT EXISTS {S}.pos_analytics (
    id                  SERIAL PRIMARY KEY,
    transaction_id      VARCHAR(36),
    transaction_date    TIMESTAMP,
    item_id             VARCHAR(36),
    product_id          VARCHAR(36),
    product_name        VARCHAR(255),
    brand_name          VARCHAR(255),
    customer_id         VARCHAR(36),
    payment_type        VARCHAR(30),
    qty                 INTEGER,
    sold_price          NUMERIC(10,3),
    cost_price          NUMERIC(10,3),
    line_total          NUMERIC(12,3),
    gross_margin        NUMERIC(12,3),
    transaction_total   NUMERIC(12,3),
    current_stock       INTEGER,
    stock_min_limit     INTEGER,
    stock_max_limit     INTEGER,
    is_refund           BOOLEAN,
    year                SMALLINT,
    month               SMALLINT,
    day                 SMALLINT,
    hour                SMALLINT,
    day_of_week         SMALLINT,
    week_of_year        SMALLINT,
    quarter             SMALLINT,
    is_weekend          SMALLINT,
    is_month_start      SMALLINT,
    is_month_end        SMALLINT,
    season              VARCHAR(20),
    special_event       VARCHAR(50),
    is_ramadan          SMALLINT DEFAULT 0,
    is_eid_al_fitr      SMALLINT DEFAULT 0,
    is_eid_al_adha      SMALLINT DEFAULT 0,
    is_rentree_scolaire SMALLINT DEFAULT 0,
    is_nouvel_an        SMALLINT DEFAULT 0,
    is_ete_peak         SMALLINT DEFAULT 0,
    is_soldes_hiver     SMALLINT DEFAULT 0,
    created_at          TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_pa_date    ON {S}.pos_analytics(transaction_date);
CREATE INDEX IF NOT EXISTS idx_pa_product ON {S}.pos_analytics(product_id);
CREATE INDEX IF NOT EXISTS idx_pa_event   ON {S}.pos_analytics(special_event);
CREATE INDEX IF NOT EXISTS idx_pa_refund  ON {S}.pos_analytics(is_refund);
"""

LOAD_COLS = [
    "transaction_id","transaction_date","item_id","product_id",
    "product_name","brand_name","customer_id","payment_type",
    "qty","sold_price","cost_price","line_total","gross_margin",
    "transaction_total","current_stock","stock_min_limit","stock_max_limit",
    "is_refund","year","month","day","hour","day_of_week","week_of_year",
    "quarter","is_weekend","is_month_start","is_month_end","season",
    "special_event","is_ramadan","is_eid_al_fitr","is_eid_al_adha",
    "is_rentree_scolaire","is_nouvel_an","is_ete_peak","is_soldes_hiver",
]


def load_analytics(conn, df: pd.DataFrame):
    cur = conn.cursor()
    cur.execute(DDL_ANALYTICS)
    cur.execute(f"TRUNCATE TABLE {S}.pos_analytics RESTART IDENTITY")
    conn.commit()

    cols = [c for c in LOAD_COLS if c in df.columns]
    df2  = df[cols].copy()

    # Forcer les types Python natifs
    for c in df2.select_dtypes(include=["int64","int32","int16","uint8"]).columns:
        df2[c] = df2[c].astype(int)
    for c in df2.select_dtypes(include=["float64","float32"]).columns:
        df2[c] = df2[c].apply(lambda x: float(x) if pd.notna(x) else None)
    df2["is_refund"] = df2["is_refund"].astype(bool)

    rows   = [tuple(r) for r in df2.itertuples(index=False)]
    ph     = ",".join(["%s"] * len(cols))
    execute_batch(
        cur,
        f"INSERT INTO {S}.pos_analytics ({','.join(cols)}) VALUES ({ph})",
        rows, page_size=500
    )
    conn.commit()
    cur.close()
    log.info(f"[ETL] pos_analytics chargée : {len(rows):,} lignes")


# ══════════════════════════════════════════════════════════════════════════
# 5. KPI
# ══════════════════════════════════════════════════════════════════════════

def compute_kpis(df: pd.DataFrame, products: pd.DataFrame) -> Dict:
    sales = df[~df["is_refund"]]
    total_rev  = sales["line_total"].sum()
    total_marg = sales["gross_margin"].sum()
    nb_tx      = sales["transaction_id"].nunique()
    avg_basket = sales.groupby("transaction_id")["line_total"].sum().mean()
    refund_pct = df["is_refund"].mean() * 100

    top10 = (
        sales.groupby("product_name")["line_total"]
        .sum().nlargest(10).reset_index()
        .rename(columns={"line_total": "revenue"})
    )
    pay_mix    = sales["payment_type"].value_counts(normalize=True).mul(100).round(1)
    by_event   = sales.groupby("special_event")["line_total"].sum().sort_values(ascending=False)
    by_season  = sales.groupby("season")["line_total"].mean().sort_values(ascending=False)

    # Vélocité → alertes stock
    days_count = max(1, sales["transaction_date"].dt.date.nunique())
    vel = (
        sales.groupby("product_id")["qty"].sum()
        .div(days_count)
        .reset_index().rename(columns={"qty": "avg_daily_qty"})
    )
    stock_df = products.merge(vel, left_on="id", right_on="product_id", how="left")
    stock_df["avg_daily_qty"]    = stock_df["avg_daily_qty"].fillna(0.01).clip(lower=0.01)
    stock_df["days_to_stockout"] = stock_df["stock_qty"] / stock_df["avg_daily_qty"]
    stock_df["rupture_score"]    = np.clip(1 - stock_df["days_to_stockout"] / 30, 0, 1)
    alerts = stock_df[stock_df["days_to_stockout"] < 14].sort_values("days_to_stockout")

    return {
        "total_revenue":    round(total_rev, 2),
        "total_margin":     round(total_marg, 2),
        "margin_pct":       round(total_marg / total_rev * 100, 2) if total_rev else 0,
        "nb_transactions":  nb_tx,
        "avg_basket":       round(avg_basket, 2),
        "refund_rate_pct":  round(refund_pct, 2),
        "top10_products":   top10,
        "payment_mix":      pay_mix,
        "revenue_by_event": by_event,
        "revenue_by_season":by_season,
        "stock_alerts":     alerts,
    }


# ══════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════

def run() -> Dict:
    log.info("═══ Pipeline ETL ═══")
    conn     = psycopg2.connect(**DB_CONFIG)
    df_raw   = extract_sales(conn)
    products = extract_products(conn)

    df_clean = clean(df_raw)
    df_feat  = add_features(df_clean)
    daily    = build_daily(df_feat)
    kpis     = compute_kpis(df_feat, products)

    load_analytics(conn, df_feat)
    conn.close()

    log.info("═══ ETL terminé ═══")
    return {"df": df_feat, "daily": daily, "products": products, "kpis": kpis}


if __name__ == "__main__":
    data = run()
    print("\n📊 Aperçu série journalière :")
    print(data["daily"].tail(7)[["date","revenue","nb_transactions","special_event"]].to_string())
    print("\n💰 KPI :")
    for k, v in data["kpis"].items():
        if not isinstance(v, (pd.DataFrame, pd.Series)):
            print(f"  {k}: {v}")
