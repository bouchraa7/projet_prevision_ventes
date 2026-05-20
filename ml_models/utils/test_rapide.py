import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ml_models.data_preparation import load_daily_data, FEATURES
import pandas as pd

# Charger les données
df = load_daily_data()

# Calculer la corrélation
corr = df[FEATURES].corr()

# Afficher les corrélations avec qty_sold
print("\n" + "="*60)
print("🔍 CORRÉLATIONS AVEC qty_sold")
print("="*60)
qty_corr = corr['qty_sold'].sort_values(ascending=False)
print(qty_corr)

# Vérifier les fortes corrélations (> 0.8)
print("\n" + "="*60)
print("⚠️ FORTES CORRÉLATIONS (> 0.8)")
print("="*60)
high_corr = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.8:
            high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))

if high_corr:
    for f1, f2, val in high_corr:
        print(f"   {f1} <-> {f2}: {val:.3f}")
else:
    print("   ✅ Aucune corrélation forte détectée")

print("\n" + "="*60)
print("💡 CONCLUSION:")
print("="*60)
if high_corr:
    print("   ⚠️ Des corrélations fortes existent → Ridge recommandé")
else:
    print("   ✅ Pas de corrélations fortes → Linear Regression suffit")