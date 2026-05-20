"""
COMPARAISON DES 5 MODÈLES - Graphiques et tableau
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Données
models = ['Linear', 'Ridge', 'Random Forest', 'XGBoost', 'LightGBM']
r2 = [0.5552, 0.5571, 0.8970, 0.8483, 0.8584]
mape = [4.36, 4.36, 4.95, 3.05, 4.32]
mae = [1258, 1257, 1435, 873, 1303]

colors = ['#95a5a6', '#3498db', '#f39c12', '#2ecc71', '#e74c3c']

# ============================================================================
# TABLEAU
# ============================================================================
df = pd.DataFrame({
    'Modèle': models,
    'R²': [f'{x:.4f}' for x in r2],
    'MAPE (%)': [f'{x:.2f}%' for x in mape],
    'MAE (TND)': [f'{x:.0f}' for x in mae]
})

print("="*70)
print("📊 TABLEAU COMPARATIF DES 5 MODÈLES")
print("="*70)
print(df.to_string(index=False))

# ============================================================================
# GRAPHIQUE 1 : R²
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].bar(models, r2, color=colors, edgecolor='black')
axes[0, 0].set_ylabel('R² Score')
axes[0, 0].set_title('R² (plus haut = meilleur)', fontweight='bold')
axes[0, 0].set_ylim(0.5, 0.95)
for i, v in enumerate(r2):
    axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# ============================================================================
# GRAPHIQUE 2 : MAPE (plus bas = meilleur)
# ============================================================================
axes[0, 1].bar(models, mape, color=colors, edgecolor='black')
axes[0, 1].set_ylabel('MAPE (%)')
axes[0, 1].set_title('MAPE (plus bas = meilleur)', fontweight='bold')
for i, v in enumerate(mape):
    axes[0, 1].text(i, v + 0.1, f'{v:.2f}%', ha='center', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# ============================================================================
# GRAPHIQUE 3 : MAE
# ============================================================================
axes[1, 0].bar(models, mae, color=colors, edgecolor='black')
axes[1, 0].set_ylabel('MAE (TND)')
axes[1, 0].set_title('MAE - Erreur absolue (plus bas = meilleur)', fontweight='bold')
for i, v in enumerate(mae):
    axes[1, 0].text(i, v + 20, f'{v:.0f}', ha='center', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# ============================================================================
# GRAPHIQUE 4 : Radar Chart (meilleurs modèles)
# ============================================================================
# Prendre les 3 meilleurs modèles
best_models = ['XGBoost', 'LightGBM', 'Random Forest']
best_r2 = [r2[3], r2[4], r2[2]]
best_mape = [mape[3], mape[4], mape[2]]
best_mae = [mae[3], mae[4], mae[2]]

# Normaliser (plus haut = meilleur)
r2_norm = [x/0.9 for x in best_r2]
mape_norm = [1 - (x/10) for x in best_mape]
mae_norm = [1 - (x/2000) for x in best_mae]

categories = ['R²', 'MAPE (inversé)', 'MAE (inversé)']
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

colors_best = ['#2ecc71', '#e74c3c', '#f39c12']

for i, (model, color) in enumerate(zip(best_models, colors_best)):
    values = [r2_norm[i], mape_norm[i], mae_norm[i]]
    values += values[:1]
    axes[1, 1].plot(angles, values, 'o-', linewidth=2, label=model, color=color)
    axes[1, 1].fill(angles, values, alpha=0.1, color=color)

axes[1, 1].set_xticks(angles[:-1])
axes[1, 1].set_xticklabels(categories)
axes[1, 1].set_ylim(0, 1)
axes[1, 1].legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
axes[1, 1].set_title('Comparaison multi-critères (meilleurs modèles)', fontweight='bold')

plt.suptitle('Comparaison des 5 Modèles ML - Prédiction des Ventes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('comparaison_modeles.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Graphique sauvegardé: comparaison_modeles.png")

# ============================================================================
# CONCLUSION
# ============================================================================
print("\n" + "="*70)
print("🏆 CONCLUSION")
print("="*70)
print("""
✅ MODÈLE RECOMMANDÉ : XGBOOST

   Paramètres optimaux:
   • n_estimators: 300
   • learning_rate: 0.05
   • max_depth: 3
   • subsample: 0.9
   • colsample_bytree: 0.9
   • reg_alpha: 0.5
   • reg_lambda: 1

   Performances:
   • MAPE = 3.05% (meilleure précision)
   • MAE = 873 TND (erreur la plus faible)
   • R² = 0.8483 (très bon)

   → XGBoost est recommandé pour la production
""")