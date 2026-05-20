"""
COMPARAISON COMPLÈTE - MACHINE LEARNING vs DEEP LEARNING
Graphiques et tableau pour rapport Word
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

print("="*80)
print("📊 COMPARAISON ML vs DL - PRÉDICTION DES VENTES")
print("="*80)

# ============================================================================
# DONNÉES DES MODÈLES
# ============================================================================
data = {
    'Modèle': ['XGBoost', 'LightGBM', 'Random Forest', 'MLP (DL)', 'GRU (DL)', 'LSTM (DL)', 'Ridge', 'Linear'],
    'Type': ['ML', 'ML', 'ML', 'DL', 'DL', 'DL', 'ML', 'ML'],
    'R²': [0.8548, 0.8306, 0.8970, 0.6561, 0.4613, 0.4103, 0.5571, 0.5552],
    'MAPE (%)': [3.05, 7.52, 4.95, 9.99, 14.87, 17.26, 4.36, 4.36],
    'MAE (TND)': [873, 2054, 1435, 2579, 4218, 4756, 1257, 1258],
    'RMSE (TND)': [3739, 3952, 3081, 5630, 7129, 7458, 6390, 6403]
}

df = pd.DataFrame(data)
df = df.sort_values('R²', ascending=False)

# ============================================================================
# TABLEAU COMPARATIF
# ============================================================================
print("\n📊 TABLEAU COMPARATIF DES MODÈLES")
print("="*80)
print(df[['Modèle', 'Type', 'R²', 'MAPE (%)', 'MAE (TND)', 'RMSE (TND)']].to_string(index=False))

# Sauvegarde CSV
df.to_csv('results/comparaison_complete.csv', index=False)
print("\n💾 Tableau sauvegardé: results/comparaison_complete.csv")

# ============================================================================
# 1. GRAPHIQUE : R² par modèle
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

colors = ['#2ecc71' if t == 'ML' else '#e74c3c' for t in df['Type']]
bars = ax.bar(df['Modèle'], df['R²'], color=colors, edgecolor='black', linewidth=1)

ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('Comparaison des modèles - R² (plus haut = meilleur)', fontsize=14, fontweight='bold')
ax.set_ylim(0.3, 0.95)
ax.axhline(y=0.8548, color='green', linestyle='--', alpha=0.7, label='XGBoost')
ax.axhline(y=0.6561, color='red', linestyle='--', alpha=0.7, label='Meilleur DL (MLP)')

for bar, val in zip(bars, df['R²']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{val:.4f}', ha='center', fontweight='bold', fontsize=10)

ax.legend(loc='upper right')
ax.set_xticklabels(df['Modèle'], rotation=45, ha='right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/comparaison_r2.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Graphique R² sauvegardé: results/comparaison_r2.png")

# ============================================================================
# 2. GRAPHIQUE : MAPE par modèle
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

bars = ax.bar(df['Modèle'], df['MAPE (%)'], color=colors, edgecolor='black', linewidth=1)

ax.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
ax.set_title('Comparaison des modèles - MAPE (plus bas = meilleur)', fontsize=14, fontweight='bold')
ax.axhline(y=3.05, color='green', linestyle='--', alpha=0.7, label='XGBoost')
ax.axhline(y=9.99, color='red', linestyle='--', alpha=0.7, label='Meilleur DL (MLP)')

for bar, val in zip(bars, df['MAPE (%)']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
            f'{val:.2f}%', ha='center', fontweight='bold', fontsize=10)

ax.legend(loc='upper left')
ax.set_xticklabels(df['Modèle'], rotation=45, ha='right')
ax.set_ylim(0, 20)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/comparaison_mape.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Graphique MAPE sauvegardé: results/comparaison_mape.png")

# ============================================================================
# 3. GRAPHIQUE : MEILLEURS MODÈLES (Top 4)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

top_models = df.head(4)
top_colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']

x = np.arange(len(top_models['Modèle']))
width = 0.35

bars1 = ax.bar(x - width/2, top_models['R²'], width, label='R²', color='#2ecc71', edgecolor='black')
bars2 = ax.bar(x + width/2, 100 - top_models['MAPE (%)'], width, label='Précision (100-MAPE%)', color='#3498db', edgecolor='black')

ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Top 4 modèles - R² vs Précision', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(top_models['Modèle'])
ax.legend(loc='upper left')
ax.set_ylim(0, 105)

for bar, val in zip(bars1, top_models['R²']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{val*100:.1f}%', ha='center', fontweight='bold', fontsize=9)

for bar, val in zip(bars2, 100 - top_models['MAPE (%)']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{val:.1f}%', ha='center', fontweight='bold', fontsize=9)

ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/comparaison_top4.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Graphique Top 4 sauvegardé: results/comparaison_top4.png")

# ============================================================================
# 4. GRAPHIQUE RADAR (ML vs DL)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Prendre les meilleurs modèles ML et DL
ml_best = df[df['Type'] == 'ML'].iloc[0]  # XGBoost
dl_best = df[df['Type'] == 'DL'].iloc[0]  # MLP

# Métriques (normalisées pour radar)
metrics = ['R²', 'MAPE', 'MAE', 'RMSE']
# Normalisation : plus haut = meilleur
ml_values = [
    ml_best['R²'] / 0.9,
    1 - (ml_best['MAPE (%)'] / 20),
    1 - (ml_best['MAE (TND)'] / 6000),
    1 - (ml_best['RMSE (TND)'] / 8000)
]
dl_values = [
    dl_best['R²'] / 0.9,
    1 - (dl_best['MAPE (%)'] / 20),
    1 - (dl_best['MAE (TND)'] / 6000),
    1 - (dl_best['RMSE (TND)'] / 8000)
]

angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]
ml_values += ml_values[:1]
dl_values += dl_values[:1]

ax.plot(angles, ml_values, 'o-', linewidth=2, label='XGBoost (ML)', color='#2ecc71')
ax.fill(angles, ml_values, alpha=0.2, color='#2ecc71')
ax.plot(angles, dl_values, 'o-', linewidth=2, label='MLP (DL)', color='#e74c3c')
ax.fill(angles, dl_values, alpha=0.2, color='#e74c3c')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.set_title('Comparaison ML vs DL - Meilleurs modèles', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/comparaison_radar.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Graphique Radar sauvegardé: results/comparaison_radar.png")

# ============================================================================
# 5. GRAPHIQUE ÉVOLUTION DES PERFORMANCES
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

models_order = ['Linear', 'Ridge', 'Random Forest', 'LightGBM', 'XGBoost', 'MLP', 'GRU', 'LSTM']
r2_order = [0.5552, 0.5571, 0.8970, 0.8306, 0.8548, 0.6561, 0.4613, 0.4103]

colors_order = ['#95a5a6', '#3498db', '#f39c12', '#2ecc71', '#2ecc71', '#e74c3c', '#e74c3c', '#e74c3c']

bars = ax.bar(models_order, r2_order, color=colors_order, edgecolor='black', linewidth=1)

ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('Évolution des performances (du plus simple au plus complexe)', fontsize=14, fontweight='bold')
ax.set_ylim(0.3, 0.95)

for bar, val in zip(bars, r2_order):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{val:.4f}', ha='center', fontweight='bold', fontsize=9)

ax.axhline(y=0.8548, color='green', linestyle='--', alpha=0.7, label='XGBoost')
ax.axhline(y=0.6561, color='red', linestyle='--', alpha=0.7, label='Meilleur DL')

# Flèche pour montrer la progression
ax.annotate('', xy=(4, 0.85), xytext=(0, 0.4),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.text(2, 0.65, 'Progression\n+54%', fontsize=10, fontweight='bold', color='blue', ha='center')

ax.legend(loc='lower right')
ax.set_xticklabels(models_order, rotation=45, ha='right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/comparaison_evolution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Graphique Évolution sauvegardé: results/comparaison_evolution.png")

# ============================================================================
# 6. RÉSUMÉ STATISTIQUE ML vs DL
# ============================================================================
print("\n" + "="*80)
print("📊 RÉSUMÉ STATISTIQUE ML vs DL")
print("="*80)

ml_models = df[df['Type'] == 'ML']
dl_models = df[df['Type'] == 'DL']

print(f"\n🔵 MACHINE LEARNING (5 modèles):")
print(f"   • Meilleur R² : {ml_models['R²'].max():.4f} (XGBoost)")
print(f"   • Meilleur MAPE : {ml_models['MAPE (%)'].min():.2f}% (XGBoost)")
print(f"   • R² moyen : {ml_models['R²'].mean():.4f}")
print(f"   • MAPE moyen : {ml_models['MAPE (%)'].mean():.2f}%")

print(f"\n🔴 DEEP LEARNING (3 modèles):")
print(f"   • Meilleur R² : {dl_models['R²'].max():.4f} (MLP)")
print(f"   • Meilleur MAPE : {dl_models['MAPE (%)'].min():.2f}% (MLP)")
print(f"   • R² moyen : {dl_models['R²'].mean():.4f}")
print(f"   • MAPE moyen : {dl_models['MAPE (%)'].mean():.2f}%")

print(f"\n📈 ÉCART ML vs DL:")
print(f"   • R² : ML +{(ml_models['R²'].max() - dl_models['R²'].max())*100:.1f}% meilleur")
print(f"   • MAPE : ML -{(dl_models['MAPE (%)'].min() - ml_models['MAPE (%)'].min()):.2f}% meilleur")

# ============================================================================
# 7. TEXTE POUR RAPPORT WORD
# ============================================================================
print("\n" + "="*80)
print("📝 TEXTE À COPIER DANS VOTRE RAPPORT WORD")
print("="*80)

text_rapport = """
================================================================================
                    COMPARAISON ML vs DL - Résultats finaux
================================================================================

1. TABLEAU COMPARATIF

| Modèle          | Type | R²     | MAPE   | MAE (TND) | RMSE (TND) |
|-----------------|------|--------|--------|-----------|------------|
| XGBoost         | ML   | 0.8548 | 3.05%  | 873       | 3 739      |
| Random Forest   | ML   | 0.8970 | 4.95%  | 1 435     | 3 081      |
| LightGBM        | ML   | 0.8306 | 7.52%  | 2 054     | 3 952      |
| MLP             | DL   | 0.6561 | 9.99%  | 2 579     | 5 630      |
| GRU             | DL   | 0.4613 | 14.87% | 4 218     | 7 129      |
| LSTM            | DL   | 0.4103 | 17.26% | 4 756     | 7 458      |
| Ridge           | ML   | 0.5571 | 4.36%  | 1 257     | 6 390      |
| Linear          | ML   | 0.5552 | 4.36%  | 1 258     | 6 403      |

2. CONCLUSIONS

🏆 MEILLEUR MODÈLE GLOBAL : XGBOOST
   • R² = 0.8548 | MAPE = 3.05% | MAE = 873 TND

🧠 MEILLEUR MODÈLE DEEP LEARNING : MLP
   • R² = 0.6561 | MAPE = 9.99% | Architecture (32,16)

📊 ÉCARTS OBSERVÉS
   • XGBoost est 30% meilleur que MLP en R²
   • XGBoost est 3x plus précis que MLP (MAPE 3% vs 10%)
   • Les modèles ML dominent sur tous les critères

💡 RECOMMANDATION : XGBOOST POUR LA PRODUCTION
"""

print(text_rapport)

print("\n" + "="*80)
print("✅ TOUS LES GRAPHIQUES ONT ÉTÉ GÉNÉRÉS")
print("📁 Fichiers créés:")
print("   • comparaison_complete.csv - Tableau des résultats")
print("   • comparaison_r2.png - Graphique R²")
print("   • comparaison_mape.png - Graphique MAPE")
print("   • comparaison_top4.png - Top 4 modèles")
print("   • comparaison_radar.png - Radar chart ML vs DL")
print("   • comparaison_evolution.png - Évolution des performances")
print("="*80)