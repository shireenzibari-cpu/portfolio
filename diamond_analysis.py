"""
Diamond Pricing Analysis
Author: Shireen Zibari

Explores which physical characteristics (carat, cut, color, clarity) most
strongly influence diamond prices, using a public dataset of 53,940 diamonds.

Libraries: pandas, matplotlib, seaborn
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# ===== Load data =====
df = sns.load_dataset('diamonds')

print(f"Dataset shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")

# ===== Color palette (kept consistent with portfolio site design) =====
ACCENT = '#2F6F6B'
palette = ['#2F6F6B', '#4A6FA5', '#C1502E', '#B8860B', '#8B5A8C']

# ===== 1. Price distribution =====
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df['price'], bins=50, color=ACCENT, alpha=0.85)
ax.set_xlabel('Price (USD)')
ax.set_ylabel('Number of Diamonds')
ax.set_title('Distribution of Diamond Prices')
plt.tight_layout()
plt.savefig('charts/price_distribution.png', dpi=150)
plt.close()

# ===== 2. Price vs. carat =====
fig, ax = plt.subplots(figsize=(7, 4))
sample = df.sample(3000, random_state=42)
ax.scatter(sample['carat'], sample['price'], alpha=0.3, s=10, color=ACCENT)
ax.set_xlabel('Carat')
ax.set_ylabel('Price (USD)')
ax.set_title('Price vs. Carat Weight')
plt.tight_layout()
plt.savefig('charts/price_vs_carat.png', dpi=150)
plt.close()

# ===== 3. Average price by cut quality =====
cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
avg_by_cut = df.groupby('cut', observed=True)['price'].mean().reindex(cut_order)

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(avg_by_cut.index, avg_by_cut.values, color=palette[:5])
ax.set_ylabel('Average Price (USD)')
ax.set_title('Average Price by Cut Quality')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 50,
             f'${height:,.0f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('charts/price_by_cut.png', dpi=150)
plt.close()

# ===== 4. Average price by color grade =====
color_order = sorted(df['color'].unique())
avg_by_color = df.groupby('color', observed=True)['price'].mean().reindex(color_order)

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(avg_by_color.index, avg_by_color.values, color=palette[1])
ax.set_xlabel('Color Grade (D = best, J = lowest in this dataset)')
ax.set_ylabel('Average Price (USD)')
ax.set_title('Average Price by Color Grade')
plt.tight_layout()
plt.savefig('charts/price_by_color.png', dpi=150)
plt.close()

# ===== 5. Correlation heatmap =====
numeric_cols = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)
ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
ax.set_yticklabels(numeric_cols)
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8)
ax.set_title('Correlation Between Numeric Features')
plt.tight_layout()
plt.savefig('charts/correlation_heatmap.png', dpi=150)
plt.close()

# ===== Summary stats =====
print(f"\nAverage price: ${df['price'].mean():,.2f}")
print(f"Median price: ${df['price'].median():,.2f}")
print(f"Price-carat correlation: {df['price'].corr(df['carat']):.2f}")
print(f"\nAverage price by cut:\n{avg_by_cut}")
print(f"\nAverage price by color:\n{avg_by_color}")
