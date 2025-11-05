#!/usr/bin/env python
"""Show detailed X-axis label mapping for the bar chart"""
import pandas as pd
import json

csv_file = "output_with_xlabels/8a22e3e7d8519fbc90bdcfc1722c1b82_c3RhdGxpbmtzLm9lY2Rjb2RlLm9yZwk5Mi4yNDMuMjMuMTM3.XLS-0-1_table.csv"

df = pd.read_csv(csv_file)

# Sort by X position
df_sorted = df.sort_values('x1').reset_index(drop=True)

print("\n" + "=" * 110)
print("X-AXIS LABEL MAPPING DETAILS")
print("=" * 110)
print(f"\n{'Bar':<6} {'X-Center':<12} {'Category':<20} {'Legend':<15} {'Value':<12}")
print("-" * 110)

for idx, row in df_sorted.iterrows():
    bar_num = idx + 1
    x_center = (row['x1'] + row['x2']) / 2
    category = row['category'] if pd.notna(row['category']) and row['category'] else '(none)'
    label = row['label'] if pd.notna(row['label']) and row['label'] else '(none)'
    value = row['value']
    
    print(f"{bar_num:<6} {x_center:<12.2f} {category:<20} {label:<15} {value:<12.2f}")

print("=" * 110)

# Group by category
if df['category'].notna().any():
    print("\nBars grouped by X-axis category:")
    print("-" * 110)
    for cat in df_sorted['category'].unique():
        if pd.notna(cat) and cat:
            cat_bars = df_sorted[df_sorted['category'] == cat]
            values = cat_bars['value'].values
            print(f"\n{cat}: {len(cat_bars)} bars")
            print(f"  Values: {', '.join([f'{v:.2f}' for v in values])}")
            print(f"  Average: {cat_bars['value'].mean():.2f}")
