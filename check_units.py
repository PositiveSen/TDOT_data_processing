#!/usr/bin/env python3
"""
Check unique units of measurement in TDOT data
"""

import pandas as pd

# Load TDOT data
print("Loading TDOT data...")
df = pd.read_csv('Data/TDOT_data.csv', encoding='latin-1')
print(f"Total records: {len(df):,}")

# Get unique units
print("\nExtracting unique units...")
units = df['Units'].dropna().astype(str).str.strip()
unique_units = units.value_counts().sort_values(ascending=False)

print(f"\nFound {len(unique_units)} unique units:")
print("Count | Unit")
print("-" * 40)
for unit, count in unique_units.items():
    print(f"{count:5d} | {unit}")

# Save to file
unique_units.to_csv('unique_units.csv')
print(f"\nSaved to unique_units.csv")