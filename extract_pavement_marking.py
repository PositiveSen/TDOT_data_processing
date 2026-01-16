import pandas as pd

# Load the unique items data
print("Loading unique_items.csv...")
df = pd.read_csv('output/unique_items.csv')

print(f"Total unique items: {len(df):,}")

# Filter for pavement marking items (716- series)
print("\nFiltering for pavement marking items (716-XX.XX)...")
pavement_marking = df[df['item_no'].str.startswith('716', na=False)].copy()

print(f"Pavement marking items: {len(pavement_marking):,}")

# Save to CSV
output_file = 'output/pavement_marking_items.csv'
pavement_marking.to_csv(output_file, index=False)
print(f"\n✓ Saved to: {output_file}")

# Show first 30 items
print("\nFirst 30 pavement marking items:")
print(pavement_marking.head(30).to_string(index=False))
