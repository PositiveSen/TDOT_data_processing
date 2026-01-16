import pandas as pd

def expand_abbreviations(desc):
    """Expand common abbreviations in pavement marking descriptions"""
    # Create mapping of abbreviations to full words
    abbreviations = {
        'PVMT': 'PAVEMENT',
        'MRKNG': 'MARKING',
        'THERMO': 'THERMOPLASTIC',
        'PLSTC': 'PLASTIC',
        'PREFRMD': 'PREFORMED',
        'MRKING': 'MARKING',
        'BI-DIR': 'BI-DIRECTIONAL',
        'MONO-DIR': 'MONO-DIRECTIONAL',
        'W/': 'WITH ',
        'TEMP': 'TEMPORARY',
        'REFL': 'REFLECTIVE',
        'SYMB': 'SYMBOL',
        'CHAN': 'CHANNELIZATION',
        'PROFILD': 'PROFILED',
        'THERMOPLST': 'THERMOPLASTIC',
        'PAVEMEMT': 'PAVEMENT',  # Fix common typo
    }
    
    expanded = desc
    for abbr, full in abbreviations.items():
        # Use word boundaries to avoid partial replacements
        import re
        expanded = re.sub(r'\b' + abbr + r'\b', full, expanded)
    
    return expanded

# Load the reference categorization
print("Loading reference Pavement Marking.csv...")
reference = pd.read_csv('Data/Pavement Marking.csv')
print(f"Reference items: {len(reference)}")

# Load our unique pavement marking items
print("\nLoading pavement_marking_items.csv...")
items = pd.read_csv('output/pavement_marking_items.csv')
print(f"Our items: {len(items)}")

# Expand abbreviations in descriptions
print("\nExpanding abbreviations...")
items['description_expanded'] = items['description'].apply(expand_abbreviations)

# Standardize item numbers for matching
reference['item_no_clean'] = reference['Item Number'].str.strip()
items['item_no_clean'] = items['item_no'].str.strip()

# Merge with reference
print("\nMatching with reference categories...")
categorized = items.merge(
    reference[['item_no_clean', 'Item Class', 'Item Type']],
    on='item_no_clean',
    how='left'
)

# Rename columns
categorized = categorized.rename(columns={
    'Item Class': 'item_class',
    'Item Type': 'item_type'
})

# Count items with and without categories
matched = categorized['item_type'].notna().sum()
unmatched = categorized['item_type'].isna().sum()

print(f"Matched with reference: {matched}")
print(f"Need categorization: {unmatched}")

# Categorize remaining items based on description patterns
def categorize_item_type(row):
    if pd.notna(row['item_type']):
        return row['item_type']
    
    desc = str(row['description']).upper()
    
    # Plastic/Thermoplastic markings (including preformed)
    if 'PLASTIC' in desc or 'THERMO' in desc or 'PREFRMD' in desc or 'PREFORMED' in desc:
        return 'Plastic-Thermal'
    
    # Painted markings
    if 'PAINTED' in desc or 'PAINT' in desc:
        return 'Painted'
    
    # Miscellaneous pavement marking items
    if any(keyword in desc for keyword in [
        'MARKER', 'REMOVAL', 'REMOVE', 'RAISED', 
        'TRUNCATED DOME', 'WARNING MAT', 'BUFFER',
        'DESIGN-BUILD', 'PAVEMENT MARKING'
    ]):
        return 'Pavement Marking Misc.'
    
    # Epoxy markings
    if 'EPOXY' in desc:
        return 'Epoxy'
    
    # Tape markings
    if 'TAPE' in desc:
        return 'Tape'
    
    # Wet night visible markings
    if 'WET NIGHT' in desc or 'WET-NIGHT' in desc:
        return 'Pavement Marking Misc.'
    
    # If still not categorized, mark as Pavement Marking Misc.
    return 'Pavement Marking Misc.'

# Apply categorization
categorized['item_type'] = categorized.apply(categorize_item_type, axis=1)

# Drop the temporary column
categorized = categorized.drop('item_no_clean', axis=1)

# Show distribution
print("\n" + "="*80)
print("CATEGORIZATION RESULTS")
print("="*80)
print("\nItem Type Distribution:")
print(categorized['item_type'].value_counts())

# Save categorized items
output_file = 'output/pavement_marking_items_categorized.csv'
categorized.to_csv(output_file, index=False)
print(f"\n✓ Saved to: {output_file}")

# Show some examples
print("\n" + "="*80)
print("Sample categorizations:")
print("="*80)
sample_cols = ['item_no', 'description', 'description_expanded', 'item_type']
print(categorized[sample_cols].head(20).to_string(index=False))
