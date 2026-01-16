import pandas as pd

# Load the TDOT data
print("Loading TDOT_data.csv...")
df = pd.read_csv('Data/TDOT_data.csv', encoding='latin-1', low_memory=False)

print(f"Total rows: {len(df):,}")

# Show first few column names to find unit
print("\nFirst 20 columns:")
for i, col in enumerate(list(df.columns)[:20]):
    print(f"  {i}: '{col}'")

# Check for Item No. and Item Description columns
item_no_col = None
item_desc_col = None
unit_col = None

for col in df.columns:
    if 'item' in col.lower() and 'no' in col.lower():
        item_no_col = col
    if 'item' in col.lower() and 'desc' in col.lower():
        item_desc_col = col
    if col.strip() == 'Units':
        unit_col = col

print(f"\nItem Number Column: {item_no_col}")
print(f"Item Description Column: {item_desc_col}")
print(f"Unit Column: {unit_col}")

if item_no_col and item_desc_col:
    # Select columns to work with
    cols_to_use = [item_no_col, item_desc_col]
    if unit_col:
        cols_to_use.append(unit_col)
    
    # Remove rows where item no or description is missing
    df_clean = df[cols_to_use].dropna(subset=[item_no_col, item_desc_col])
    
    print(f"Rows with both Item No. and Description: {len(df_clean):,}")
    
    # Get unique item numbers with their descriptions and units
    # Group by item number and take the first description and unit for each
    if unit_col:
        unique_items = df_clean.groupby(item_no_col).agg({
            item_desc_col: 'first',
            unit_col: 'first'
        }).reset_index()
        unique_items.columns = ['item_no', 'description', 'unit']
        
        # Count occurrences for separate file
        item_counts_data = df_clean.groupby(item_no_col).agg({
            item_desc_col: 'first',
            unit_col: 'first'
        }).reset_index()
        item_counts_data.columns = ['item_no', 'description', 'unit']
        counts = df_clean[item_no_col].value_counts().to_dict()
        item_counts_data['count'] = item_counts_data['item_no'].map(counts)
    else:
        unique_items = df_clean.groupby(item_no_col)[item_desc_col].first().reset_index()
        unique_items.columns = ['item_no', 'description']
        
        # Count occurrences for separate file
        item_counts_data = unique_items.copy()
        counts = df_clean[item_no_col].value_counts().to_dict()
        item_counts_data['count'] = item_counts_data['item_no'].map(counts)
    
    print(f"Unique Item Numbers: {len(unique_items):,}")
    
    # Remove all types of quotes from descriptions
    unique_items['description'] = unique_items['description'].str.replace('"', '', regex=False).str.replace("'", '', regex=False).str.replace('"', '', regex=False).str.replace('"', '', regex=False)
    item_counts_data['description'] = item_counts_data['description'].str.replace('"', '', regex=False).str.replace("'", '', regex=False).str.replace('"', '', regex=False).str.replace('"', '', regex=False)
    
    # Pad item numbers to have exactly 2 decimal parts (.00)
    def pad_item_number(item_no):
        item_no = str(item_no).strip()
        parts = item_no.split('.')
        if len(parts) == 1:
            # No decimal, add .00
            return f"{parts[0]}.00"
        elif len(parts) == 2:
            # One decimal, pad second part to 2 digits
            return f"{parts[0]}.{parts[1].zfill(2)}"
        else:
            # Already has 2 or more parts, keep as is
            return item_no
    
    unique_items['item_no'] = unique_items['item_no'].apply(pad_item_number)
    item_counts_data['item_no'] = item_counts_data['item_no'].apply(pad_item_number)
    
    # Sort counts by frequency (most frequent first)
    item_counts_data = item_counts_data.sort_values('count', ascending=False)
    
    # Save unique items (without count)
    output_file = 'output/unique_items.csv'
    unique_items.to_csv(output_file, index=False)
    print(f"\n✓ Saved unique items to: {output_file}")
    
    # Save item counts (separate file)
    counts_file = 'output/item_counts.csv'
    item_counts_data.to_csv(counts_file, index=False)
    print(f"✓ Saved item counts to: {counts_file}")
    
    # Show first 20 unique items
    print("\nFirst 20 unique items:")
    print(unique_items.head(20).to_string(index=False))
    
    # Show top 30 most frequent items
    print("\n\nTop 30 most frequent items:")
    print(item_counts_data.head(30).to_string(index=False))
    
else:
    print("\n❌ Could not find Item No. and/or Item Description columns")
    print("\nAvailable columns:")
    for col in df.columns:
        print(f"  - {col}")
