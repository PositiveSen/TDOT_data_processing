import os
import pandas as pd
from parse_bid_tabs import extract_bid_info

# Configuration
bid_tabs_dir = 'Data/Bid Tabs'
output_file = 'output/bid_tabs_data.csv'

results = []
pdf_files = []

# Find all PDF files
for root, dirs, files in os.walk(bid_tabs_dir):
    for file in files:
        if file.endswith('.pdf'):
            pdf_files.append(os.path.join(root, file))

total_files = len(pdf_files)
print(f"Found {total_files} PDF files to process\n")

# Process each PDF
for idx, pdf_path in enumerate(pdf_files, 1):
    print(f"Processing {idx}/{total_files}: {os.path.basename(pdf_path)}")
    
    # extract_bid_info now returns a list of dictionaries (one per contract)
    contracts_info = extract_bid_info(pdf_path)
    
    # Filter out empty or invalid contract info
    valid_contracts = [c for c in contracts_info if c.get('contract_number') is not None]
    
    # Debug: Check for empty results
    if not valid_contracts:
        print(f"  WARNING: No valid contracts found in this PDF")
    else:
        print(f"  Found {len(valid_contracts)} valid contracts")
    
    # Add relative path for reference to each valid contract
    for info in valid_contracts:
        info['file_path'] = os.path.relpath(pdf_path)
        results.append(info)

# Create DataFrame
df = pd.DataFrame(results)

# Select the columns we need
columns_to_keep = ['contract_number', 'county', 'project_description', 'project_length', 'project_length_unit']
if 'error' in df.columns:
    columns_to_keep.append('error')

df = df[columns_to_keep]

# Create output directory if needed
output_dir = os.path.dirname(output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save to CSV
df.to_csv(output_file, index=False)

print(f"\n{'=' * 80}")
print(f"Processing complete!")
print(f"Total PDF files available: {total_files}")
print(f"PDF files processed: {len(pdf_files)}")
print(f"Total contract entries: {len(df)}")
print(f"Contracts found: {df['contract_number'].notna().sum()}")
print(f"Unique contracts: {df['contract_number'].nunique()}")
print(f"Results saved to: {output_file}")
print(f"{'=' * 80}")

# Display summary statistics
print("\nSummary:")
print(f"  - Entries with contract numbers: {df['contract_number'].notna().sum()}")
print(f"  - Entries with project descriptions: {df['project_description'].notna().sum()}")
print(f"  - Entries with project lengths: {df['project_length'].notna().sum()}")
if 'error' in df.columns:
    print(f"  - Entries with errors: {df['error'].notna().sum()}")

# Display first few rows
print("\nFirst 10 rows of data:")
print(df.head(10).to_string())

# Display unique contract numbers
print(f"\nTotal unique contract numbers: {df['contract_number'].nunique()}")
