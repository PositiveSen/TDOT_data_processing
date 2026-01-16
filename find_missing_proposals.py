import pandas as pd
import os

# Load bid tabs contracts
print("Loading bid tabs data...")
bid_tabs = pd.read_csv('output/bid_tabs_data.csv')
print(f'Found {bid_tabs["contract_number"].nunique()} unique contracts in bid_tabs_data.csv')

# Load and merge missing data if available
if os.path.exists('output/missing_data.csv'):
    print("\nLoading missing data...")
    missing_data = pd.read_csv('output/missing_data.csv')
    print(f'Found {len(missing_data)} contracts in missing_data.csv')
    
    # Merge the two datasets
    print("\nMerging bid_tabs_data and missing_data...")
    combined = pd.concat([bid_tabs, missing_data], ignore_index=True)
    
    # Save combined data
    combined_output = 'output/bid_tabs_combined.csv'
    combined.to_csv(combined_output, index=False)
    print(f'Saved combined data to {combined_output}')
    print(f'Total contracts in combined: {len(combined)} ({combined["contract_number"].nunique()} unique)')
    
    # Use combined data for finding missing
    bid_contracts = set(combined['contract_number'].unique())
else:
    print("\nNo missing_data.csv found, using only bid_tabs_data.csv")
    bid_contracts = set(bid_tabs['contract_number'].unique())

# Load unique proposals list with letting dates
print("\nLoading unique proposals...")
with open('output/unique_proposals.txt', 'r') as f:
    lines = f.readlines()

unique_data = []
for line in lines:
    line = line.strip()
    if not line or line.startswith('=') or line.startswith('Unique'):
        continue
    # Parse: "1. CNN001 - 20140110"
    parts = line.split('.', 1)
    if len(parts) == 2:
        # Split by ' - ' to get contract and letting date
        content = parts[1].strip()
        if ' - ' in content:
            contract, letting_date = content.split(' - ', 1)
            unique_data.append({'Proposal_ID': contract.strip(), 'Letting_Date': letting_date.strip()})
        else:
            unique_data.append({'Proposal_ID': content, 'Letting_Date': None})

unique_df = pd.DataFrame(unique_data)
print(f'Found {len(unique_df)} proposals in unique_proposals.txt')

# Find proposals in unique list but not in bid tabs
missing = unique_df[~unique_df['Proposal_ID'].isin(bid_contracts)]
missing = missing.sort_values('Letting_Date')

print(f'\nFound {len(missing)} proposals in unique list but NOT in bid tabs')

# Save to CSV
output_file = 'output/missing_proposals.csv'
missing.to_csv(output_file, index=False)
print(f'\nSaved to {output_file}')

print(f'\nFirst 20 missing proposals:')
print(missing.head(20).to_string(index=False))

print(f'\nLast 20 missing proposals:')
print(missing.tail(20).to_string(index=False))
