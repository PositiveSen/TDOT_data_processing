import os
import pandas as pd
from parse_bid_tabs import extract_bid_info

# Test specific problematic files
test_files = [
    'Data/Bid Tabs/2019/February 8 2019 Summary of Bids.pdf',  # Has corruption warnings
    'Data/Bid Tabs/2021/20211221_SummaryOfBids.pdf',  # Has reversed county pattern
    'Data/Bid Tabs/2022/20221021_SummaryOfBids.pdf',  # Has COUNTIES pattern
    'Data/Bid Tabs/2019/Summary of Bids April 5 2019.pdf',  # Another 2019 file
    'Data/Bid Tabs/2022/20220617_SummaryOfBids.pdf',  # Known good file (33 contracts)
]

results = []

print("Testing specific problematic files:\n")

for pdf_path in test_files:
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        continue
        
    print(f"Processing: {os.path.basename(pdf_path)}")
    
    contracts_info = extract_bid_info(pdf_path)
    
    if contracts_info and any(c.get('contract_number') for c in contracts_info):
        count = len([c for c in contracts_info if c.get('contract_number')])
        print(f"  ✓ Found {count} contracts")
        
        # Show first and last contract
        valid_contracts = [c for c in contracts_info if c.get('contract_number')]
        if valid_contracts:
            print(f"    First: {valid_contracts[0]['contract_number']} - {valid_contracts[0].get('county', 'NO COUNTY')[:50]}")
            if len(valid_contracts) > 1:
                print(f"    Last:  {valid_contracts[-1]['contract_number']} - {valid_contracts[-1].get('county', 'NO COUNTY')[:50]}")
        
        results.extend(contracts_info)
    else:
        print(f"  ❌ No contracts found")
    
    print()

# Create test output
if results:
    df = pd.DataFrame(results)
    columns_to_keep = ['contract_number', 'county', 'project_description', 'project_length', 'file_name']
    df = df[columns_to_keep]
    
    os.makedirs('output', exist_ok=True)
    df.to_csv('output/test_extraction.csv', index=False)
    
    print(f"\n{'='*60}")
    print(f"Total contracts extracted: {len(df)}")
    print(f"Saved to: output/test_extraction.csv")
    print(f"{'='*60}")
