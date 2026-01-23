#!/usr/bin/env python3
"""
Export Unique Missing Item Numbers
Generates a clean list of missing item numbers from the provided CSV files
"""

import pandas as pd
import os
from collections import defaultdict

def get_unique_missing_items():
    """
    Get unique missing item numbers organized by category
    """
    print("=" * 60)
    print("UNIQUE MISSING ITEM NUMBERS")
    print("=" * 60)
    
    # Load the item classifications from CSV files
    item_lists_dir = 'Data/Item Lists'
    target_files = {
        'section3_base_and_subgrade_treatments.csv': {'keep_categories': ['302', '303']},
        'section4_flexible_surfaces.csv': {'keep_categories': ['401', '402', '403', '404', '405', '406', '407', '408', '409']},
        'section7_incidemtal_construction_and_services.csv': {'keep_categories': ['716']}
    }
    
    # Build classification mapping
    classification_mapping = set()
    
    for filename, config in target_files.items():
        filepath = os.path.join(item_lists_dir, filename)
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            keep_categories = config['keep_categories']
            category_pattern = '|'.join(f'^{cat}' for cat in keep_categories)
            
            filtered_df = df[df['Item Number'].str.match(category_pattern, na=False)]
            
            for _, row in filtered_df.iterrows():
                item_number = str(row['Item Number']).strip()
                classification_mapping.add(item_number)
    
    # Load bidding data
    df = pd.read_csv('Data/TDOT_data.csv', encoding='latin-1')
    
    # Filter for target categories (only the ones we have CSV files for)
    target_categories = ['716', '302', '303', '401', '402', '403', '404', '405', '406', '407', '408', '409']
    target_mask = df['Item No.'].str.match(r'^(' + '|'.join(target_categories) + ')-', na=False)
    target_data = df[target_mask].copy()
    
    # Find missing items by category
    missing_by_category = {}
    
    for category in target_categories:
        cat_data = target_data[target_data['Item No.'].str.match(f'^{category}-', na=False)]
        unique_items = sorted(cat_data['Item No.'].unique())
        
        missing_items = []
        for item in unique_items:
            if item not in classification_mapping:
                missing_items.append(item)
        
        missing_by_category[category] = missing_items
    
    # Output results
    all_missing = []
    category_names = {
        '716': 'Pavement Markings',
        '302': 'Base Course Materials',
        '303': 'Aggregates',
        '401': 'Mineral Aggregate Surface',
        '402': 'Bituminous Prime Coat',
        '403': 'Hot Mix Asphalt',
        '404': 'Asphalt Materials',
        '405': 'Seal Coat',
        '406': 'Micro-Surfacing',
        '407': 'Crack Sealing',
        '408': 'Chip Seal',
        '409': 'Other Asphalt'
    }
    
    for category in target_categories:
        missing_items = missing_by_category[category]
        if missing_items:
            print(f"\n{category_names[category]} ({category}):")
            print(f"Missing {len(missing_items)} unique items:")
            for item in missing_items:
                print(f"  {item}")
                all_missing.append(item)
        else:
            print(f"\n{category_names[category]} ({category}):")
            print("  All items available in CSV files ✅")
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"SUMMARY")
    print(f"=" * 60)
    print(f"Total unique missing items: {len(all_missing)}")
    
    # Save to file for easy copying
    with open('missing_item_numbers.txt', 'w') as f:
        f.write("UNIQUE MISSING ITEM NUMBERS\n")
        f.write("=" * 30 + "\n\n")
        
        for category in target_categories:
            missing_items = missing_by_category[category]
            f.write(f"{category_names[category]} ({category}):\n")
            if missing_items:
                for item in missing_items:
                    f.write(f"{item}\n")
            else:
                f.write("All items available ✅\n")
            f.write("\n")
        
        f.write(f"Total unique missing items: {len(all_missing)}\n")
    
    print(f"Complete list saved to: missing_item_numbers.txt")
    
    return missing_by_category

if __name__ == "__main__":
    missing_items = get_unique_missing_items()