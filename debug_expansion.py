#!/usr/bin/env python3

import pandas as pd
import json
import re

def debug_expansion():
    # Load config
    with open('description_corrections.json', 'r') as f:
        config = json.load(f)
    print("Config loaded successfully")
    
    # Load sample data
    df = pd.read_csv('Data/TDOT_data.csv', encoding='latin-1')
    print(f"Data loaded: {len(df)} records")
    
    # Filter to 716
    df_716 = df[df['Item No.'].str.startswith('716', na=False)].copy()
    print(f"716 records: {len(df_716)}")
    
    # Get a sample
    sample_descriptions = df_716['Item Description'].head(10).tolist()
    print(f"\nSample original descriptions:")
    for i, desc in enumerate(sample_descriptions[:5]):
        print(f"  {i+1}. {desc}")
    
    # Copy for expansion
    df_716['item_description_expanded'] = df_716['Item Description'].copy()
    
    # Apply expansions - replicate the exact logic from the main script
    if config and 'description_corrections' in config:
        corrections = config['description_corrections']
        
        # Apply abbreviations
        if 'abbreviations' in corrections:
            abbreviation_map = corrections['abbreviations']
            print(f"\nApplying abbreviations: {abbreviation_map}")
            
            for abbrev, expansion in abbreviation_map.items():
                pattern = f'\\b{abbrev}\\b'
                print(f"  Applying pattern: {pattern} -> {expansion}")
                
                before = df_716['item_description_expanded'].iloc[0]
                df_716['item_description_expanded'] = df_716['item_description_expanded'].str.replace(
                    pattern, expansion, regex=True, case=False
                )
                after = df_716['item_description_expanded'].iloc[0]
                
                if before != after:
                    print(f"    CHANGED: {before} -> {after}")
                else:
                    print(f"    No change for first item")
        
        # Apply regex patterns  
        if 'regex_patterns' in corrections:
            print(f"\nApplying regex patterns...")
            for pattern_config in corrections['regex_patterns']:
                pattern = pattern_config['pattern']
                replacement = pattern_config['replacement']
                print(f"  Pattern: {pattern} -> {replacement}")
                
                df_716['item_description_expanded'] = df_716['item_description_expanded'].str.replace(
                    pattern, replacement, regex=True
                )
    
    # Show results
    print(f"\nResults comparison:")
    for i in range(min(5, len(df_716))):
        original = df_716['Item Description'].iloc[i]
        expanded = df_716['item_description_expanded'].iloc[i] 
        print(f"  {i+1}. Original: {original}")
        print(f"     Expanded: {expanded}")
        print(f"     Changed: {original != expanded}")
        print()

if __name__ == "__main__":
    debug_expansion()