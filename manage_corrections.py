#!/usr/bin/env python3
"""
Utility to manage and test description corrections for TDOT pavement marking data.
Allows easy addition of new correction rules without modifying code.
"""

import json
import pandas as pd
from enhanced_pavement_marking_model import load_description_corrections, expand_and_correct_description

def add_correction(correction_type, old_text, new_text, config_file='description_corrections.json'):
    """Add a new correction rule to the configuration"""
    config = load_description_corrections(config_file)
    
    if correction_type not in config['description_corrections']:
        config['description_corrections'][correction_type] = {}
    
    config['description_corrections'][correction_type][old_text] = new_text
    
    # Save updated configuration
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Added correction: {old_text} → {new_text} (type: {correction_type})")

def test_corrections_on_data(sample_size=100):
    """Test corrections on actual data samples"""
    print("🔍 Testing corrections on real data samples...")
    
    # Load sample data
    df = pd.read_csv('Data/TDOT_data.csv', encoding='latin-1')
    pavement_data = df[df['Item No.'].str.startswith('716-', na=False)].copy()
    
    # Get sample of unique descriptions
    sample_descriptions = pavement_data['Item Description'].drop_duplicates().sample(min(sample_size, len(pavement_data)))
    
    config = load_description_corrections()
    
    print(f"\\nSample corrections (showing first 15):")
    print("="*80)
    
    for i, desc in enumerate(sample_descriptions.head(15)):
        original = str(desc)
        corrected = expand_and_correct_description(desc, config)
        
        if original != corrected:
            print(f"{i+1:2d}. CHANGED:")
            print(f"    Original:  {original}")
            print(f"    Corrected: {corrected}")
        else:
            print(f"{i+1:2d}. No change: {original}")
        print()

def find_potential_corrections():
    """Find potential description issues that might need correction"""
    print("🔍 Analyzing data for potential corrections...")
    
    df = pd.read_csv('Data/TDOT_data.csv', encoding='latin-1')
    pavement_data = df[df['Item No.'].str.startswith('716-', na=False)].copy()
    
    descriptions = pavement_data['Item Description'].dropna()
    
    # Look for common patterns that might need correction
    patterns_to_check = [
        ('Potential abbreviations', r'\\b[A-Z]{2,6}\\b'),
        ('Numbers with IN', r'\\d+\\s*IN\\b'),
        ('Potential typos', r'\\b\\w*[A-Z]{2,}[A-Z]\\w*\\b'),
    ]
    
    print("\\nPotential correction opportunities:")
    print("="*50)
    
    for pattern_name, pattern in patterns_to_check:
        import re
        matches = descriptions.str.extractall(f'({pattern})')[0].value_counts().head(10)
        if not matches.empty:
            print(f"\\n{pattern_name}:")
            for match, count in matches.items():
                print(f"  {match:<25} ({count:,} occurrences)")

def show_current_config():
    """Display current configuration"""
    config = load_description_corrections()
    
    print("📋 Current Description Correction Configuration:")
    print("="*60)
    
    for correction_type in config.get('processing_order', []):
        corrections = config['description_corrections'].get(correction_type, {})
        if corrections:
            print(f"\\n{correction_type.replace('_', ' ').title()}:")
            if correction_type == 'regex_patterns':
                for pattern in corrections:
                    print(f"  Pattern: {pattern['pattern']}")
                    print(f"    → {pattern['replacement']} ({pattern['description']})")
            else:
                for old, new in corrections.items():
                    print(f"  {old:<20} → {new}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("📋 Description Correction Utility")
        print("="*40)
        print("Usage:")
        print("  python manage_corrections.py show           - Show current config")
        print("  python manage_corrections.py test           - Test on real data")
        print("  python manage_corrections.py find           - Find potential issues")
        print("  python manage_corrections.py add <type> <old> <new>  - Add correction")
        print("\\nExample:")
        print("  python manage_corrections.py add abbreviations MRKNG MARKING")
    
    elif sys.argv[1] == 'show':
        show_current_config()
    elif sys.argv[1] == 'test':
        test_corrections_on_data()
    elif sys.argv[1] == 'find':
        find_potential_corrections()
    elif sys.argv[1] == 'add' and len(sys.argv) == 5:
        add_correction(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("❌ Invalid command. Use 'show', 'test', 'find', or 'add <type> <old> <new>'")