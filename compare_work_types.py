import pandas as pd
import numpy as np

def load_tdot_categories():
    """
    Load TDOT_data and aggregate category information by proposal.
    """
    print("Loading TDOT_data.csv...")
    df = pd.read_csv('Data/TDOT_data.csv', encoding='latin-1')
    
    # Get unique proposals with their primary categories
    # Group by Proposal ID and aggregate categories
    proposal_categories = df.groupby('Proposal ID').agg({
        'Category Description': lambda x: ', '.join(sorted(set(x.dropna()))),
        'Primary County': 'first',
        'Letting Date': 'first'
    }).reset_index()
    
    # Determine primary work type from TDOT categories
    def get_tdot_work_type(categories_str):
        if pd.isna(categories_str):
            return 'Unknown'
        
        cats = categories_str.upper()
        
        # Map TDOT categories to our work types
        if 'BRIDGE' in cats and 'ROADWAY' not in cats:
            return 'Bridge'
        elif 'MAINTENANCE' in cats:
            return 'Maintenance'
        elif 'ROADWAY' in cats and 'BRIDGE' in cats:
            return 'Roadway+Bridge'
        elif 'ROADWAY' in cats:
            return 'Roadway'
        elif 'UTILITY' in cats:
            return 'Utility'
        else:
            return 'Other'
    
    proposal_categories['tdot_work_type'] = proposal_categories['Category Description'].apply(get_tdot_work_type)
    
    return proposal_categories


def load_categorized_data():
    """
    Load our categorized bid tabs data.
    """
    print("Loading bid_tabs_combined_categorized.csv...")
    df = pd.read_csv('output/bid_tabs_combined_categorized.csv')
    
    return df[['contract_number', 'work_type', 'project_description', 'county']]


def compare_work_types():
    """
    Compare TDOT categories with our categorized work types.
    """
    # Load both datasets
    tdot_cats = load_tdot_categories()
    our_cats = load_categorized_data()
    
    # Merge on Proposal ID / contract_number
    print("\nMerging datasets...")
    merged = pd.merge(
        our_cats,
        tdot_cats,
        left_on='contract_number',
        right_on='Proposal ID',
        how='inner'
    )
    
    print(f"Found {len(merged)} matching proposals\n")
    
    # Create a mapping for comparison
    def categorize_match(row):
        our_type = row['work_type']
        tdot_type = row['tdot_work_type']
        
        # Define matches
        matches = {
            'Bridge Work': ['Bridge', 'Roadway+Bridge'],
            'Maintenance': ['Maintenance'],
            'Roadway Improvements': ['Roadway', 'Roadway+Bridge'],
            'Resurfacing': ['Roadway'],
            'New Construction/Reconstruction': ['Roadway', 'Roadway+Bridge'],
            'Surface Treatment': ['Roadway', 'Maintenance'],
        }
        
        if our_type in matches:
            return tdot_type in matches[our_type]
        else:
            return False
    
    merged['match'] = merged.apply(categorize_match, axis=1)
    
    # Summary statistics
    print("="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\nTotal proposals compared: {len(merged)}")
    print(f"Matching work types: {merged['match'].sum()} ({merged['match'].sum()/len(merged)*100:.1f}%)")
    print(f"Different work types: {(~merged['match']).sum()} ({(~merged['match']).sum()/len(merged)*100:.1f}%)")
    
    # Cross-tabulation
    print("\n" + "="*80)
    print("WORK TYPE COMPARISON")
    print("="*80)
    crosstab = pd.crosstab(merged['work_type'], merged['tdot_work_type'], margins=True)
    print(crosstab)
    
    # Show some mismatches
    print("\n" + "="*80)
    print("SAMPLE MISMATCHES (first 20)")
    print("="*80)
    mismatches = merged[~merged['match']]
    
    if len(mismatches) > 0:
        for idx, row in mismatches.head(20).iterrows():
            print(f"\n{row['contract_number']}:")
            print(f"  Our category: {row['work_type']}")
            print(f"  TDOT category: {row['tdot_work_type']} ({row['Category Description'][:80]}...)")
            print(f"  Description: {row['project_description'][:100]}...")
    
    # Save detailed comparison
    output_file = 'output/work_type_comparison.csv'
    merged.to_csv(output_file, index=False)
    print(f"\n{'='*80}")
    print(f"Detailed comparison saved to: {output_file}")
    print(f"{'='*80}\n")
    
    return merged


if __name__ == "__main__":
    comparison = compare_work_types()
