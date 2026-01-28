#!/usr/bin/env python3
"""
TDOT Data Reader - Simple class to get unique items with filtering
"""

import pandas as pd
from typing import Optional
import glob
import os


class TDOTDataReader:
    """
    Simple reader for TDOT_data.csv to get unique items with filtering.
    """
    
    def __init__(self, data_file: str = 'Data/TDOT_data.csv'):
        """Initialize TDOT data reader."""
        self.data_file = data_file
        self.df = None
        self.item_lists = None
        
    def load_data(self, encoding: str = 'latin-1') -> pd.DataFrame:
        """Load TDOT data from CSV file."""
        self.df = pd.read_csv(self.data_file, encoding=encoding)
        self._load_item_lists()
        return self.df
    
    def _load_item_lists(self):
        """Load item mapping data from Item Lists folder."""
        item_lists_path = "Data/Item Lists/"
        csv_files = glob.glob(os.path.join(item_lists_path, "section*.csv"))
        
        all_items = []
        for file in csv_files:
            try:
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                df = None
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file, encoding=encoding)
                        print(f"Loaded {os.path.basename(file)} with {encoding} encoding ({len(df)} items)")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is not None:
                    all_items.append(df)
                else:
                    print(f"Warning: Could not load {file} with any encoding")
                    
            except Exception as e:
                print(f"Warning: Could not load {file}: {e}")
        
        if all_items:
            self.item_lists = pd.concat(all_items, ignore_index=True)
            # Clean item numbers for consistent matching
            self.item_lists['Item Number'] = self.item_lists['Item Number'].astype(str).str.strip()
            print(f"Total loaded: {len(self.item_lists)} items from item lists")
        else:
            print("Warning: No item list files found")
    
    def get_unique_items(self, category_filter=None) -> pd.DataFrame:
        """Get unique items with optional filtering at 3 levels.
        
        Args:
            category_filter: Can be:
                - Level 1: '105' (all 105-xx.xx items)
                - Level 2: '105-01' (all 105-01.xx items)  
                - Level 3: '105-01.20' (specific item)
                - Tuple: ('105-01', '105-08') for ranges
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Select the specific columns we need
        columns = ['Item No.', 'Item Description', 'Units', 'Supplemental Description', 'Proposal ID']
        item_df = self.df[columns].copy()
        
        # Rename to standard names
        item_df = item_df.rename(columns={
            'Item No.': 'item_number',
            'Item Description': 'item_description', 
            'Units': 'unit_measure',
            'Supplemental Description': 'supplementary_info',
            'Proposal ID': 'proposal_id'
        })
        
        # Apply filter if provided
        if category_filter:
            if isinstance(category_filter, tuple):
                # Range filter
                start, end = category_filter
                item_df = item_df[
                    (item_df['item_number'] >= start) & 
                    (item_df['item_number'] <= end)
                ]
            else:
                # Prefix filter - works for all 3 levels
                item_df = item_df[item_df['item_number'].str.startswith(category_filter, na=False)]
        
        # Clean data
        item_df = self._clean_data(item_df)
        
        # Calculate counts before deduplication
        count_stats = item_df.groupby(['item_number', 'supplementary_info']).agg({
            'proposal_id': ['nunique', 'count']  # Unique proposals and total occurrences
        })
        count_stats.columns = ['unique_proposals', 'total_occurrences']
        count_stats = count_stats.reset_index()
        
        # Get unique items by item number AND supplementary info
        unique_df = item_df.drop_duplicates(subset=['item_number', 'supplementary_info'])
        
        # Merge with count statistics
        unique_df = unique_df.merge(count_stats, on=['item_number', 'supplementary_info'], how='left')
        
        # Add item list mappings
        unique_df = self._add_item_mappings(unique_df)
        
        # Select final columns and sort by item number
        final_columns = ['item_number', 'item_description', 'unit_measure', 'unit_full_name', 'item_class', 'item_type', 'supplementary_info', 'unique_proposals', 'total_occurrences']
        unique_df = unique_df[final_columns].sort_values('item_number')
        
        # Report items not found in CSV files
        if 'found_in_csv' in unique_df.columns:
            not_found = unique_df[~unique_df['found_in_csv']]
            if len(not_found) > 0:
                print(f"\n⚠️  {len(not_found)} items not found in CSV files:")
                for item in not_found['item_number'].head(20):  # Show first 20
                    print(f"   {item}")
                if len(not_found) > 20:
                    print(f"   ... and {len(not_found) - 20} more")
            else:
                print("✅ All items found in CSV files")
            # Remove the helper column before returning
            unique_df = unique_df.drop('found_in_csv', axis=1)
        
        return unique_df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        cleaned_df = df.copy()
        
        # Remove completely empty rows
        cleaned_df = cleaned_df.dropna(how='all')
        
        # Clean item numbers and descriptions
        for col in ['item_number', 'item_description']:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                cleaned_df = cleaned_df[
                    (cleaned_df[col].notna()) & 
                    (cleaned_df[col] != '') &
                    (cleaned_df[col] != 'nan')
                ]
        
        # Fill missing supplementary info with empty string for consistent grouping
        if 'supplementary_info' in cleaned_df.columns:
            cleaned_df['supplementary_info'] = cleaned_df['supplementary_info'].fillna('').astype(str).str.strip()
        
        # Clean bidder column
        if 'bidder' in cleaned_df.columns:
            cleaned_df['bidder'] = cleaned_df['bidder'].astype(str).str.strip()
        
        return cleaned_df
    
    def _add_item_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add item class, item type, and full unit names from item lists."""
        if self.item_lists is None:
            # Add empty columns if no mapping data
            df['unit_full_name'] = ''
            df['item_class'] = ''
            df['item_type'] = ''
            df['found_in_csv'] = False
        else:
            # Merge with item lists
            merged_df = df.merge(
                self.item_lists[['Item Number', 'U/M', 'Item Class', 'Item Type']],
                left_on='item_number',
                right_on='Item Number',
                how='left'
            )
            
            # Mark items found in CSV
            merged_df['found_in_csv'] = merged_df['Item Number'].notna()
            
            # Rename and fill missing values
            df = merged_df.rename(columns={
                'U/M': 'unit_full_name',
                'Item Class': 'item_class',
                'Item Type': 'item_type'
            })
            
            # Fill missing mappings
            df['unit_full_name'] = df['unit_full_name'].fillna('')
            df['item_class'] = df['item_class'].fillna('')
            df['item_type'] = df['item_type'].fillna('')
            
            # Drop the duplicate Item Number column
            df = df.drop('Item Number', axis=1)
        
        # Add manual utility mappings
        df = self._add_utility_mappings(df)
        
        return df
    
    def _add_utility_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add manual utility mappings for 790+ items."""
        utility_mappings = {
            '790': ('Utilities', 'Electrical Utilities'),
            '791': ('Utilities', 'Gas/Petroleum Distribution Utilities'), 
            '793': ('Utilities', 'Phone Utilities (Bst, Peoples, Etc.)'),
            '795': ('Utilities', 'Potable Water Utilities'),
            '797': ('Utilities', 'Sewers'),
            '798': ('Utilities', 'Cable Tv')
        }
        
        for prefix, (item_class, item_type) in utility_mappings.items():
            mask = df['item_number'].str.startswith(prefix, na=False)
            # Only override if not already set from item lists
            df.loc[mask & (df['item_class'] == ''), 'item_class'] = item_class
            df.loc[mask & (df['item_type'] == ''), 'item_type'] = item_type
        
        # Set Unknown for any remaining empty values
        df['item_class'] = df['item_class'].replace('', 'Unknown')
        df['item_type'] = df['item_type'].replace('', 'Unknown')
        
        return df


if __name__ == "__main__":
    # Example usage - generates CSV files
    reader = TDOTDataReader()
    
    print("Loading TDOT data...")
    df = reader.load_data()
    print(f"Total records: {len(df):,}")
    
    print("\nExtracting unique items...")
    
    # Get all unique items
    all_items = reader.get_unique_items()
    all_items.to_csv('unique_items_all.csv', index=False)
    print(f"Saved {len(all_items)} unique items to 'unique_items_all.csv'")
    