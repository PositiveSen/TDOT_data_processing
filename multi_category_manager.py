"""
Multi-Category Data Manager - Complete utility items processing and management
Handles categorization, storage, and retrieval of utility items with efficient caching
"""

import pandas as pd
import os
import pickle
import time
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

class MultiCategoryManager:
    """
    Complete utility items processing and management system.
    
    Handles:
    - Processing raw utility data into categorized DataFrames
    - Efficient storage and retrieval with automatic caching
    - Category-specific analysis and export
    - Memory-efficient separate DataFrame storage
    """
    
    def __init__(self, df: pd.DataFrame = None, storage_dir: str = "processed_data"):
        """
        Initialize with either a single DataFrame or empty for loading categories separately
        
        Args:
            df: Optional single DataFrame with 'category' column (will be split)
            storage_dir: Directory for storing processed data
        """
        self.category_dfs: Dict[str, pd.DataFrame] = {}
        self.base_columns = ['ITEM', 'ITEM SHORT DESCRIPTION', 'UNITS', 'ITEM LONG DESCRIPTION']
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        if df is not None:
            self._split_dataframe(df)
    
    def _split_dataframe(self, df: pd.DataFrame):
        """Split a single DataFrame into category-specific DataFrames"""
        for category in df['category'].unique():
            if category != 'UNCLASSIFIED':
                cat_df = df[df['category'] == category].copy()
                # Remove columns that are all NaN for this category
                cat_df = cat_df.dropna(axis=1, how='all')
                self.category_dfs[category] = cat_df
    
    def add_category(self, category_name: str, df: pd.DataFrame):
        """Add a category DataFrame"""
        self.category_dfs[category_name] = df.copy()
    
    def get_category(self, category_name: str) -> pd.DataFrame:
        """Get DataFrame for specific category"""
        return self.category_dfs.get(category_name, pd.DataFrame())
    
    def get_categories(self) -> List[str]:
        """Get list of all category names"""
        return list(self.category_dfs.keys())
    
    def get_category_fields(self, category_name: str) -> List[str]:
        """Get list of specific fields for a category (excluding base columns)"""
        if category_name not in self.category_dfs:
            return []
        
        cat_df = self.category_dfs[category_name]
        all_columns = set(cat_df.columns)
        base_set = set(self.base_columns + ['category'])
        category_fields = sorted(list(all_columns - base_set))
        
        # Only return fields that have non-null values
        return [field for field in category_fields if cat_df[field].notna().any()]
    
    def summary(self) -> pd.DataFrame:
        """Get summary statistics by category"""
        summary_data = []
        for category, df in self.category_dfs.items():
            fields = self.get_category_fields(category)
            summary_data.append({
                'Category': category,
                'Item Count': len(df),
                'Columns': len(df.columns),
                'Specific Fields': ', '.join(fields),
                'Memory Usage (bytes)': df.memory_usage(deep=True).sum()
            })
        
        summary_df = pd.DataFrame(summary_data)
        if not summary_df.empty:
            summary_df = summary_df.sort_values('Item Count', ascending=False)
        return summary_df
    
    # === PROCESSING AND STORAGE METHODS ===
    
    @classmethod
    def from_utility_categories(cls, 
                              utility_categories: List[str], 
                              apply_typo_correction: bool = True,
                              force_reprocess: bool = False,
                              storage_dir: str = "processed_data"):
        """
        Create MultiCategoryManager from utility categories with automatic processing and caching.
        
        Args:
            utility_categories: List of utility categories to process (e.g., ["790Electrical"])
            apply_typo_correction: Whether to apply typo corrections to descriptions
            force_reprocess: If True, ignore stored data and reprocess from scratch
            storage_dir: Directory for storing processed data
            
        Returns:
            MultiCategoryManager with processed utility items
        """
        manager = cls(storage_dir=storage_dir)
        
        data_file = manager._get_data_file_path(utility_categories, apply_typo_correction)
        
        # Use stored data if current and valid
        if not force_reprocess and manager._is_data_current(data_file):
            try:
                print(f"📁 Loading processed data: {data_file.name}")
                with open(data_file, 'rb') as f:
                    stored_manager = pickle.load(f)
                    manager.category_dfs = stored_manager.category_dfs
                return manager
            except (pickle.PickleError, EOFError) as e:
                print(f"⚠️ Stored data corrupted, reprocessing: {e}")
        
        # Process fresh data
        print(f"⚙️ Processing utility categories...")
        start_time = time.time()
        
        manager._process_utility_categories(utility_categories, apply_typo_correction)
        
        process_time = time.time() - start_time
        
        # Store processed data
        print(f"💾 Storing processed data: {data_file.name}")
        try:
            with open(data_file, 'wb') as f:
                pickle.dump(manager, f)
        except Exception as e:
            print(f"⚠️ Warning: Could not store processed data: {e}")
        
        print(f"✅ Processing completed in {process_time:.2f} seconds")
        return manager
    
    def _process_utility_categories(self, utility_categories, apply_typo_correction=True):
        """Process utility categories into structured DataFrames with categorization."""
        
        print("=== PROCESSING UTILITY CATEGORIES ===")
        
        if apply_typo_correction:
            print("✅ Typo correction enabled")
        
        for category in utility_categories:
            category_number = re.match(r'\d+', category).group()
            category_name = re.search(r'[A-Za-z]+', category).group()
            
            print(f"\nProcessing {category}...")
            
            # Load raw data with typo correction using internal method
            df_temp = self._load_utility_items(category, apply_typo_correction=apply_typo_correction)
            df_temp["ITEM"] = df_temp["ITEM"].apply(self.normalize_item_code)

            # Load specifier data
            project_root = Path(__file__).parent.parent
            
            # Try different filename variations for compatibility
            possible_filenames = [
                f"specifiers_{category_name}_new.json",
                f"Specifiers_{category_name}_new.json"
            ]
            
            specifier_file = None
            for filename in possible_filenames:
                potential_file = project_root / "data" / filename
                if potential_file.exists():
                    specifier_file = potential_file
                    break
            
            if specifier_file is None:
                raise FileNotFoundError(f"Could not find specifier file for {category}. Tried: {possible_filenames}")
                
            with open(specifier_file, "r") as f:
                specifier_data = json.load(f)

            print(f"Available categories: {list(specifier_data.get('categories', {}).keys())}")
            print(f"Processing {len(df_temp)} items...")
            
            # Track description usage
            long_desc_count = 0
            short_desc_count = 0
            
            # Group items by category
            category_data = {}
            unclassified_data = []
            
            for _, row in df_temp.iterrows():
                # Use SHORT description
                description_to_use = row["ITEM SHORT DESCRIPTION"]
                
                # Categorize this item using the selected description
                result = self.categorize_item_new_format(row["ITEM"], description_to_use, specifier_data)
                cat_name = result.get('category', 'UNCLASSIFIED')
                
                # Combine original data with categorization result
                combined_row = {**row.to_dict(), **result}
                
                if cat_name == 'UNCLASSIFIED':
                    unclassified_data.append(combined_row)
                else:
                    if cat_name not in category_data:
                        category_data[cat_name] = []
                    category_data[cat_name].append(combined_row)
            
            # Create separate DataFrames for each category
            print(f"Creating category DataFrames...")
            for cat_name, rows in category_data.items():
                cat_df = pd.DataFrame(rows)
                # Remove columns that are all NaN for this category
                cat_df = cat_df.dropna(axis=1, how='all')
                
                print(f"  {cat_name}: {len(cat_df)} items, {len(cat_df.columns)} columns")
                self.add_category(cat_name, cat_df)
            
            # Handle unclassified items
            if unclassified_data:
                print(f"  UNCLASSIFIED: {len(unclassified_data)} items")
    
    def _get_data_file_path(self, utility_categories: List[str], apply_typo_correction: bool) -> Path:
        """Generate file path for processed data based on processing parameters."""
        categories_str = "_".join(sorted(utility_categories))
        correction_str = "typo_corrected" if apply_typo_correction else "raw"
        filename = f"{categories_str}_{correction_str}.pkl"
        return self.storage_dir / filename
    
    def _is_data_current(self, data_file: Path) -> bool:
        """Check if processed data is current compared to source files."""
        if not data_file.exists():
            return False
        
        data_timestamp = data_file.stat().st_mtime
        
        # Check if source files are newer than processed data
        source_files = [
            'src/multi_category_manager.py', 
            'src/data_loading.py',
            'src/typo_corrector.py'
        ]
        
        for source_file in source_files:
            if os.path.exists(source_file):
                if os.path.getmtime(source_file) > data_timestamp:
                    print(f"Source file {source_file} is newer than processed data")
                    return False
        
        return True

    def _load_utility_items(self, category: str, apply_typo_correction: bool = True):
        """
        Loads utility items from a PDF file, extracts relevant data into a DataFrame,
        and removes repeated headers and empty rows.
        
        Args:
            category (str): The category of the utility items to load.
            apply_typo_correction (bool): Whether to apply typo corrections to descriptions.
            
        Returns:
            pandas.DataFrame: A DataFrame containing the utility items with columns 'ITEM',
            'ITEM SHORT DESCRIPTION', 'UNITS', and 'ITEM LONG DESCRIPTION'.
        """
        import pdfplumber
        try:
            from .typo_corrector import TypoCorrector
        except ImportError:
            from typo_corrector import TypoCorrector
        
        extracted_data = []
        project_root = Path(__file__).parent.parent
        pdf_path = project_root / "data" / f"Const-{category}.pdf"
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        if any(row):
                            extracted_data.append(row[:4])

        column_label = ['ITEM', 'ITEM SHORT DESCRIPTION', 'UNITS', 'ITEM LONG DESCRIPTION']
        df = pd.DataFrame(extracted_data, columns=column_label)

        # Remove repeated headers
        header_rows = df.iloc[:, 0].str.strip() == "ITEM"
        repeated_header_indices = header_rows[header_rows].index
        df = df.drop(index=repeated_header_indices).reset_index(drop=True)

        # Remove empty rows
        df = df[~df.apply(lambda row: row.astype(str).str.strip().eq('').all(), axis=1)]

        # Apply typo corrections if requested
        if apply_typo_correction:
            typo_corrector = TypoCorrector()
            typo_corrector.correct_dataframe_column(df, 'ITEM SHORT DESCRIPTION')
            typo_corrector.correct_dataframe_column(df, 'ITEM LONG DESCRIPTION')

        return df

    # === CATEGORIZATION METHODS ===
    
    @staticmethod
    def item_to_tuple(item_code):
        """Converts an item code in the format '123-45-67' to a tuple of integers.
        Args:
            item_code (str): The item code to convert.
        Returns:
            tuple: A tuple containing three integers representing the section and two parts of the item code."""
        parts = item_code.split("-")
        section = parts[0]
        rest = parts[1].split(".")
        return (int(section), int(rest[0]), int(rest[1]))

    @staticmethod
    def parse_item_range(item_str):
        """Parse item range string like '01..22,99' or '01..04' into individual item numbers.
        
        Args:
            item_str (str): String describing item ranges (e.g., "01..22,99")
            
        Returns:
            list: List of item numbers as integers
        """
        if not item_str:
            return []
        
        items = []
        parts = item_str.split(',')
        
        for part in parts:
            part = part.strip()
            if '..' in part:
                # Range like "01..22"
                start_str, end_str = part.split('..')
                start = int(start_str)
                end = int(end_str)
                items.extend(range(start, end + 1))
            else:
                # Single item like "99"
                items.append(int(part))
        
        return items

    @staticmethod
    def match_item_code_new_format(item_code, category_data, group):
        """Check if an item code matches any of the ranges in the new JSON format.
        
        Args:
            item_code (str): The item code to check (e.g., "790-01.01")
            category_data (dict): Category data from JSON
            group (str): The group number (e.g., "790")
            
        Returns:
            tuple: (is_match, sub_section) if match found, (False, None) otherwise
        """
        # Handle None or non-string values
        if item_code is None or not isinstance(item_code, str):
            return False, None
            
        # Parse item code: "790-01.01" -> group=790, sub=01, item=01
        parts = item_code.split('-')
        if len(parts) != 2:
            return False, None
            
        code_group = parts[0]
        if code_group != group:
            return False, None
            
        sub_item = parts[1].split('.')
        if len(sub_item) != 2:
            return False, None
            
        sub_section = sub_item[0]
        item_number = int(sub_item[1])
        
        # Check if this sub-section exists in the index
        for index_entry in category_data.get("index", []):
            if index_entry["sub"] == sub_section:
                if "items" in index_entry:
                    # Parse the items range and check if our item number is in it
                    valid_items = MultiCategoryManager.parse_item_range(index_entry["items"])
                    if item_number in valid_items:
                        return True, sub_section
                    else:
                        return False, None
                else:
                    # If no items specified, ALL items in this sub-section are valid for this category
                    # but we won't extract detailed fields for them
                    return True, sub_section
        
        return False, None

    @staticmethod
    def extract_fields_new_format(description, fields_config):
        """Extract fields from description using the new JSON format.
        
        Args:
            description (str): The item description to extract from
            fields_config (dict): Field configuration from JSON
            
        Returns:
            dict: Extracted field values
        """
        result = {}
        
        # Process regular fields first
        for field_name, field_spec in fields_config.items():
            # Skip conditional_fields - we'll handle them separately
            if field_name == "conditional_fields":
                continue
                
            method = field_spec.get("method", "unknown")
            
            if method == "match_first":
                # Find first matching option in the description
                options = field_spec.get("options", [])
                default = field_spec.get("default", "UNKNOWN")
                
                description_upper = description.upper()
                found = False
                for option in options:
                    if option.upper() in description_upper:
                        result[field_name] = option
                        found = True
                        break
                
                if not found:
                    result[field_name] = default
                    
            elif method == "regex_extract":
                # Extract using regex pattern
                pattern = field_spec.get("pattern", "")
                fallback = field_spec.get("fallback", "UNKNOWN")
                flags = field_spec.get("flags", [])
                data_type = field_spec.get("data_type", None)
                
                # Convert flags to re flags
                re_flags = 0
                if "i" in flags:
                    re_flags |= re.IGNORECASE
                if "m" in flags:
                    re_flags |= re.MULTILINE
                if "s" in flags:
                    re_flags |= re.DOTALL
                    
                match = re.search(pattern, description, re_flags)
                if match:
                    # Find the first non-empty group in the match
                    extracted_value = None
                    for group in match.groups():
                        if group is not None:
                            extracted_value = group
                            break
                    if extracted_value is None:
                        extracted_value = fallback
                else:
                    extracted_value = fallback
                
                # Apply data type conversion if specified
                if data_type and extracted_value != fallback:
                    try:
                        if data_type == "float":
                            extracted_value = float(extracted_value)
                        elif data_type == "int":
                            extracted_value = int(extracted_value)
                        elif data_type == "str":
                            extracted_value = str(extracted_value)
                    except (ValueError, TypeError) as e:
                        print(f"⚠️ Warning: Could not convert '{extracted_value}' to {data_type} for field '{field_name}': {e}")
                        # Keep original extracted value if conversion fails
                
                result[field_name] = extracted_value
                
            else:
                result[field_name] = "UNKNOWN"
        
        # Process conditional fields
        conditional_fields = fields_config.get("conditional_fields", [])
        for conditional_group in conditional_fields:
            apply_conditions = conditional_group.get("apply_if", {})
            conditional_field_configs = conditional_group.get("fields", {})
            
            # Check if conditions are met
            conditions_met = True
            for condition_field, condition_value in apply_conditions.items():
                # Check against already extracted fields
                if condition_field in result:
                    if result[condition_field].upper() != condition_value.upper():
                        conditions_met = False
                        break
                else:
                    # Check against description directly if field not extracted yet
                    if condition_value.upper() not in description.upper():
                        conditions_met = False
                        break
            
            # If conditions are met, extract the conditional fields
            if conditions_met:
                conditional_result = MultiCategoryManager.extract_fields_new_format(description, conditional_field_configs)
                result.update(conditional_result)
            else:
                # If conditions are NOT met, set conditional fields to "N/A"
                for field_name in conditional_field_configs.keys():
                    result[field_name] = "N/A"
        
        return result

    @staticmethod
    def categorize_item_new_format(item_code, description, specifier_data):
        """Categorize an item using the new JSON format.
        
        Args:
            item_code (str): The item code (e.g., "790-01.01")
            description (str): The item description
            specifier_data (dict): The specifier data in new JSON format
            
        Returns:
            dict: Category information and extracted fields
        """
        group = specifier_data.get("group", "")
        categories = specifier_data.get("categories", {})

        for category_name, category_data in categories.items():
            # Check if the item code matches this category
            is_match, sub_section = MultiCategoryManager.match_item_code_new_format(item_code, category_data, group)
            
            if is_match:
                # Check if this sub-section has specific items defined
                has_specific_items = False
                for index_entry in category_data.get("index", []):
                    if index_entry["sub"] == sub_section and "items" in index_entry:
                        has_specific_items = True
                        break
                
                if has_specific_items:
                    # Extract fields using the new format for specifically defined items
                    fields_config = category_data.get("fields", {})
                    conditional_fields = category_data.get("conditional_fields", [])
                    
                    # Combine regular fields with conditional fields for extraction
                    complete_config = fields_config.copy()
                    if conditional_fields:
                        complete_config['conditional_fields'] = conditional_fields
                    
                    extracted_fields = MultiCategoryManager.extract_fields_new_format(description, complete_config)
                    
                    return {
                        "category": category_name,
                        **extracted_fields
                    }
                else:
                    # For sub-sections without specific items, ALL items belong to this category
                    # Extract fields normally since no items means all items are included
                    fields_config = category_data.get("fields", {})
                    conditional_fields = category_data.get("conditional_fields", [])
                    
                    # Combine regular fields with conditional fields for extraction
                    complete_config = fields_config.copy()
                    if conditional_fields:
                        complete_config['conditional_fields'] = conditional_fields
                    
                    extracted_fields = MultiCategoryManager.extract_fields_new_format(description, complete_config)
                    
                    return {
                        "category": category_name,
                        **extracted_fields
                    }
        
        return {"category": "UNCLASSIFIED"}

    @staticmethod
    def normalize_item_code(code):
        """Normalize item code format from '790-01-02' to '790-01.02'."""
        if code is None or not isinstance(code, str):
            return code
        match = re.match(r"(\d{3}-\d{2})-(\d{2})", code)
        if match:
            return f"{match.group(1)}.{match.group(2)}"
        return code


# === CONVENIENCE FUNCTION ===

def load_utility_items(utility_categories: List[str] = ["790Electrical"], 
                      apply_typo_correction: bool = True,
                      force_reprocess: bool = False,
                      storage_dir: str = "processed_data") -> MultiCategoryManager:
    """
    Load processed utility items with automatic processing and storage management.
    
    Args:
        utility_categories: List of utility category codes (e.g., ["790Electrical"])
        apply_typo_correction: Whether to apply typo corrections to descriptions
        force_reprocess: If True, ignore stored data and reprocess from scratch
        storage_dir: Directory for storing processed data files
        
    Returns:
        MultiCategoryManager with categorized utility items ready for analysis
        
    Example:
        # Quick load (uses stored data after first run)
        manager = load_utility_items(["790Electrical"])
        pole_items = manager.get_category('POLE')
        
        # Force fresh processing
        manager = load_utility_items(["790Electrical"], force_reprocess=True)
    """
    return MultiCategoryManager.from_utility_categories(
        utility_categories=utility_categories,
        apply_typo_correction=apply_typo_correction,
        force_reprocess=force_reprocess,
        storage_dir=storage_dir
    )


if __name__ == "__main__":
    print("=== UTILITY ITEMS PROCESSING DEMO ===")
    
    # Load processed utility items (will process and store on first run)
    manager = load_utility_items(["790Electrical"])
    print(f"Categories available: {manager.get_categories()}")
