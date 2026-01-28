"""
TDOT LLM Feature Extraction - Version 3 (4-Step Approach)
Clean implementation focusing on 716 pavement markings only
"""

import os
import pandas as pd
import json
import re
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import anthropic


@dataclass
class LLMExtractedFeatures:
    """Container for features extracted by LLM"""
    key_insights: str
    extracted_features: Dict[str, Any]
    raw_response: str


class TDOTLLMFeatureExtractor:
    """4-Step LLM Feature Extraction for 716 Pavement Markings"""
    
    def __init__(self, 
                 data_file: str = 'Data/TDOT_data.csv',
                 item_lists_dir: str = 'Data/Item Lists',
                 corrections_config_file: str = 'description_corrections.json',
                 supplemental_classifications_file: str = 'Data/supplemental_item_classifications.json',
                 output_dir: str = 'llm_output'):
        
        self.data_file = data_file
        self.item_lists_dir = item_lists_dir
        self.corrections_config_file = corrections_config_file
        self.supplemental_classifications_file = supplemental_classifications_file
        self.output_dir = output_dir
        
        # Data storage
        self.df = None
        self.item_classifications = {}
        self.supplemental_classifications = {}
        self.corrections_config = None
        self.extracted_features = {}
        
        # LLM setup
        self.llm_provider = 'anthropic'
        self.model_name = 'claude-sonnet-4-5'
        self.llm_client = None
        self._initialize_llm()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load configurations
        self._load_configurations()
        
    def _load_configurations(self):
        """Load JSON configuration files like universal model"""
        print("Loading configuration files...")
        
        # Load description corrections
        self.corrections_config = self._load_description_corrections()
        print(f"  ✅ Description corrections loaded")
        
        # Load supplemental classifications
        self.supplemental_classifications = self._load_supplemental_classifications()
        print(f"  ✅ Supplemental classifications loaded")
    
    def _load_description_corrections(self) -> dict:
        """Load description correction rules from JSON configuration"""
        try:
            with open(self.corrections_config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {self.corrections_config_file} not found, no abbreviation expansion will be applied")
            return {}
    
    def _load_supplemental_classifications(self) -> dict:
        """Load supplemental item classifications for missing items"""
        try:
            with open(self.supplemental_classifications_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {self.supplemental_classifications_file} not found")
            return {}
        
    def _initialize_llm(self):
        """Initialize Claude client"""
        try:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                print("❌ ANTHROPIC_API_KEY environment variable not set")
                return
                
            self.llm_client = anthropic.Anthropic(api_key=api_key)
            print(f"🤖 Using Claude {self.model_name}")
            
        except Exception as e:
            print(f"❌ Failed to initialize Claude: {e}")
            self.llm_client = None
    
    def load_data(self) -> bool:
        """Load and filter TDOT data to 716 pavement markings only"""
        try:
            print(f"📊 Loading TDOT data from Data/TDOT_data.csv...")
            self.df = pd.read_csv('Data/TDOT_data.csv', encoding='latin-1')
            print(f"  Loaded {len(self.df):,} total records")
            
            # Filter to 716 pavement markings only
            original_count = len(self.df)
            self.df = self.df[self.df['Item No.'].str.startswith('716', na=False)].copy()
            print(f"  Filtered to 716 pavement markings: {len(self.df):,} records")
            
            # Apply exclusion filters from universal model
            print(f"  Applying exclusion filters...")
            before_exclusion = len(self.df)
            
            # Exclude 716-99.* (generic items)
            self.df = self.df[~self.df['Item No.'].str.match('716-99.*', na=False)]
            
            # Exclude specific removal items: 716-08.31-34 and 716-08.11-17
            removal_items = [f"716-08.{i:02d}" for i in list(range(31, 35)) + list(range(11, 18))]
            self.df = self.df[~self.df['Item No.'].isin(removal_items)]
            
            excluded_count = before_exclusion - len(self.df)
            if excluded_count > 0:
                print(f"  Excluded {excluded_count:,} records (generic and removal items)")
            
            print(f"  Final 716 dataset: {len(self.df):,} records")
            
            # Clean data
            print(f"  Applying data cleaning...")
            initial_count = len(self.df)
            self.df = self.df.dropna(subset=['Item Description'])
            self.df = self.df[self.df['Item Description'].str.strip() != '']
            removed_count = initial_count - len(self.df)
            if removed_count > 0:
                print(f"  Removed {removed_count} invalid entries")
            print(f"  Final dataset: {len(self.df):,} records")
            
            # Expand abbreviations
            print(f"  Expanding abbreviations in item descriptions...")
            self._expand_abbreviations()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def _expand_abbreviations(self):
        """Expand abbreviations using JSON configuration only"""
        self.df['item_description_expanded'] = self.df['Item Description'].copy()
        
        # Use only the abbreviations from the loaded JSON configuration
        if self.corrections_config and 'description_corrections' in self.corrections_config:
            corrections = self.corrections_config['description_corrections']
            
            # Apply abbreviations if they exist in config
            if 'abbreviations' in corrections:
                abbreviation_map = corrections['abbreviations']
                print(f"  Applying {len(abbreviation_map)} abbreviations...")
                
                for abbrev, expansion in abbreviation_map.items():
                    # Replace whole words only (word boundaries)
                    pattern = f'\\b{abbrev}\\b'
                    self.df['item_description_expanded'] = self.df['item_description_expanded'].str.replace(
                        pattern, expansion, regex=True, case=False
                    )
            
            # Apply regex patterns if they exist in config
            if 'regex_patterns' in corrections:
                print(f"  Applying {len(corrections['regex_patterns'])} regex patterns (inch standardization)...")
                for pattern_config in corrections['regex_patterns']:
                    pattern = pattern_config['pattern']
                    replacement = pattern_config['replacement']
                    self.df['item_description_expanded'] = self.df['item_description_expanded'].str.replace(
                        pattern, replacement, regex=True
                    )
        
        # Show summary
        pvmt_original = len(self.df[self.df['Item Description'].str.contains('PVMT', na=False)])
        pvmt_expanded = len(self.df[self.df['item_description_expanded'].str.contains('PVMT', na=False)])
        print(f"  ✅ Abbreviations expanded: {pvmt_original - pvmt_expanded} PVMT→PAVEMENT conversions")
    
    def load_item_classifications(self) -> bool:
        """Load item classifications from CSV files"""
        try:
            print(f"\n📋 Loading item classifications from Data/Item Lists...")
            classifications_dir = 'Data/Item Lists'
            
            if not os.path.exists(classifications_dir):
                print(f"❌ Classifications directory not found: {classifications_dir}")
                return False
            
            total_items = 0
            for filename in os.listdir(classifications_dir):
                if filename.endswith('.csv'):
                    file_path = os.path.join(classifications_dir, filename)
                    try:
                        df_class = pd.read_csv(file_path)
                        
                        # Standard columns
                        if 'Item Number' in df_class.columns and 'Item Class' in df_class.columns:
                            for _, row in df_class.iterrows():
                                item_num = str(row['Item Number']).strip()
                                if pd.notna(item_num) and item_num:
                                    self.item_classifications[item_num] = {
                                        'item_class': str(row.get('Item Class', '')).strip(),
                                        'item_type': str(row.get('Item Type', '')).strip()
                                    }
                                    total_items += 1
                        
                        print(f"    Processed {filename}: {len(df_class)} items")
                        
                    except Exception as e:
                        print(f"    ⚠️  Error processing {filename}: {e}")
            
            print(f"  ✅ Total item classifications loaded: {total_items:,}")
            print(f"  📊 Unique item numbers mapped: {len(self.item_classifications):,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading classifications: {e}")
            return False
    
    def merge_classifications(self):
        """Merge item classifications with the main dataset"""
        print(f"\n🔄 Merging classifications with bidding data...")
        
        # Load supplemental classifications
        supplemental_path = 'Data/supplemental_item_classifications.json'
        supplemental_count = 0
        if os.path.exists(supplemental_path):
            with open(supplemental_path, 'r') as f:
                supplemental = json.load(f)
                self.item_classifications.update(supplemental)
                supplemental_count = len(supplemental)
                print(f"  📋 Loaded {supplemental_count} supplemental classifications")
        
        # Apply classifications
        self.df['item_class'] = 'Unknown'
        self.df['item_type'] = 'Unknown'
        
        matched_count = 0
        for idx, row in self.df.iterrows():
            item_num = str(row['Item No.']).strip()
            if item_num in self.item_classifications:
                classification = self.item_classifications[item_num]
                self.df.at[idx, 'item_class'] = classification.get('item_class', 'Unknown')
                self.df.at[idx, 'item_type'] = classification.get('item_type', 'Unknown')
                matched_count += 1
        
        print(f"  ✅ Matched {matched_count:,} records with classifications")
        match_rate = (matched_count / len(self.df)) * 100 if len(self.df) > 0 else 0
        print(f"  📊 Match rate: {match_rate:.1f}%")
        
        # Show missing items
        unmatched = self.df[self.df['item_class'] == 'Unknown']['Item No.'].unique()
        if len(unmatched) > 0:
            print(f"  ⚠️  Still missing: {len(unmatched)} unique items")
            print(f"    Sample: {list(unmatched[:5])}")
        
        # Show summary
        print(f"\n  Item Classes found:")
        class_counts = self.df['item_class'].value_counts()
        for item_class, count in class_counts.head(10).items():
            print(f"    • {item_class}: {count:,} records")
    
    def extract_features_4_step(self):
        """
        6-step data-driven approach: understand → group similar items → extract properties → group properties → convert CSV → remove redundancy
        """
        print(f"\n🧪 6-Step Data-Driven LLM Feature Extraction...")
        
        if not self.llm_client:
            print("❌ No LLM client available")
            return False
        
        # Get unique 716 items using expanded descriptions with Item No.
        unique_items = self.df[['Item No.', 'item_description_expanded', 'Item Description']].drop_duplicates()
        print(f"  📊 Processing {len(unique_items)} unique pavement marking descriptions")
        
        # Execute 6-step process with exports at each step
        step1_response = self._step1_understand_items(unique_items)
        if not step1_response:
            return False
        
        step2_response = self._step2_group_similar_items(unique_items)
        if not step2_response:
            return False
        
        step3_response = self._step3_extract_properties(unique_items)
        if not step3_response:
            return False
            
        step4_response = self._step4_group_properties(step3_response)
        if not step4_response:
            return False
        
        step5_success = self._step5_convert_to_csv(unique_items, step4_response)
        if not step5_success:
            return False
            
        return self._step6_remove_redundancy()
    
    def _step1_understand_items(self, unique_items):
        """Step 1: Ask LLM to understand all items"""
        print(f"    STEP 1: Understanding items...")
        
        items_list = "\n".join([
            f"{row['Item No.']}: {row['item_description_expanded']}"
            for _, row in unique_items.iterrows()
        ])
        
        step1_prompt = f"""
Please read and understand these {len(unique_items)} pavement marking item descriptions:

{items_list}

For each item, tell me what you understand about it. What do you see in each description? Don't categorize yet, just tell me what you observe about each item.
"""
        
        step1_response = self._call_claude(step1_prompt)
        if not step1_response:
            print("  ❌ Failed to get Step 1 response from Claude")
            return None
        
        # Export Step 1 output
        step1_path = os.path.join(self.output_dir, 'step1_understanding.txt')
        with open(step1_path, 'w') as f:
            f.write("STEP 1: UNDERSTANDING ITEMS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Prompt:\n{step1_prompt}\n\n")
            f.write("=" * 50 + "\n")
            f.write(f"Claude's Response:\n{step1_response}\n")
        
        print(f"    ✅ Step 1 complete - exported to {step1_path}")
        return True

    def _step2_group_similar_items(self, unique_items):
        """
        Step 2: Group very similar items to ensure consistent parsing
        """
        print(f"  🎯 Step 2: Grouping similar items for consistent parsing...")
        
        step2_prompt = """📋 TASK: Group very similar TDOT pavement marking items to ensure consistent feature extraction.

🎯 OBJECTIVE: Identify groups of similar items that should be parsed consistently to prevent category explosion and parsing inconsistencies.

❗ PROBLEM: Currently similar items get parsed differently:
• "6 INCH LINE" might become width="6 INCH", type="LINE" 
• "4 INCH LINE" might become type="4 INCH LINE" (inconsistent!)

⚙️ APPROACH: Group items that are essentially the same type of marking with minor variations:
• Same marking type but different dimensions
• Same materials but different widths/colors
• Same function but different specifications

🔍 GROUPING CRITERIA:
1. FUNCTIONAL SIMILARITY: Same basic purpose/function
2. MATERIAL SIMILARITY: Same material type (thermoplastic, paint, etc.)
3. STRUCTURAL SIMILARITY: Same marking structure (line, symbol, etc.)
4. SPECIFICATION SIMILARITY: Similar technical specifications

📊 INPUT ITEMS:
"""
        
        # Add all unique items to prompt
        for _, row in unique_items.iterrows():
            item_no = row['Item No.']
            expanded_desc = row['item_description_expanded']
            step2_prompt += f"• {item_no}: {expanded_desc}\n"
            
        step2_prompt += """

📋 EXPECTED OUTPUT FORMAT:
For each group, provide:
```
GROUP N: [Group Name]
Description: [What makes these items similar]
Items:
• Item No.: Description
• Item No.: Description
...
Consistent parsing approach: [How these should be parsed consistently]
```

🎯 GOAL: Ensure similar items get consistent feature extraction to reduce categories and improve ML effectiveness.
"""
        
        # Make LLM call
        step2_response = self._call_claude(step2_prompt)
        if not step2_response:
            print(f"❌ Step 2 failed")
            return False
            
        # Export Step 2 response
        step2_file = os.path.join(self.output_dir, "step2_grouped_items.md")
        with open(step2_file, 'w') as f:
            f.write(step2_response)
        print(f"  💾 Step 2 exported: {step2_file}")
        
        return True
    
    def _step3_extract_properties(self, unique_items):
        """Step 3: Extract all properties from descriptions (with batching to prevent truncation)"""
        print(f"    STEP 3: Extracting properties...")
        
        # Process in batches to prevent JSON truncation
        batch_size = 30  # Process 30 items at a time to stay within token limits
        items_list = list(unique_items.iterrows())
        total_items = len(items_list)
        all_properties = []
        
        # Track property parsing patterns for consistency across batches
        accumulated_patterns = {}  # Track how similar terms were previously parsed
        
        print(f"    🔄 Processing {total_items} items in batches of {batch_size}...")
        
        for batch_start in range(0, total_items, batch_size):
            batch_end = min(batch_start + batch_size, total_items)
            batch_items = items_list[batch_start:batch_end]
            
            print(f"    📦 Processing batch {batch_start//batch_size + 1} ({batch_start+1}-{batch_end} of {total_items})")
            
            batch_list = "\n".join([
                f"{row['Item No.']}: {row['item_description_expanded']}"
                for _, row in batch_items
            ])
            
            # Build consistency context from previous batches
            consistency_context = ""
            if accumulated_patterns:
                consistency_context = f"""
CONSISTENCY EXAMPLES FROM PREVIOUS BATCHES:
Here are a few examples of how similar items were parsed:

{chr(10).join([f"• \"{desc}\" → {properties}" for desc, properties in list(accumulated_patterns.items())[-3:]])}

Parse similar items consistently.
"""
            
            batch_prompt = f"""
Extract ALL properties you can identify from these pavement marking descriptions.

{batch_list}
{consistency_context}

RULES:
1. Only extract words and properties that are LITERALLY written in each description
2. Do NOT add synonyms, technical terms, or your own interpretations  
3. Be consistent with similar items - if you see examples above, follow similar patterns
4. Keep measurements with their units together (like "4 INCH" not separate "4" and "INCH")

For each item, list ONLY the exact words, measurements, and characteristics that appear in the description text.

Return ONLY a JSON array with this exact format:
[
  {{
    "item_no": "{batch_items[0][1]['Item No.']}",
    "description": "original description text",
    "properties": ["exact_word1", "exact_word2", "exact_measurement"]
  }},
  ...
]

Use ONLY words that appear in the descriptions. Use the actual Item No. as shown above. Return only the JSON array, no other text.
"""
            
            batch_response = self._call_claude(batch_prompt)
            if not batch_response:
                print(f"    ❌ Failed to get response for batch {batch_start//batch_size + 1}")
                continue
                
            # Parse the batch response and add to all_properties
            try:
                batch_data = self._extract_json_from_response(batch_response)
                if batch_data and isinstance(batch_data, list):
                    all_properties.extend(batch_data)
                    print(f"    ✅ Extracted properties for {len(batch_data)} items")
                    
                    # Update accumulated patterns for next batch consistency
                    for item_data in batch_data:
                        desc = item_data.get('description', '').strip()
                        properties = item_data.get('properties', [])
                        
                        # Track actual descriptions and their parsed properties for consistency
                        # Store the last few similar items to provide as examples
                        if len(accumulated_patterns) < 20:  # Keep recent examples, not all
                            accumulated_patterns[desc] = properties
                            
                else:
                    print(f"    ⚠️  Batch response not a valid list, skipping batch {batch_start//batch_size + 1}")
            except Exception as e:
                print(f"    ⚠️  Error processing batch {batch_start//batch_size + 1}: {e}")
                continue
        
        # Combine all batches into final JSON
        final_json = json.dumps(all_properties, indent=2)
        
        # Export Step 2 output
        step3_path = os.path.join(self.output_dir, 'step3_properties.json')
        with open(step3_path, 'w') as f:
            f.write(final_json)
        
        step3_txt_path = os.path.join(self.output_dir, 'step3_properties.txt')
        with open(step3_txt_path, 'w') as f:
            f.write("STEP 3: EXTRACTING PROPERTIES (BATCHED PROCESSING)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total items processed: {total_items}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Total batches: {len(range(0, total_items, batch_size))}\n")
            f.write(f"Properties extracted for {len(all_properties)} items\n\n")
            f.write("=" * 60 + "\n")
            f.write(f"Combined Results:\n{final_json}\n")
        
        print(f"    ✅ Step 3 complete - processed {len(all_properties)} items in {len(range(0, total_items, batch_size))} batches")
        print(f"    📁 Exported to {step3_path} and {step3_txt_path}")
        return final_json
    
    def _step4_group_properties(self, step3_properties):
        """Step 4: Make global decisions about categories vs flags, then apply to batches"""
        print(f"    STEP 4: Making global categorization decisions...")
        
        # First, get the properties data
        try:
            # Clean step3 response if needed
            clean_step3 = step3_properties.strip()
            if clean_step3.startswith('```json'):
                clean_step3 = clean_step3[7:]
            if clean_step3.startswith('```'):
                clean_step3 = clean_step3[3:]
            if clean_step3.endswith('```'):
                clean_step3 = clean_step3[:-3]
            clean_step3 = clean_step3.strip()
            
            step3_data = json.loads(clean_step3)
        except json.JSONDecodeError as e:
            print(f"    ❌ Failed to parse Step 3 data: {e}")
            return None
        
        # STEP 4A: Global decision making - analyze ALL properties at once
        print(f"    🧠 Step 4A: Analyzing all properties globally for categorization decisions...")
        global_decisions = self._make_global_categorization_decisions(step3_data)
        if not global_decisions:
            return None
            
        # STEP 4B: Apply global decisions to items in batches
        print(f"    ⚙️  Step 4B: Applying global decisions to items...")
        final_result = self._apply_global_decisions_to_items(step3_data, global_decisions)
        
        step4_response = json.dumps(final_result, indent=2)
        
        # Export Step 4 output  
        step4_path = os.path.join(self.output_dir, 'step4_categorization.json')
        with open(step4_path, 'w') as f:
            f.write(step4_response)
        
        step4_txt_path = os.path.join(self.output_dir, 'step4_categorization.txt')
        with open(step4_txt_path, 'w') as f:
            f.write("STEP 4: GLOBAL CATEGORIZATION DECISIONS + BATCH APPLICATION\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Processed {len(step3_data)} items with global consistency\n\n")
            f.write(f"Final Result:\n{step4_response}\n")
        
        print(f"    ✅ Step 4 complete - global decisions applied consistently")
        print(f"    📄 Exported to {step4_path} and {step4_txt_path}")
        return step4_response
    
    def _step5_convert_to_csv(self, unique_items, step4_categorization):
        """Step 5: Convert categorizations to CSV"""
        print(f"    STEP 5: Converting to ML-ready CSV...")
        
        try:
            # Clean the JSON response - remove markdown code blocks if present
            clean_response = step4_categorization.strip()
            if clean_response.startswith('```json'):
                clean_response = clean_response[7:]
            if clean_response.startswith('```'):
                clean_response = clean_response[3:]
            if clean_response.endswith('```'):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            # Parse the combined JSON structure
            try:
                categorization_data = json.loads(clean_response)
            except json.JSONDecodeError as e:
                print(f"    ❌ JSON parsing failed: {e}")
                print(f"    Attempting to fix common JSON issues...")
                
                # Try to fix common issues
                # Remove any trailing commas
                import re
                clean_response = re.sub(r',(\s*[}\]])', r'\1', clean_response)
                
                # Try parsing again
                try:
                    categorization_data = json.loads(clean_response)
                    print(f"    ✅ JSON fixed and parsed successfully")
                except json.JSONDecodeError as e2:
                    print(f"    ❌ JSON still malformed after fixes: {e2}")
                    print(f"    First 200 chars: {clean_response[:200]}...")
                    print(f"    Last 200 chars: ...{clean_response[-200:]}")
                    return False

            # Extract item categorizations
            item_categorizations = categorization_data.get('item_categorizations', [])
            
            # Discover all unique categories and flags from all batches
            all_categories = set()
            all_flags = set()
            
            for item_cat in item_categorizations:
                categories = item_cat.get('categories', {})
                flags = item_cat.get('flags', {})
                
                all_categories.update(categories.keys())
                all_flags.update(flags.keys())
            
            print(f"    📊 Found {len(all_categories)} categories and {len(all_flags)} flags")
            print(f"    🎯 Categories: {sorted(all_categories)}")
            print(f"    🚩 Flags: {sorted(all_flags)}")
            
            # Standardize flag names to remove inconsistencies (is_painted vs painted)
            standardized_flags = set()
            flag_mapping = {}
            
            for flag in all_flags:
                # Remove is_ prefix for standardization
                if flag.startswith('is_'):
                    base_name = flag[3:]  # Remove 'is_' prefix
                else:
                    base_name = flag
                
                # Use the base name without is_ prefix
                standardized_flags.add(base_name)
                flag_mapping[flag] = base_name
            
            # Update item categorizations to use standardized flag names
            for item_cat in item_categorizations:
                old_flags = item_cat.get('flags', {})
                new_flags = {}
                for old_flag, value in old_flags.items():
                    new_flag = flag_mapping.get(old_flag, old_flag)
                    # If multiple versions of same flag exist, use OR logic
                    new_flags[new_flag] = new_flags.get(new_flag, False) or value
                item_cat['flags'] = new_flags
            
            # Update all_flags to use standardized names
            all_flags = standardized_flags
            print(f"    ✅ Standardized to {len(all_flags)} consistent flag names")
            
            # Create mapping from Item No. to categorizations
            item_cat_map = {}
            for item_cat in item_categorizations:
                item_no = item_cat.get('item_no')
                if item_no:
                    item_cat_map[item_no] = {
                        'categories': item_cat.get('categories', {}),
                        'flags': item_cat.get('flags', {})
                    }
            
            # Build CSV data
            ml_data = []
            
            for _, row in unique_items.iterrows():
                item_no = row['Item No.']
                original_desc = row['Item Description']
                expanded_desc = row['item_description_expanded']
                
                # Start with base data
                row_data = {
                    'item_no': item_no,
                    'item_description_original': original_desc,
                    'item_description_expanded': expanded_desc,
                }
                
                # Add columns for categories (with ml_ prefix)
                for cat_name in all_categories:
                    row_data[f'ml_{cat_name}'] = 'unknown'  # Default for categories (ML-friendly)
                
                # Add columns for flags (with ml_ prefix)  
                for flag_name in all_flags:
                    row_data[f'ml_{flag_name}'] = 0  # Default for flags (0 instead of False)
                
                # Fill in actual values from Claude's categorization
                if item_no in item_cat_map:
                    # Fill category values
                    categories = item_cat_map[item_no]['categories']
                    for cat_name, value in categories.items():
                        row_data[f'ml_{cat_name}'] = value
                    
                    # Fill flag values (convert boolean to 0/1)
                    flags = item_cat_map[item_no]['flags']
                    for flag_name, value in flags.items():
                        row_data[f'ml_{flag_name}'] = 1 if value else 0
                
                ml_data.append(row_data)
            
            # Create DataFrame and save
            ml_df = pd.DataFrame(ml_data)
            ml_csv_path = os.path.join(self.output_dir, 'ml_ready_716_features.csv')
            ml_df.to_csv(ml_csv_path, index=False)
            
            # Export final summary
            summary_path = os.path.join(self.output_dir, 'step4_final_summary.txt')
            with open(summary_path, 'w') as f:
                f.write("STEP 4: FINAL ML-READY CSV CONVERSION\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Categories discovered by Claude:\n")
                for cat_name in sorted(all_categories):
                    f.write(f"  ml_{cat_name}: categorical feature\n")
                f.write(f"\nBoolean flags discovered by Claude:\n") 
                for flag_name in sorted(all_flags):
                    f.write(f"  ml_{flag_name}: boolean flag\n")
                f.write(f"\nFinal CSV: {ml_csv_path}\n")
                f.write(f"Total items: {len(ml_df)}\n")
                f.write(f"Total columns: {len(ml_df.columns)}\n")
                f.write(f"ML columns: {[col for col in ml_df.columns if col.startswith('ml_')]}\n")
            
            print(f"  💾 ML-ready CSV saved: {ml_csv_path}")
            print(f"  📊 Total unique items: {len(ml_df)}")
            print(f"  📋 Columns created: {len(ml_df.columns)}")
            print(f"  🎯 Categories: {sorted(all_categories)}")
            print(f"  🚩 Flags: {sorted(all_flags)}")
            print(f"  📄 Summary exported: {summary_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Step 5 conversion failed: {e}")
            return False

    def _step6_remove_redundancy(self):
        """
        Step 6: Remove redundant columns using LLM analysis
        """
        print(f"  🧹 Step 6: Removing redundant columns...")
        
        # Load the CSV to analyze
        ml_csv_path = os.path.join(self.output_dir, 'ml_ready_716_features.csv')
        if not os.path.exists(ml_csv_path):
            print(f"❌ CSV file not found: {ml_csv_path}")
            return False
            
        df = pd.read_csv(ml_csv_path)
        
        # Get ML columns (excluding metadata columns)
        ml_columns = [col for col in df.columns if col.startswith('ml_')]
        
        # Sample some data to show column patterns
        sample_data = df[['item_no'] + ml_columns].head(20).to_string()
        
        step6_prompt = f"""📋 TASK: Identify and merge redundant columns in this TDOT pavement marking dataset.

🎯 OBJECTIVE: Remove duplicate/redundant columns that represent the same information to create a clean, non-redundant feature set for machine learning.

📊 CURRENT COLUMNS ({len(ml_columns)}):
{chr(10).join([f"• {col}" for col in ml_columns])}

📈 SAMPLE DATA (first 20 rows):
{sample_data}

🔍 ANALYSIS INSTRUCTIONS:
1. **IDENTIFY REDUNDANT PAIRS**: Look for columns that capture the same information
   - Similar names (flat_line vs flatline)  
   - Semantic overlap (line_width vs width, raised vs raised_marker)
   - Boolean flags that represent the same concept

2. **MERGE STRATEGY**: For each redundant group, pick the BEST column name to keep:
   - Most descriptive name
   - Most consistent with other naming patterns
   - Contains more complete information

3. **PRESERVE INFORMATION**: Ensure no unique information is lost during merging

📋 OUTPUT FORMAT: JSON with merge instructions
```json
{{
  "redundant_groups": [
    {{
      "group_description": "Description of what makes these redundant",
      "columns_to_merge": ["col1", "col2", "col3"],
      "keep_column": "best_column_name",
      "merge_logic": "how to combine the values (OR for flags, COALESCE for categories)"
    }}
  ],
  "columns_to_remove": ["list", "of", "columns", "to", "delete"],
  "final_column_count": 35,
  "reduction_summary": "Reduced from 40 to 35 columns by merging 5 redundant groups"
}}
```

🎯 GOAL: Create a clean, non-redundant feature set optimized for machine learning.

IMPORTANT: Return ONLY valid JSON, no explanations, no markdown, no extra text.
"""
        
        # Call LLM for redundancy analysis
        step6_response = self._call_claude(step6_prompt)
        if not step6_response:
            print(f"❌ Step 6 failed")
            return False
        
        # Parse the response
        try:
            clean_response = step6_response.strip()
            if clean_response.startswith('```json'):
                clean_response = clean_response[7:]
            if clean_response.startswith('```'):
                clean_response = clean_response[3:]
            if clean_response.endswith('```'):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            # Debug: check if response is empty or malformed
            if not clean_response:
                print(f"❌ Empty response after cleaning markdown")
                print(f"    Original response length: {len(step6_response)}")
                print(f"    Original response preview: {step6_response[:200]}...")
                return False
            
            merge_instructions = json.loads(clean_response)
            
            # Apply the merge instructions
            df_cleaned = self._apply_column_merges(df, merge_instructions)
            
            # Save cleaned CSV
            cleaned_csv_path = os.path.join(self.output_dir, 'ml_ready_716_features_clean.csv')
            df_cleaned.to_csv(cleaned_csv_path, index=False)
            
            # Export Step 6 analysis
            step6_file = os.path.join(self.output_dir, 'step6_redundancy_analysis.json')
            with open(step6_file, 'w') as f:
                json.dump(merge_instructions, f, indent=2)
            
            print(f"  🧹 {merge_instructions.get('reduction_summary', 'Redundancy analysis complete')}")
            print(f"  💾 Clean CSV saved: {cleaned_csv_path}")
            print(f"  📋 Analysis exported: {step6_file}")
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse redundancy analysis: {e}")
            print(f"    Response length: {len(clean_response)}")
            print(f"    Response preview: {clean_response[:500]}...")
            
            # Try to save the raw response for debugging
            debug_file = os.path.join(self.output_dir, 'step6_debug_response.txt')
            with open(debug_file, 'w') as f:
                f.write("ORIGINAL RESPONSE:\n")
                f.write("=" * 50 + "\n")
                f.write(step6_response)
                f.write("\n\nCLEANED RESPONSE:\n")
                f.write("=" * 50 + "\n")
                f.write(clean_response)
            print(f"    🐛 Debug response saved: {debug_file}")
            return False
            return False
        except Exception as e:
            print(f"❌ Error applying column merges: {e}")
            return False

    def _apply_column_merges(self, df, merge_instructions):
        """Apply the LLM-suggested column merges"""
        df_result = df.copy()
        
        for group in merge_instructions.get('redundant_groups', []):
            columns_to_merge = group['columns_to_merge']
            keep_column = group['keep_column']
            merge_logic = group.get('merge_logic', 'OR')
            
            print(f"    🔄 Merging {columns_to_merge} → {keep_column} ({merge_logic})")
            
            # Check if columns exist
            existing_cols = [col for col in columns_to_merge if col in df_result.columns]
            if not existing_cols:
                continue
                
            if merge_logic.upper() == 'OR':
                # For boolean flags: OR logic (1 if any is 1)
                df_result[keep_column] = df_result[existing_cols].max(axis=1)
            elif merge_logic.upper() == 'COALESCE':
                # For categories: Take first non-unknown value
                df_result[keep_column] = df_result[existing_cols].apply(
                    lambda row: next((val for val in row if val != 'unknown'), 'unknown'), axis=1
                )
            else:
                # Default: take the first column's values
                df_result[keep_column] = df_result[existing_cols[0]]
        
        # Remove redundant columns
        columns_to_remove = merge_instructions.get('columns_to_remove', [])
        columns_to_remove = [col for col in columns_to_remove if col in df_result.columns]
        
        if columns_to_remove:
            df_result = df_result.drop(columns=columns_to_remove)
            print(f"    🗑️ Removed {len(columns_to_remove)} redundant columns")
        
        return df_result            

    def _make_global_categorization_decisions(self, step3_data):
        """Make global decisions about which properties become categories vs flags"""
        
        # Collect ALL unique properties across ALL items
        all_properties = set()
        property_frequency = {}
        property_examples = {}
        
        for item_data in step3_data:
            properties = item_data.get('properties', [])
            item_desc = item_data.get('description', '')
            
            for prop in properties:
                prop_clean = str(prop).strip()
                if prop_clean:
                    all_properties.add(prop_clean)
                    property_frequency[prop_clean] = property_frequency.get(prop_clean, 0) + 1
                    if prop_clean not in property_examples:
                        property_examples[prop_clean] = []
                    if len(property_examples[prop_clean]) < 3:  # Keep up to 3 examples
                        property_examples[prop_clean].append(item_desc)
        
        print(f"      📊 Found {len(all_properties)} unique properties across {len(step3_data)} items")
        
        # Filter out properties that appear only once - they're not useful
        frequent_properties = {prop: freq for prop, freq in property_frequency.items() if freq > 1}
        ignored_single = set(all_properties) - set(frequent_properties.keys())
        
        print(f"      🗑️ Ignoring {len(ignored_single)} properties that appear only once")
        print(f"      📊 Analyzing {len(frequent_properties)} properties that appear 2+ times")
        
        # Build comprehensive analysis for LLM (only frequent properties)
        property_analysis = []
        for prop in sorted(frequent_properties.keys()):
            freq = frequent_properties[prop]
            examples = property_examples[prop][:2]  # Show 2 examples max
            property_analysis.append(f"• \"{prop}\" (appears in {freq} items) - Examples: {examples}")
        
        global_prompt = f"""🧠 GLOBAL CATEGORIZATION DECISION TASK

You are analyzing ALL properties from {len(step3_data)} TDOT pavement marking items to make consistent categorization decisions.

📊 ALL PROPERTIES FOUND ACROSS ALL ITEMS:
{chr(10).join(property_analysis)}

🎯 YOUR TASK: Look at these properties and decide which ones should be:
1. **CATEGORIES** (properties that have multiple specific values that classify items differently)
2. **FLAGS** (properties that are either present or absent - true/false)  
3. **IGNORE** (properties that aren't useful for distinguishing items)

🧠 DECISION LOGIC:
- If a property appears with different specific values across items → make it a category
- If a property is just present/absent as a descriptive word → make it a flag
- If a property is too rare or doesn't help distinguish items → ignore it

🎯 PRIORITIZE BY IMPORTANCE:
Focus on properties that are MOST useful for distinguishing between different types of pavement markings.
- High priority: Core characteristics (materials, dimensions, marking types)
- Medium priority: Important modifiers (colors, application methods)  
- Low priority: Descriptive attributes that appear frequently
- Ignore: Rare properties, redundant terms, or properties that don't meaningfully distinguish items

BE SELECTIVE - create fewer, more meaningful features rather than many weak ones.

🚫 AVOID DUPLICATE CATEGORIES:
Do NOT create multiple categories for the same concept with slightly different names:
- Don't create both "marker_type" and "marking_type" - pick ONE name
- Don't create both "material" and "materials" - pick ONE name  
- Don't create both "line_type" and "marking_type" - consolidate into ONE category
- Use consistent naming - if you use "type" for one category, use it consistently

Simple examples:
- "THERMOPLASTIC" vs "PLASTIC" vs "PAINTED" → material category
- "4 INCH" vs "8 INCH" vs "12 INCH" → width category  
- "CONTRAST" (present or not) → contrast flag
- "ENHANCED" (present or not) → enhanced flag

Look at the actual data patterns and frequencies. Use your judgment based on what you see.

Return your decisions in this JSON format:
{{
  "categories": {{
    "category_name1": ["value1", "value2", "value3"],
    "category_name2": ["valueA", "valueB"]
  }},
  "flags": ["flag1", "flag2", "flag3"],
  "ignored": ["ignored1", "ignored2"]
}}

Create whatever category names and flag names make sense for the actual data. Return ONLY JSON.
"""
        
        global_response = self._call_claude(global_prompt)
        if not global_response:
            print(f"    ❌ Failed to get global categorization decisions")
            return None
            
        # Parse response
        try:
            clean_response = global_response.strip()
            if clean_response.startswith('```json'):
                clean_response = clean_response[7:]
            if clean_response.startswith('```'):
                clean_response = clean_response[3:]
            if clean_response.endswith('```'):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            global_decisions = json.loads(clean_response)
            
            # Export global decisions
            decisions_path = os.path.join(self.output_dir, 'step4a_global_decisions.json')
            with open(decisions_path, 'w') as f:
                json.dump(global_decisions, f, indent=2)
            
            print(f"      ✅ Global decisions made:")
            print(f"         Categories: {len(global_decisions.get('categories', {}))}")
            print(f"         Flags: {len(global_decisions.get('flags', []))}")
            print(f"         Ignored: {len(global_decisions.get('ignored', []))}")
            print(f"      📁 Exported: {decisions_path}")
            
            return global_decisions
            
        except json.JSONDecodeError as e:
            print(f"    ❌ Failed to parse global decisions: {e}")
            return None

    def _apply_global_decisions_to_items(self, step3_data, global_decisions):
        """Apply the global categorization decisions to all items"""
        
        categories_schema = global_decisions.get('categories', {})
        flags_list = global_decisions.get('flags', [])
        
        print(f"      ⚙️  Applying decisions to {len(step3_data)} items...")
        
        all_item_categorizations = []
        
        for item_data in step3_data:
            item_no = item_data.get('item_no')
            description = item_data.get('description', '')
            properties = item_data.get('properties', [])
            
            # Initialize item categorization
            item_result = {
                'item_no': item_no,
                'description': description,
                'categories': {},
                'flags': {}
            }
            
            # Apply categories
            for cat_name, possible_values in categories_schema.items():
                for prop in properties:
                    if prop in possible_values:
                        item_result['categories'][cat_name] = prop
                        break
                if cat_name not in item_result['categories']:
                    item_result['categories'][cat_name] = 'unknown'
            
            # Apply flags
            for flag_name in flags_list:
                # Check if any property matches this flag (case insensitive)
                flag_present = False
                for prop in properties:
                    if str(prop).upper() == flag_name.upper():
                        flag_present = True
                        break
                item_result['flags'][flag_name] = flag_present
            
            all_item_categorizations.append(item_result)
        
        return {"item_categorizations": all_item_categorizations}


    def _extract_json_from_response(self, response_text):
        """Extract JSON from Claude's response, handling cases where JSON is embedded in text"""
        if not response_text:
            return None
            
        # Try to parse the entire response as JSON first
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Look for JSON array or object in the response
        import re
        
        # Try to find JSON array
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        print(f"    ⚠️  Could not extract valid JSON from response: {response_text[:200]}...")
        return None

    def _call_claude(self, prompt: str):
        """Call Claude API"""
        try:
            print(f"    🔄 Sending request to Claude (prompt length: {len(prompt):,} chars)...")
            response = self.llm_client.messages.create(
                model=self.model_name,
                max_tokens=12000,  # Increased for longer JSON responses
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.content[0].text.strip()
            print(f"    ✅ Received response ({len(response_text):,} chars)")
            return response_text
        except Exception as e:
            print(f"    ❌ Claude API error: {e}")
            return None


def main():
    """Main function to run 4-step LLM feature extraction"""
    print("🧪 TDOT LLM Feature Extraction - Version 3 (4-Step Clean)")
    print("=" * 60)
    
    # Initialize extractor  
    extractor = TDOTLLMFeatureExtractor()
    
    # Load data
    if not extractor.load_data():
        return False
    
    # Run 4-step extraction on unique items
    success = extractor.extract_features_4_step()
    
    if success:
        print("\n🎉 4-Step LLM feature extraction completed!")
        return True
    else:
        print("\n❌ Feature extraction failed")
        return False


if __name__ == "__main__":
    main()