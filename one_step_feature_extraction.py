#!/usr/bin/env python3
"""
One-Step LLM Feature Extraction for 790.01-790.19 Traffic Control Items
Combines data loading, LLM analysis, and ML-ready feature generation in one step
"""

import pandas as pd
import numpy as np
import json
import os
import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
import anthropic


class OneStepFeatureExtractor:
    def __init__(self, 
                 data_file: str = 'Data/TDOT_data.csv',
                 output_file: str = 'ml_ready_790_features_onestep.csv',
                 api_key: str = None):
        """
        Initialize the one-step feature extractor for 790 traffic control items
        
        Args:
            data_file: Path to TDOT bidding data
            output_file: Output path for ML-ready features
            api_key: Anthropic API key (will try environment if not provided)
        """
        self.data_file = data_file
        self.output_file = output_file
        
        # Initialize Anthropic client
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            try:
                self.client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
            except Exception as e:
                print(f"⚠️  Warning: Could not initialize Anthropic client: {e}")
                print("   LLM extraction will be skipped")
                self.client = None
        
        # Global decision storage
        self.global_decisions = {
            'marking_types': {},
            'materials': {},
            'widths': {},
            'thicknesses': {},
            'colors': {},
            'arrow_types': {},
            'directions': {},
            'flag_patterns': {}
        }
    
    def load_790_data(self) -> pd.DataFrame:
        """Load and filter 790.01-790.19 traffic control data"""
        print("📁 Loading TDOT bidding data...")
        df = pd.read_csv(self.data_file, encoding='latin-1')
        print(f"   Total records: {len(df):,}")
        
        # Filter for 790.01-790.19 traffic control items
        pattern = r'^790\.(0[1-9]|1[0-9])$'
        df_790 = df[df['Item No.'].str.match(pattern, na=False)].copy()
        print(f"   790.01-790.19 records: {len(df_790):,}")
        
        # Get unique items for analysis
        unique_items = df_790[['Item No.', 'Item Description']].drop_duplicates()
        print(f"   Unique 790 items: {len(unique_items)}")
        
        return df_790, unique_items
    
    def extract_global_patterns(self, unique_items: pd.DataFrame) -> Dict[str, Any]:
        """Extract global patterns from all 790.01-790.19 items using LLM from bid estimator perspective"""
        if not self.client:
            print("⚠️  Skipping LLM extraction - no API client")
            return self._create_dummy_features(unique_items)
        
        print("\n🧠 Analyzing 790 traffic control item names for bid-relevant properties...")
        
        # Prepare items for LLM analysis
        items_text = []
        for _, row in unique_items.iterrows():
            items_text.append(f"{row['Item No.']}: {row['Item Description']}")
        
        # Split into chunks for API limits
        chunk_size = 50
        chunks = [items_text[i:i + chunk_size] for i in range(0, len(items_text), chunk_size)]
        
        all_extracted_features = {}
        
        for i, chunk in enumerate(chunks):
            print(f"   Processing chunk {i+1}/{len(chunks)} ({len(chunk)} items)...")
            
            chunk_text = "\n".join(chunk)
            
            prompt = f"""You are a bid price estimator working for the Tennessee Department of Transportation (TDOT). Your job is to analyze construction item names and extract the key properties that affect bid pricing.

ITEMS TO ANALYZE:
{chunk_text}

Read each item number and name carefully. The item name contains all the information you need - do not make assumptions beyond what is explicitly stated in the name.

For each item, first identify:
1. THE MAIN THING - What is this item fundamentally about? (e.g., "SIGN", "BARRIER", "MARKER", etc.)

Then extract:
2. CATEGORICAL FEATURES - Different categories/types/specifications mentioned in the names
3. BINARY FLAGS - Yes/no properties that are clearly stated or implied in the names

Focus only on information that is clearly present in the item names themselves. If a property isn't mentioned in the name, don't assume it exists.

Return ONLY valid JSON with this structure:
{{
  "790.XX": {{
    "main_item_type": "what_this_item_fundamentally_is",
    "category_feature_1": "value_from_name",
    "category_feature_2": "value_from_name",
    "category_feature_N": "value_from_name",
    "flag_feature_1": 0,
    "flag_feature_2": 0,
    "flag_feature_N": 0
  }}
}}

Extract only what is explicitly mentioned in the item names. Use "NOT_SPECIFIED" for categorical features not mentioned in the name, and 0 for binary flags not clearly indicated."""

            try:
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Parse JSON response
                response_text = response.content[0].text.strip()
                
                # Clean response to extract JSON
                if '```json' in response_text:
                    json_start = response_text.find('```json') + 7
                    json_end = response_text.find('```', json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif '{' in response_text:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    response_text = response_text[json_start:json_end]
                
                chunk_features = json.loads(response_text)
                all_extracted_features.update(chunk_features)
                
            except Exception as e:
                print(f"   ⚠️  Error processing chunk {i+1}: {e}")
                continue
        
        print(f"✅ Successfully extracted features for {len(all_extracted_features)} items")
        return all_extracted_features
    
    def _create_dummy_features(self, unique_items: pd.DataFrame) -> Dict[str, Any]:
        """Create dummy features when LLM is not available"""
        print("   Creating dummy features...")
        dummy_features = {}
        
        for _, row in unique_items.iterrows():
            item_no = row['Item No.']
            dummy_features[item_no] = {
                'main_item_type': 'NOT_SPECIFIED',
                'size_specification': 'NOT_SPECIFIED',
                'material_specification': 'NOT_SPECIFIED',
                'color_specification': 'NOT_SPECIFIED',
                'has_size_mentioned': 0,
                'has_material_mentioned': 0,
                'has_color_mentioned': 0,
                'has_special_feature': 0
            }
        
        return dummy_features
    
    def create_ml_features(self, df_790: pd.DataFrame, extracted_features: Dict[str, Any]) -> pd.DataFrame:
        """Create ML-ready dataset by merging extracted features with bidding data"""
        print("\n📊 Creating ML-ready dataset...")
        
        # Create feature mapping DataFrame
        feature_rows = []
        for item_no, features in extracted_features.items():
            row = {'Item No.': item_no}
            
            # Add all features with ml_ prefix (LLM will determine feature names)
            for feature_name, feature_value in features.items():
                row[f'ml_{feature_name}'] = feature_value
            
            feature_rows.append(row)
        
        feature_df = pd.DataFrame(feature_rows)
        print(f"   Created feature mapping: {len(feature_df)} items")
        
        # Merge with bidding data
        print("   Merging with bidding data...")
        merged_df = df_790.merge(feature_df, on='Item No.', how='left')
        
        # Dynamically determine categorical vs numerical columns
        ml_columns = [col for col in merged_df.columns if col.startswith('ml_')]
        categorical_cols = []
        numerical_cols = []
        
        for col in ml_columns:
            if merged_df[col].dtype == 'object':
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        
        # Fill missing values
        for col in categorical_cols:
            merged_df[col] = merged_df[col].fillna('OTHER')
        
        for col in numerical_cols:
            merged_df[col] = merged_df[col].fillna(0)
        
        print(f"   Final ML dataset: {len(merged_df):,} records")
        
        # Find the first categorical column for coverage check
        first_cat_col = categorical_cols[0] if categorical_cols else None
        if first_cat_col:
            coverage_count = len(merged_df[merged_df[first_cat_col].notna()])
            print(f"   Feature coverage: {coverage_count:,} records ({coverage_count/len(merged_df)*100:.1f}%)")
        
        return merged_df
    
    def display_feature_summary(self, ml_df: pd.DataFrame):
        """Display summary of extracted features"""
        print("\n📈 Feature Extraction Summary:")
        print("="*50)
        
        # Categorical features
        categorical_cols = [col for col in ml_df.columns if col.startswith('ml_') and 
                          ml_df[col].dtype == 'object']
        
        print("\n🏷️  CATEGORICAL FEATURES:")
        for col in sorted(categorical_cols):
            unique_vals = ml_df[col].nunique()
            top_val = ml_df[col].mode().iloc[0] if len(ml_df[col].mode()) > 0 else 'N/A'
            print(f"   {col:<25} {unique_vals:>3} categories (top: {top_val})")
        
        # Numerical features (flags)
        numerical_cols = [col for col in ml_df.columns if col.startswith('ml_') and 
                         ml_df[col].dtype in ['int64', 'float64']]
        
        print(f"\n🔢 NUMERICAL FEATURES (FLAGS):")
        for col in sorted(numerical_cols):
            pos_count = (ml_df[col] == 1).sum()
            pos_pct = pos_count / len(ml_df) * 100
            print(f"   {col:<25} {pos_count:>6,} positive ({pos_pct:>5.1f}%)")
        
        print(f"\n✅ Total features created: {len(categorical_cols + numerical_cols)}")
        print(f"   Records processed: {len(ml_df):,}")
        print(f"   Unique 716 items: {ml_df['Item No.'].nunique()}")
    
    def save_results(self, ml_df: pd.DataFrame):
        """Save the ML-ready dataset"""
        print(f"\n💾 Saving results to {self.output_file}...")
        ml_df.to_csv(self.output_file, index=False)
        print(f"✅ Saved {len(ml_df):,} records with ML features")
        
        # Save feature summary
        summary_file = self.output_file.replace('.csv', '_summary.json')
        
        categorical_cols = [col for col in ml_df.columns if col.startswith('ml_') and 
                          ml_df[col].dtype == 'object']
        numerical_cols = [col for col in ml_df.columns if col.startswith('ml_') and 
                         ml_df[col].dtype in ['int64', 'float64']]
        
        summary = {
            'extraction_stats': {
                'total_records': len(ml_df),
                'unique_items': int(ml_df['Item No.'].nunique()),
                'feature_coverage': f"{len(ml_df[ml_df['ml_marking_type'].notna()])/len(ml_df)*100:.1f}%"
            },
            'categorical_features': {
                col: {
                    'unique_values': int(ml_df[col].nunique()),
                    'top_categories': ml_df[col].value_counts().head(5).to_dict()
                } for col in categorical_cols
            },
            'numerical_features': {
                col: {
                    'positive_count': int((ml_df[col] == 1).sum()),
                    'positive_percentage': f"{(ml_df[col] == 1).sum()/len(ml_df)*100:.1f}%"
                } for col in numerical_cols
            }
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Feature summary saved to {summary_file}")
    
    def run_extraction(self):
        """Run the complete one-step feature extraction process"""
        print("\n" + "="*80)
        print("🚀 ONE-STEP LLM FEATURE EXTRACTION FOR 790 TRAFFIC CONTROL ITEMS")
        print("="*80)
        
        # Step 1: Load data
        df_790, unique_items = self.load_790_data()
        
        # Step 2: Extract features using LLM
        extracted_features = self.extract_global_patterns(unique_items)
        
        # Step 3: Create ML dataset
        ml_df = self.create_ml_features(df_790, extracted_features)
        
        # Step 4: Display summary
        self.display_feature_summary(ml_df)
        
        # Step 5: Save results
        self.save_results(ml_df)
        
        print("\n" + "="*80)
        print("✅ ONE-STEP FEATURE EXTRACTION COMPLETED!")
        print(f"📁 Output file: {self.output_file}")
        print("="*80)
        
        return ml_df


def main():
    """Main execution function"""
    # Initialize extractor
    extractor = OneStepFeatureExtractor(
        data_file='Data/TDOT_data.csv',
        output_file='ml_ready_790_features_onestep.csv'
    )
    
    # Run extraction
    result_df = extractor.run_extraction()
    
    return result_df


if __name__ == "__main__":
    result = main()