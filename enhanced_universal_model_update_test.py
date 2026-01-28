#!/usr/bin/env python3
"""
Enhanced Universal TDOT Construction Price Prediction Model
Incorporates item classification data (Item Class, Item Type) with MPNet embeddings
Class-based approach for better organization and maintainability
"""

import pandas as pd
import numpy as np
import re
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Tuple

class EnhancedUniversalTDOTModel:
    """
    Enhanced Universal TDOT Construction Price Prediction Model
    Combines item classification data with semantic embeddings and proven methodology
    """
    
    def __init__(self, 
                 data_file: str = 'Data/TDOT_data.csv',
                 item_lists_dir: str = 'Data/Item Lists',
                 corrections_config_file: str = 'description_corrections.json',
                 hidden_items_config_file: str = 'hidden_items_config.json',
                 supplemental_classifications_file: str = 'supplemental_item_classifications.json',
                 output_dir: str = 'output'):
        """
        Initialize the enhanced model with configuration paths
        
        Args:
            data_file: Path to main TDOT bidding data CSV
            item_lists_dir: Directory containing item classification CSV files
            corrections_config_file: JSON file with description correction rules
            hidden_items_config_file: JSON file with data filtering rules
            supplemental_classifications_file: JSON file with AI-suggested classifications for missing items
            output_dir: Directory to save model and results
        """
        self.data_file = data_file
        self.item_lists_dir = item_lists_dir
        self.corrections_config_file = corrections_config_file
        self.hidden_items_config_file = hidden_items_config_file
        self.supplemental_classifications_file = supplemental_classifications_file
        self.output_dir = output_dir
        
        # Initialize internal state
        self.item_classification_mapping = {}
        self.supplemental_classifications = {}
        self.corrections_config = None
        self.hidden_items_config = None
        self.model = None
        self.feature_names = []
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def load_configurations(self) -> None:
        """Load all configuration files"""
        print("Loading configuration files...")
        
        # Load description corrections
        self.corrections_config = self._load_description_corrections()
        print(f"  ✅ Description corrections loaded")
        
        # Load hidden items configuration
        self.hidden_items_config = self._load_hidden_items_config()
        print(f"  ✅ Hidden items filtering rules loaded")
        
        # Load supplemental classifications
        self.supplemental_classifications = self._load_supplemental_classifications()
        print(f"  ✅ Supplemental classifications loaded")
    
    def _load_description_corrections(self) -> Dict:
        """Load description correction rules from JSON configuration"""
        try:
            with open(self.corrections_config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {self.corrections_config_file} not found, using basic corrections")
            return {
                "description_corrections": {
                    "abbreviations": {
                        "PVMT": "PAVEMENT", "MRKNG": "MARKING", "THERMO": "THERMOPLASTIC"
                    },
                    "regex_patterns": [
                        {"pattern": r"\b(\d+)\s*IN\b", "replacement": r'\1"', "description": "inches"}
                    ]
                },
                "processing_order": ["abbreviations", "regex_patterns"]
            }
    
    def _load_hidden_items_config(self) -> Dict:
        """Load hidden items configuration for filtering training data"""
        try:
            with open(self.hidden_items_config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {self.hidden_items_config_file} not found, using default filtering")
            return {
                "filtering_rules": {
                    "716_pavement_markings": {
                        "exclude_patterns": [{"pattern": "716-99.*", "reason": "Generic items"}],
                        "exclude_specific_items": [
                            {"items": [f"716-08.{i:02d}" for i in list(range(31, 35)) + list(range(11, 18))], 
                             "reason": "Removal items"}
                        ]
                    }
                }
            }
    
    def _load_supplemental_classifications(self) -> Dict:
        """Load supplemental item classifications for missing items"""
        try:
            with open(self.supplemental_classifications_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {self.supplemental_classifications_file} not found, no supplemental classifications available")
            return {"supplemental_classifications": {"items": {}}}
    
    def load_item_classifications(self) -> Dict[str, Dict[str, str]]:
        """
        Load item classification data from CSV files
        Focus on sections 3, 4, and 7 as requested
        Keep all 300s and 400s, only 716 from 700s
        
        Returns:
            Dict mapping item numbers to classification info
        """
        print("\n=== LOADING ITEM CLASSIFICATION DATA ===")
        
        # Define which files to process and which categories to keep
        target_files = {
            'section3_base_and_subgrade_treatments.csv': {'keep_categories': ['302', '303']},  # All 300s
            'section4_flexible_surfaces.csv': {'keep_categories': ['401', '402', '403', '404', '405', '406', '407', '408', '409']},  # All 400s
            'section7_incidemtal_construction_and_services.csv': {'keep_categories': ['716']}  # Only 716 from 700s
        }
        
        classification_mapping = {}
        total_items_loaded = 0
        
        for filename, config in target_files.items():
            filepath = os.path.join(self.item_lists_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"  ⚠️  File not found: {filename}")
                continue
                
            print(f"\n  Processing {filename}:")
            
            try:
                df = pd.read_csv(filepath)
                print(f"    Loaded {len(df)} items from file")
                
                # Filter for target categories
                keep_categories = config['keep_categories']
                category_pattern = '|'.join(f'^{cat}' for cat in keep_categories)
                
                filtered_df = df[df['Item Number'].str.match(category_pattern, na=False)]
                print(f"    Filtered to {len(filtered_df)} items matching categories: {keep_categories}")
                
                # Create mapping
                for _, row in filtered_df.iterrows():
                    item_number = str(row['Item Number']).strip()
                    classification_mapping[item_number] = {
                        'item_class': str(row['Item Class']).strip() if pd.notna(row['Item Class']) else 'Unknown',
                        'item_type': str(row['Item Type']).strip() if pd.notna(row['Item Type']) else 'Unknown',
                        'um_type': str(row['U/M Type']).strip() if 'U/M Type' in row else 'Unknown',
                        'source_section': filename.replace('.csv', '').replace('section', 'Section ')
                    }
                
                total_items_loaded += len(filtered_df)
                print(f"    Added {len(filtered_df)} item classifications to mapping")
                
                # Show sample of categories found
                categories_found = filtered_df['Item Number'].str.extract(r'^(\d{3})')[0].unique()
                print(f"    Categories found: {sorted(categories_found)}")
                
            except Exception as e:
                print(f"    ❌ Error processing {filename}: {e}")
        
        print(f"\n  📊 Total item classifications loaded: {total_items_loaded:,}")
        
        # Show classification summary
        if classification_mapping:
            all_classes = [info['item_class'] for info in classification_mapping.values()]
            all_types = [info['item_type'] for info in classification_mapping.values()]
            
            print(f"\n  Item Classes found:")
            class_counts = pd.Series(all_classes).value_counts()
            for class_name, count in class_counts.head(10).items():
                print(f"    {class_name}: {count} items")
            
            print(f"\n  Item Types found:")
            type_counts = pd.Series(all_types).value_counts()
            for type_name, count in type_counts.head(10).items():
                print(f"    {type_name}: {count} items")
        
        self.item_classification_mapping = classification_mapping
        
        # Add supplemental classifications
        supplemental_items = self.supplemental_classifications.get('supplemental_classifications', {}).get('items', {})
        added_count = 0
        for item_number, classification in supplemental_items.items():
            if item_number not in classification_mapping:
                classification_mapping[item_number] = {
                    'item_class': classification['item_class'],
                    'item_type': classification['item_type'],
                    'um_type': 'Unknown',
                    'source_section': 'AI Supplemental'
                }
                added_count += 1
        
        if added_count > 0:
            print(f"\n  🤖 Added {added_count} AI-suggested classifications from supplemental file")
            print(f"  📊 Total classifications available: {len(classification_mapping):,}")
        
        return classification_mapping
    
    def load_project_data(self) -> pd.DataFrame:
        """
        Load project description data from categorized bid tabs CSV
        
        Returns:
            DataFrame with project-level information
        """
        print("\n=== LOADING PROJECT DATA ===")
        
        project_file = 'output/bid_tabs_combined_categorized.csv'
        
        try:
            df = pd.read_csv(project_file)
            print(f"  📁 Loaded {len(df)} projects from {project_file}")
            print(f"  📊 Unique contracts: {df['contract_number'].nunique()}")
            
            # Show work type distribution
            print(f"\n  Work type distribution:")
            work_type_counts = df['work_type'].value_counts()
            for work_type, count in work_type_counts.head(8).items():
                print(f"    {work_type}: {count}")
            
            # Show special flags
            on_call_count = df['is_on_call'].sum()
            emergency_count = df['is_emergency'].sum()
            print(f"\n  Special flags:")
            print(f"    On-call projects: {on_call_count}")
            print(f"    Emergency projects: {emergency_count}")
            
            return df
            
        except FileNotFoundError:
            print(f"  ⚠️  Project file not found: {project_file}")
            print(f"     Using empty project data")
            return pd.DataFrame(columns=['contract_number', 'work_type', 'project_description', 
                                       'project_length', 'is_on_call', 'is_emergency'])
        except Exception as e:
            print(f"  ❌ Error loading project data: {e}")
            return pd.DataFrame(columns=['contract_number', 'work_type', 'project_description',
                                       'project_length', 'is_on_call', 'is_emergency'])
    
    def merge_bidding_with_classifications(self, bidding_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge bidding data with item classification information
        
        Args:
            bidding_df: DataFrame with bidding data containing 'Item No.' column
            
        Returns:
            Enhanced DataFrame with item classification columns added
        """
        print(f"\n=== MERGING BIDDING DATA WITH ITEM CLASSIFICATIONS ===")
        print(f"Bidding data records: {len(bidding_df):,}")
        print(f"Available classifications: {len(self.item_classification_mapping):,}")
        
        # Create classification columns
        bidding_df['item_class'] = 'Unknown'
        bidding_df['item_type'] = 'Unknown'
        bidding_df['classification_source'] = 'Not Found'
        bidding_df['has_classification'] = False
        
        # Match item numbers with classifications
        matched_count = 0
        for idx, row in bidding_df.iterrows():
            item_no = str(row['Item No.']).strip()
            
            if item_no in self.item_classification_mapping:
                classification = self.item_classification_mapping[item_no]
                bidding_df.at[idx, 'item_class'] = classification['item_class']
                bidding_df.at[idx, 'item_type'] = classification['item_type'] 
                bidding_df.at[idx, 'classification_source'] = classification['source_section']
                bidding_df.at[idx, 'has_classification'] = True
                matched_count += 1
        
        match_rate = (matched_count / len(bidding_df)) * 100
        print(f"Matched classifications: {matched_count:,} ({match_rate:.1f}%)")
        
        # Show classification distribution in bidding data
        print(f"\nClassification distribution in bidding data:")
        class_dist = bidding_df[bidding_df['has_classification']]['item_class'].value_counts()
        for class_name, count in class_dist.head(10).items():
            print(f"  {class_name}: {count:,} bidding records")
        
        return bidding_df
    
    def merge_bidding_with_project_data(self, bidding_df: pd.DataFrame, project_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge bidding data with project information
        
        Args:
            bidding_df: DataFrame with bidding data
            project_df: DataFrame with project data from bid tabs
            
        Returns:
            Enhanced DataFrame with project features added
        """
        print(f"\n=== MERGING BIDDING DATA WITH PROJECT DATA ===")
        
        if len(project_df) == 0:
            print("  ⚠️  No project data available, adding empty columns")
            # Add empty columns
            bidding_df['work_type'] = 'Unknown'
            bidding_df['project_length'] = 0.0
            bidding_df['is_on_call'] = 0
            bidding_df['is_emergency'] = 0
            bidding_df['has_project_data'] = 0
            return bidding_df
        
        original_size = len(bidding_df)
        
        # Merge on Proposal ID = contract_number
        print(f"  Merging {len(bidding_df):,} bidding records with {len(project_df):,} project records")
        
        # Select relevant project columns (including project_description for embeddings)
        project_cols = ['contract_number', 'work_type', 'project_description', 'project_length', 'is_on_call', 'is_emergency']
        project_subset = project_df[project_cols].copy()
        
        # Merge (left join to keep all bidding records)
        merged_df = bidding_df.merge(project_subset, 
                                   left_on='Proposal ID', 
                                   right_on='contract_number', 
                                   how='left')
        
        # Fill missing project data
        merged_df['work_type'] = merged_df['work_type'].fillna('Unknown')
        merged_df['project_description'] = merged_df['project_description'].fillna('Unknown Project')
        merged_df['project_length'] = merged_df['project_length'].fillna(0.0)
        merged_df['is_on_call'] = merged_df['is_on_call'].fillna(0).astype(int)
        merged_df['is_emergency'] = merged_df['is_emergency'].fillna(0).astype(int)
        
        # Add flag indicating if project data was available
        merged_df['has_project_data'] = (~merged_df['contract_number'].isna()).astype(int)
        
        # Drop the redundant contract_number column
        merged_df = merged_df.drop('contract_number', axis=1)
        
        # Show merge results
        with_project_data = merged_df['has_project_data'].sum()
        coverage = (with_project_data / len(merged_df)) * 100
        
        print(f"  📊 Merge results:")
        print(f"    Records with project data: {with_project_data:,} ({coverage:.1f}%)")
        print(f"    Records without project data: {len(merged_df) - with_project_data:,}")
        
        # Show work type distribution
        print(f"  📈 Work type distribution:")
        work_type_counts = merged_df['work_type'].value_counts()
        for work_type, count in work_type_counts.head(8).items():
            print(f"    {work_type}: {count:,}")
        
        print(f"  ✅ Enhanced dataset size: {len(merged_df):,}")
        return merged_df
    
    def load_llm_extracted_features(self) -> pd.DataFrame:
        """
        Load LLM-extracted features from the ml_ready_716_features_clean.csv file
        
        Returns:
            DataFrame with LLM-extracted features for 716 items
        """
        print(f"\n=== LOADING LLM-EXTRACTED 716 FEATURES ===")
        
        llm_features_file = 'llm_output/ml_ready_716_features_clean.csv'
        
        try:
            llm_df = pd.read_csv(llm_features_file)
            print(f"  📁 Loaded LLM features: {len(llm_df)} unique 716 items")
            print(f"  📊 LLM feature columns: {len([col for col in llm_df.columns if col.startswith('ml_')])}")
            
            # Show available LLM features
            ml_columns = [col for col in llm_df.columns if col.startswith('ml_')]
            print(f"  🧠 LLM-extracted features:")
            for col in ml_columns:
                unique_vals = llm_df[col].nunique()
                print(f"    • {col}: {unique_vals} unique values")
            
            return llm_df
            
        except FileNotFoundError:
            print(f"❌ LLM features file not found: {llm_features_file}")
            print("   Continuing without LLM features...")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error loading LLM features: {e}")
            return pd.DataFrame()
    
    def merge_bidding_with_llm_features(self, bidding_df: pd.DataFrame, llm_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge bidding data with LLM-extracted features
        
        Args:
            bidding_df: DataFrame with bidding data containing 'Item No.' column
            llm_features_df: DataFrame with LLM-extracted features
            
        Returns:
            Enhanced DataFrame with LLM features added
        """
        if len(llm_features_df) == 0:
            print("  ⚠️  No LLM features available - skipping merge")
            return bidding_df
            
        print(f"\n=== MERGING BIDDING DATA WITH LLM FEATURES ===")
        print(f"Bidding data records: {len(bidding_df):,}")
        print(f"Available LLM features: {len(llm_features_df):,}")
        
        # Merge on Item No.
        original_size = len(bidding_df)
        
        # Get LLM feature columns (exclude metadata columns)
        llm_feature_cols = ['item_no'] + [col for col in llm_features_df.columns if col.startswith('ml_')]
        llm_subset = llm_features_df[llm_feature_cols].copy()
        
        # Merge (left join to keep all bidding records)
        merged_df = bidding_df.merge(llm_subset, 
                                   left_on='Item No.', 
                                   right_on='item_no', 
                                   how='left')
        
        # Drop the redundant item_no column
        merged_df = merged_df.drop('item_no', axis=1, errors='ignore')
        
        # Fill missing LLM features with defaults
        ml_columns = [col for col in merged_df.columns if col.startswith('ml_')]
        
        for col in ml_columns:
            if merged_df[col].dtype == 'object':  # Categorical features
                merged_df[col] = merged_df[col].fillna('unknown')
            else:  # Numerical features (flags)
                merged_df[col] = merged_df[col].fillna(0)
        
        # Add flag indicating if LLM features were available
        merged_df['has_llm_features'] = (~merged_df[ml_columns].isnull().all(axis=1)).astype(int)
        
        # Show merge results
        with_llm_features = merged_df['has_llm_features'].sum()
        coverage = (with_llm_features / len(merged_df)) * 100
        
        print(f"  📊 LLM Features merge results:")
        print(f"    Records with LLM features: {with_llm_features:,} ({coverage:.1f}%)")
        print(f"    Records without LLM features: {len(merged_df) - with_llm_features:,}")
        print(f"    LLM features added: {len([col for col in ml_columns])}")
        
        print(f"  ✅ Enhanced dataset with LLM features: {len(merged_df):,}")
        return merged_df
    
    def analyze_construction_categories(self, df: pd.DataFrame) -> pd.Series:
        """Analyze all construction categories to determine expansion targets"""
        print("\n=== CONSTRUCTION CATEGORY ANALYSIS ===")
        
        # Extract category prefixes from item numbers
        df['category_prefix'] = df['Item No.'].str.extract(r'^(\d{3})')
        category_counts = df['category_prefix'].value_counts()
        
        print(f"Top construction categories by record count:")
        for category, count in category_counts.head(15).items():
            sample_items = df[df['category_prefix'] == category]['Item Description'].unique()[:3]
            print(f"  {category}-XX.XX: {count:>6,} records - {', '.join(sample_items[:2])}, ...")
        
        return category_counts
    
    def generate_input_statistics(self, construction_data: pd.DataFrame) -> Dict:
        """Generate comprehensive statistics on input data with visualizations"""
        print(f"\n" + "="*80)
        print("GENERATING COMPREHENSIVE INPUT DATA STATISTICS AND PLOTS")
        print("="*80)
        
        # Set up matplotlib for better plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create output directory for statistics plots
        stats_plots_dir = os.path.join(self.output_dir, 'input_statistics_plots')
        os.makedirs(stats_plots_dir, exist_ok=True)
        
        total_records = len(construction_data)
        print(f"Total records for analysis: {total_records:,}")
        print("📊 Generating comprehensive input data plots...")
        
        # 1. ITEM CLASSIFICATION ANALYSIS
        print("📋 Creating item classification plots...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Item class distribution
        item_class_counts = construction_data['item_class'].value_counts()
        ax1.pie(item_class_counts.values, labels=item_class_counts.index, autopct='%1.1f%%')
        ax1.set_title('Distribution by Item Class')
        
        # Item type distribution (top 10)
        item_type_counts = construction_data['item_type'].value_counts().head(10)
        ax2.barh(item_type_counts.index, item_type_counts.values)
        ax2.set_title('Top 10 Item Types')
        ax2.set_xlabel('Count')
        
        # Category prefix distribution
        if 'category_prefix' in construction_data.columns:
            category_counts = construction_data['category_prefix'].value_counts()
            ax3.bar(category_counts.index, category_counts.values)
            ax3.set_title('Records by Category Prefix')
            ax3.set_xlabel('Category')
            ax3.set_ylabel('Count')
            ax3.tick_params(axis='x', rotation=45)
        
        # Classification coverage
        classification_coverage = construction_data['has_classification'].value_counts()
        coverage_labels = []
        coverage_values = []
        
        if True in classification_coverage.index:
            coverage_labels.append('With Classification')
            coverage_values.append(classification_coverage[True])
        if False in classification_coverage.index:
            coverage_labels.append('Without Classification')
            coverage_values.append(classification_coverage[False])
            
        if len(coverage_values) > 0:
            ax4.pie(coverage_values, labels=coverage_labels, autopct='%1.1f%%')
            ax4.set_title('Classification Coverage')
        else:
            ax4.text(0.5, 0.5, 'No Classification Data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Classification Coverage')
        
        plt.tight_layout()
        plt.savefig(os.path.join(stats_plots_dir, '1_item_classification_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. PROJECT CONTEXT ANALYSIS
        if 'work_type' in construction_data.columns:
            print("🏗️ Creating project context plots...")
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Work type distribution
            work_type_counts = construction_data['work_type'].value_counts().head(10)
            ax1.barh(work_type_counts.index, work_type_counts.values)
            ax1.set_title('Top 10 Work Types')
            ax1.set_xlabel('Count')
            
            # Emergency vs regular projects
            if 'is_emergency' in construction_data.columns:
                emergency_counts = construction_data['is_emergency'].value_counts()
                emergency_labels = ['Regular', 'Emergency']
                emergency_values = [
                    emergency_counts.get(0, 0) + emergency_counts.get(False, 0),
                    emergency_counts.get(1, 0) + emergency_counts.get(True, 0)
                ]
                # Only plot if we have both categories
                if emergency_values[1] > 0:
                    ax2.pie(emergency_values, labels=emergency_labels, autopct='%1.1f%%')
                    ax2.set_title('Emergency vs Regular Projects')
                else:
                    ax2.pie([emergency_values[0]], labels=['Regular Projects'], autopct='%1.1f%%')
                    ax2.set_title('Project Types (All Regular)')
            
            # On-call projects
            if 'is_on_call' in construction_data.columns:
                on_call_counts = construction_data['is_on_call'].value_counts()
                on_call_labels = ['Regular', 'On-Call']
                on_call_values = [
                    on_call_counts.get(0, 0) + on_call_counts.get(False, 0),
                    on_call_counts.get(1, 0) + on_call_counts.get(True, 0)
                ]
                # Only plot if we have both categories
                if on_call_values[1] > 0:
                    ax3.pie(on_call_values, labels=on_call_labels, autopct='%1.1f%%')
                    ax3.set_title('On-Call vs Regular Projects')
                else:
                    ax3.pie([on_call_values[0]], labels=['Regular Projects'], autopct='%1.1f%%')
                    ax3.set_title('Project Types (All Regular)')
            
            # Project length distribution
            if 'project_length_miles' in construction_data.columns:
                ax4.hist(construction_data['project_length_miles'].dropna(), bins=50, alpha=0.7)
                ax4.set_title('Project Length Distribution')
                ax4.set_xlabel('Length (miles)')
                ax4.set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig(os.path.join(stats_plots_dir, '2_project_context_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. PRICE AND QUANTITY ANALYSIS
        print("💰 Creating price and quantity analysis plots...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Price distribution (log scale)
        prices = construction_data[' Bid Unit Price '].dropna()
        ax1.hist(np.log1p(prices), bins=100, alpha=0.7)
        ax1.set_title('Unit Price Distribution (Log Scale)')
        ax1.set_xlabel('Log(Price + 1)')
        ax1.set_ylabel('Frequency')
        
        # Quantity distribution (log scale)
        quantities = construction_data['Project Qty'].dropna()
        ax2.hist(np.log1p(quantities), bins=100, alpha=0.7)
        ax2.set_title('Project Quantity Distribution (Log Scale)')
        ax2.set_xlabel('Log(Quantity + 1)')
        ax2.set_ylabel('Frequency')
        
        # Price vs Quantity scatter (sample)
        sample_data = construction_data.sample(min(5000, len(construction_data)))
        ax3.scatter(np.log1p(sample_data['Project Qty']), np.log1p(sample_data[' Bid Unit Price ']), alpha=0.5)
        ax3.set_title('Price vs Quantity Relationship (Sample)')
        ax3.set_xlabel('Log(Quantity + 1)')
        ax3.set_ylabel('Log(Price + 1)')
        
        # Price by item class boxplot
        item_classes = construction_data['item_class'].unique()
        price_by_class = [np.log1p(construction_data[construction_data['item_class'] == ic][' Bid Unit Price '].dropna()) 
                         for ic in item_classes]
        ax4.boxplot(price_by_class, labels=item_classes)
        ax4.set_title('Price Distribution by Item Class (Log Scale)')
        ax4.set_ylabel('Log(Price + 1)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(stats_plots_dir, '3_price_quantity_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. TEMPORAL ANALYSIS
        print("📅 Creating temporal analysis plots...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Records by year
        if 'year' in construction_data.columns:
            year_counts = construction_data['year'].value_counts().sort_index()
            ax1.plot(year_counts.index, year_counts.values, marker='o')
            ax1.set_title('Records by Year')
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
        
        # Records by month
        if 'month' in construction_data.columns:
            month_counts = construction_data['month'].value_counts().sort_index()
            ax2.bar(month_counts.index, month_counts.values)
            ax2.set_title('Records by Month')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Count')
        
        # Average price by year
        if 'year' in construction_data.columns:
            avg_price_by_year = construction_data.groupby('year')[' Bid Unit Price '].mean()
            ax3.plot(avg_price_by_year.index, avg_price_by_year.values, marker='s')
            ax3.set_title('Average Unit Price by Year')
            ax3.set_xlabel('Year')
            ax3.set_ylabel('Average Price ($)')
            ax3.tick_params(axis='x', rotation=45)
        
        # Records by quarter
        if 'quarter' in construction_data.columns:
            quarter_counts = construction_data['quarter'].value_counts().sort_index()
            ax4.bar(['Q1', 'Q2', 'Q3', 'Q4'], quarter_counts.reindex([1,2,3,4], fill_value=0))
            ax4.set_title('Records by Quarter')
            ax4.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(stats_plots_dir, '4_temporal_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. GEOGRAPHIC AND CONTRACT ANALYSIS
        print("🗺️ Creating geographic and contract analysis plots...")
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Unique contracts over time
        if 'Contract #' in construction_data.columns and 'Letting Date' in construction_data.columns:
            construction_data['Letting Date'] = pd.to_datetime(construction_data['Letting Date'], format='%Y%m%d', errors='coerce')
            monthly_contracts = construction_data.groupby(construction_data['Letting Date'].dt.to_period('M'))['Contract #'].nunique()
            ax1.plot(monthly_contracts.index.to_timestamp(), monthly_contracts.values)
            ax1.set_title('Unique Contracts by Month')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Number of Contracts')
            ax1.tick_params(axis='x', rotation=45)
        
        # Records per contract distribution
        if 'Contract #' in construction_data.columns:
            records_per_contract = construction_data['Contract #'].value_counts()
            ax2.hist(records_per_contract.values, bins=50, alpha=0.7)
            ax2.set_title('Distribution of Records per Contract')
            ax2.set_xlabel('Records per Contract')
            ax2.set_ylabel('Frequency')
        
        # Top contracts by record count
        if 'Contract #' in construction_data.columns:
            top_contracts = construction_data['Contract #'].value_counts().head(10)
            ax3.barh(range(len(top_contracts)), top_contracts.values)
            ax3.set_yticks(range(len(top_contracts)))
            ax3.set_yticklabels([f"Contract {i+1}" for i in range(len(top_contracts))])
            ax3.set_title('Top 10 Contracts by Record Count')
            ax3.set_xlabel('Number of Records')
        
        # Data quality: Missing values heatmap
        missing_data = construction_data.isnull().sum().sort_values(ascending=False).head(10)
        if len(missing_data) > 0:
            ax4.barh(missing_data.index, missing_data.values)
            ax4.set_title('Top 10 Columns with Missing Values')
            ax4.set_xlabel('Missing Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(stats_plots_dir, '5_geographic_contract_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated 5 comprehensive analysis plots in: {stats_plots_dir}")
        
        # Return summary statistics
        stats_summary = {
            'total_records': total_records,
            'plots_directory': stats_plots_dir,
            'unique_contracts': construction_data['Contract #'].nunique() if 'Contract #' in construction_data.columns else 0,
            'date_range': {
                'start': construction_data['year'].min() if 'year' in construction_data.columns else None,
                'end': construction_data['year'].max() if 'year' in construction_data.columns else None
            },
            'item_classes': construction_data['item_class'].nunique() if 'item_class' in construction_data.columns else 0,
            'item_types': construction_data['item_type'].nunique() if 'item_type' in construction_data.columns else 0,
            'price_stats': {
                'mean': float(construction_data[' Bid Unit Price '].mean()),
                'median': float(construction_data[' Bid Unit Price '].median()),
                'std': float(construction_data[' Bid Unit Price '].std())
            } if ' Bid Unit Price ' in construction_data.columns else None
        }
        
        return stats_summary
    

    
    def extract_mpnet_embeddings(self, descriptions: pd.Series, 
                                 model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                                 column_prefix: str = 'mpnet_dim_',
                                 reduce_dimensions: int = None) -> pd.DataFrame:
        """Extract MPNet embeddings for construction descriptions"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("❌ sentence_transformers not installed. Installing now...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
            from sentence_transformers import SentenceTransformer
        
        description_type = "project" if "project" in column_prefix else "item"
        print(f"🤖 Extracting MPNet embeddings for {description_type} descriptions using {model_name}...")
        print(f"  Loading pre-trained model...")
        
        # Load pre-trained model
        model = SentenceTransformer(model_name)
        
        # Generate embeddings for all descriptions
        print(f"  Encoding {len(descriptions)} {description_type} descriptions...")
        embeddings = model.encode(descriptions.tolist(), show_progress_bar=True)
        
        # Reduce dimensions if specified (for project embeddings)
        if reduce_dimensions and reduce_dimensions < embeddings.shape[1]:
            print(f"  🔧 Reducing dimensions from {embeddings.shape[1]} to {reduce_dimensions} using PCA...")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=reduce_dimensions, random_state=42)
            embeddings = pca.fit_transform(embeddings)
            print(f"  📊 PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Convert to DataFrame with column names
        embedding_cols = [f'{column_prefix}{i}' for i in range(embeddings.shape[1])]
        embedding_df = pd.DataFrame(embeddings, columns=embedding_cols, index=descriptions.index)
        
        print(f"  ✅ Generated {embeddings.shape[1]} semantic dimensions for {len(descriptions)} {description_type}s")
        return embedding_df
    
    def extract_universal_text_features(self, df: pd.DataFrame, description_col: str) -> pd.DataFrame:
        """Extract text features that work across all construction categories"""
        print("🔍 Extracting universal construction text features...")
        
        descriptions = df[description_col].astype(str)
        features = pd.DataFrame(index=df.index)
        
        # Basic text statistics
        features['desc_length'] = descriptions.str.len()
        features['word_count'] = descriptions.str.split().str.len()
        features['desc_upper_ratio'] = descriptions.str.count(r'[A-Z]') / descriptions.str.len()
        
        # Universal dimension patterns
        features['has_dimensions'] = descriptions.str.contains(r'\b\d+["\']|\b\d+\s*(?:IN|INCH|FT|FOOT|MM|CM)', case=False, na=False)
        
        # Extract primary dimension
        dimension_pattern = r'\b(\d+(?:\.\d+)?)\s*(?:"|\'|IN|INCH|FT|FOOT|MM|CM)\b'
        features['primary_dimension'] = descriptions.str.extract(dimension_pattern, flags=re.IGNORECASE)[0].astype(float)
        features['primary_dimension'] = features['primary_dimension'].fillna(0)
        
        # Universal material detection
        features['material_concrete'] = descriptions.str.contains(r'\bCONCRETE\b', case=False, na=False)
        features['material_steel'] = descriptions.str.contains(r'\bSTEEL\b', case=False, na=False)
        features['material_asphalt'] = descriptions.str.contains(r'\b(?:ASPHALT|BITUM)\b', case=False, na=False)
        features['material_aggregate'] = descriptions.str.contains(r'\b(?:AGGREGATE|STONE|GRAVEL)\b', case=False, na=False)
        features['material_plastic'] = descriptions.str.contains(r'\bPLASTIC\b', case=False, na=False)
        features['material_aluminum'] = descriptions.str.contains(r'\bALUM(?:INUM)?\b', case=False, na=False)
        features['material_thermoplastic'] = descriptions.str.contains(r'\bTHERMO(?:PLASTIC)?\b', case=False, na=False)
        
        # Pavement marking specific (for backward compatibility)
        features['material_painted'] = descriptions.str.contains(r'\bPAINT(?:ED)?\b', case=False, na=False)
        features['is_stop_line'] = descriptions.str.contains(r'\bSTOP\s+LINE\b', case=False, na=False)
        features['is_arrow'] = descriptions.str.contains(r'\bARROW\b', case=False, na=False)
        features['is_striping'] = descriptions.str.contains(r'\bSTRIP(?:E|ING)\b', case=False, na=False)
        
        # Extract line width for pavement markings
        line_width_pattern = r'\b(\d+)\s*(?:"|\'|IN|INCH)\s+LINE\b'
        features['line_width'] = descriptions.str.extract(line_width_pattern, flags=re.IGNORECASE)[0].astype(float)
        features['line_width'] = features['line_width'].fillna(0)
        features['has_line_width'] = features['line_width'] > 0
        
        # Universal work type detection  
        features['is_removal'] = descriptions.str.contains(r'\bREMOV(?:AL|E|ING)\b', case=False, na=False)
        features['is_installation'] = descriptions.str.contains(r'\b(?:INSTALL|PLACE|FURNISH)\b', case=False, na=False)
        features['is_maintenance'] = descriptions.str.contains(r'\b(?:REPAIR|PATCH|SEAL|MAINT)\b', case=False, na=False)
        features['is_enhanced'] = descriptions.str.contains(r'\bENHANCED\b', case=False, na=False)
        
        # Universal grade/class detection
        features['has_grade_class'] = descriptions.str.contains(r'\b(?:CLASS|GRADE|TYPE)\s+[A-Z0-9]\b', case=False, na=False)
        
        # Universal complexity indicators
        features['has_parentheses'] = descriptions.str.contains(r'\([^)]+\)', na=False)
        features['specification_count'] = descriptions.str.upper().str.count(r'\b(?:CLASS|GRADE|TYPE|PSI|MIL|\d+["\']\s*(?:X|\s+X\s+))')
        
        print(f"  Extracted {len(features.columns)} universal text features")
        return features
    
    def expand_and_correct_description(self, desc: str) -> str:
        """Apply description corrections using configuration"""
        corrected = str(desc).strip()
        config = self.corrections_config.get('description_corrections', {})
        processing_order = self.corrections_config.get('processing_order', [])
        
        # Apply corrections in specified order
        for correction_type in processing_order:
            if correction_type == 'regex_patterns':
                # Apply regex patterns
                for pattern_config in config.get('regex_patterns', []):
                    corrected = re.sub(pattern_config['pattern'], pattern_config['replacement'], corrected)
            else:
                # Apply simple string replacements
                replacements = config.get(correction_type, {})
                for old_text, new_text in replacements.items():
                    corrected = re.sub(r'\b' + re.escape(old_text) + r'\b', new_text, corrected)
        
        return corrected.strip()
    
    def apply_data_filtering(self, df: pd.DataFrame, category_key: str) -> pd.DataFrame:
        """Apply filtering rules from configuration to dataframe"""
        filtered_df = df.copy()
        
        rules = self.hidden_items_config.get('filtering_rules', {}).get(category_key, {})
        
        # Apply pattern exclusions
        for pattern_rule in rules.get('exclude_patterns', []):
            pattern = pattern_rule['pattern']
            before_count = len(filtered_df)
            filtered_df = filtered_df[~filtered_df['Item No.'].str.match(pattern, na=False)]
            removed = before_count - len(filtered_df)
            if removed > 0:
                print(f"  Excluded {removed:,} records matching pattern '{pattern}' - {pattern_rule['reason']}")
        
        # Apply specific item exclusions
        for item_rule in rules.get('exclude_specific_items', []):
            items_to_exclude = item_rule['items']
            before_count = len(filtered_df)
            filtered_df = filtered_df[~filtered_df['Item No.'].isin(items_to_exclude)]
            removed = before_count - len(filtered_df)
            if removed > 0:
                print(f"  Excluded {removed:,} records for specific items - {item_rule['reason']}")
        
        return filtered_df
    
    def train_enhanced_model(self, target_categories: List[str] = None) -> Tuple[CatBoostRegressor, float]:
        """
        Train the enhanced model with item classification features
        
        Args:
            target_categories: List of category prefixes to include (only categories with CSV files)
            
        Returns:
            Tuple of (trained_model, r2_score)
        """
        print("\n" + "="*80)
        print("ENHANCED UNIVERSAL TDOT CONSTRUCTION PRICE PREDICTION WITH ITEM CLASSIFICATIONS")
        print("="*80)
        
        if target_categories is None:
            # Focus on 716 pavement markings only
            target_categories = ['716']
        
        # Load configurations
        self.load_configurations()
        
        # Load item classifications
        self.load_item_classifications()
        
        # Load main bidding data
        print(f"\nLoading main TDOT bidding data from {self.data_file}...")
        df = pd.read_csv(self.data_file, encoding='latin-1')
        print(f"Total bidding records: {len(df):,}")
        
        print(f"\nFiltering for target categories: {target_categories}")
        # Filter for target categories FIRST (more efficient)
        target_mask = df['Item No.'].str.match(r'^(' + '|'.join(target_categories) + ')-', na=False)
        construction_data = df[target_mask].copy()
        print(f"Filtered to target category records: {len(construction_data):,}")
        
        # Load project data to get available contract numbers
        project_data = self.load_project_data()
        
        if len(project_data) > 0:
            # Filter to only keep items within projects that exist in bid tabs data
            available_contracts = set(project_data['contract_number'].unique())
            contract_filter_mask = construction_data['Proposal ID'].isin(available_contracts)
            construction_data = construction_data[contract_filter_mask]
            
            print(f"Filtered to records with project data available: {len(construction_data):,}")
            print(f"Available project contracts: {len(available_contracts):,}")
        else:
            print("⚠️  No project data available - keeping all target category records")
        
        # Merge with item classifications (only for filtered data)
        construction_data = self.merge_bidding_with_classifications(construction_data)
        
        # Merge with project data (only for filtered records that have matching contracts)
        construction_data = self.merge_bidding_with_project_data(construction_data, project_data)
        
        # Load and merge LLM-extracted features for 716 items
        llm_features_data = self.load_llm_extracted_features()
        construction_data = self.merge_bidding_with_llm_features(construction_data, llm_features_data)
        
        # Analyze categories (using the pre-filtered data)
        category_counts = self.analyze_construction_categories(construction_data)
        
        print(f"\nTarget categories for enhanced model: {target_categories}")
        print(f"Using Item Class and Item Type from CSV files instead of hardcoded mappings")
        
        # Add category prefix for analysis
        construction_data['category_prefix'] = construction_data['Item No.'].str.extract(r'^(\d{3})')
        
        # Show enhanced item class distribution with classification info
        print(f"\nEnhanced item class distribution:")
        for item_class, count in construction_data['item_class'].value_counts().items():
            classified_count = construction_data[
                (construction_data['item_class'] == item_class) & 
                (construction_data['has_classification'])
            ].shape[0]
            classification_rate = (classified_count / count) * 100
            print(f"  {item_class}: {count:,} records ({classified_count:,} with classifications - {classification_rate:.1f}%)")
        
        # Apply filtering rules per category (using the pre-processed data)
        print(f"\nApplying configured filtering rules by category:")
        filtered_construction_data = pd.DataFrame()
        
        for category in target_categories:
            category_data = construction_data[construction_data['category_prefix'] == category].copy()
            category_key = f"{category}_category"
            
            if len(category_data) > 0:
                # Show item class for this category
                primary_item_class = category_data['item_class'].mode().iloc[0] if len(category_data) > 0 else 'Unknown'
                print(f"\n  Processing Category {category} (Primary Item Class: {primary_item_class}):")
                filtered_category = self.apply_data_filtering(category_data, category_key)
                print(f"    Kept {len(filtered_category):,} of {len(category_data):,} records")
                filtered_construction_data = pd.concat([filtered_construction_data, filtered_category])
        
        construction_data = filtered_construction_data
        print(f"\nAfter category filtering: {len(construction_data):,} records")
        
        # Apply global filtering if configured
        if 'global_filters' in self.hidden_items_config:
            print(f"\nApplying global filtering rules...")
            original_size = len(construction_data)
            
            global_filters = self.hidden_items_config['global_filters']
            
            # Apply pattern-based exclusions
            if 'exclude_patterns' in global_filters:
                for pattern_config in global_filters['exclude_patterns']:
                    pattern = pattern_config['pattern']
                    case_sensitive = pattern_config.get('case_sensitive', True)
                    use_regex = pattern_config.get('regex', False)
                    
                    if use_regex:
                        if case_sensitive:
                            mask = construction_data['Item Description'].str.contains(pattern, na=False, regex=True)
                        else:
                            mask = construction_data['Item Description'].str.contains(pattern, case=False, na=False, regex=True)
                    else:
                        if case_sensitive:
                            mask = construction_data['Item Description'].str.contains(pattern, na=False)
                        else:
                            mask = construction_data['Item Description'].str.contains(pattern, case=False, na=False)
                    
                    excluded_count = mask.sum()
                    construction_data = construction_data[~mask]
                    
                    if excluded_count > 0:
                        print(f"  Excluded {excluded_count:,} records matching '{pattern}' - {pattern_config['reason']}")
            
            # Apply price threshold filtering
            if 'price_thresholds' in global_filters:
                thresholds = global_filters['price_thresholds']
                max_price = thresholds.get('max_unit_price')
                if max_price is not None:
                    construction_data[' Bid Unit Price '] = construction_data[' Bid Unit Price '].str.replace('$', '').str.replace(',', '').str.strip()
                    construction_data[' Bid Unit Price '] = pd.to_numeric(construction_data[' Bid Unit Price '], errors='coerce')
                    
                    high_price_mask = construction_data[' Bid Unit Price '] > max_price
                    excluded_count = high_price_mask.sum()
                    construction_data = construction_data[~high_price_mask]
                    
                    if excluded_count > 0:
                        print(f"  Excluded {excluded_count:,} records with unit price > ${max_price:,} - {thresholds['reason']}")
            
            total_excluded = original_size - len(construction_data)
            print(f"After global filtering: {len(construction_data):,} records ({total_excluded:,} excluded)")
        
        # Final data cleaning
        print(f"\nFinalizing data cleaning...")
        if construction_data[' Bid Unit Price '].dtype == 'object':
            construction_data[' Bid Unit Price '] = construction_data[' Bid Unit Price '].str.replace('$', '').str.replace(',', '').str.strip()
            construction_data[' Bid Unit Price '] = pd.to_numeric(construction_data[' Bid Unit Price '], errors='coerce')
        construction_data['Project Qty'] = pd.to_numeric(construction_data['Project Qty'], errors='coerce')
        
        # Remove invalid entries
        original_size = len(construction_data)
        construction_data = construction_data.dropna(subset=[' Bid Unit Price ', 'Project Qty'])
        construction_data = construction_data[
            (construction_data[' Bid Unit Price '] > 0) & 
            (construction_data['Project Qty'] > 0)
        ]
        removed = original_size - len(construction_data)
        print(f"After removing invalid prices/quantities: {len(construction_data):,} ({removed} removed)")
        
        # Generate comprehensive input statistics with plots (AFTER data cleaning)
        input_stats = self.generate_input_statistics(construction_data)
        
        print(f"\nTarget categories for enhanced model: {target_categories}")
        print(f"Using Item Class and Item Type from CSV files instead of hardcoded mappings")
        
        # Add category prefix for analysis
        construction_data['category_prefix'] = construction_data['Item No.'].str.extract(r'^(\d{3})')
        
        # Show enhanced item class distribution with classification info
        print(f"\nEnhanced item class distribution:")
        for item_class, count in construction_data['item_class'].value_counts().items():
            classified_count = construction_data[
                (construction_data['item_class'] == item_class) & 
                (construction_data['has_classification'])
            ].shape[0]
            classification_rate = (classified_count / count) * 100
            print(f"  {item_class}: {count:,} records ({classified_count:,} with classifications - {classification_rate:.1f}%)")
        
        # Apply filtering rules per category (using the pre-processed data)
        print(f"\nApplying configured filtering rules by category:")
        filtered_construction_data = pd.DataFrame()
        
        for category in target_categories:
            category_data = construction_data[construction_data['category_prefix'] == category].copy()
            category_key = f"{category}_category"
            
            if len(category_data) > 0:
                # Show item class for this category
                primary_item_class = category_data['item_class'].mode().iloc[0] if len(category_data) > 0 else 'Unknown'
                print(f"\n  Processing Category {category} (Primary Item Class: {primary_item_class}):")
                filtered_category = self.apply_data_filtering(category_data, category_key)
                print(f"    Kept {len(filtered_category):,} of {len(category_data):,} records")
                filtered_construction_data = pd.concat([filtered_construction_data, filtered_category])
        
        construction_data = filtered_construction_data
        print(f"\nAfter category filtering: {len(construction_data):,} records")
        
        # Expand descriptions using proven correction system
        print(f"\nExpanding item descriptions...")
        construction_data['item_description_expanded'] = construction_data['Item Description'].apply(
            self.expand_and_correct_description
        )
        
        # Temporal features
        print(f"Processing temporal features...")
        construction_data['Letting Date'] = pd.to_datetime(construction_data['Letting Date'].astype(str), format='%Y%m%d')
        construction_data['year'] = construction_data['Letting Date'].dt.year
        construction_data['month'] = construction_data['Letting Date'].dt.month
        construction_data['day'] = construction_data['Letting Date'].dt.day
        construction_data['quarter'] = construction_data['Letting Date'].dt.quarter
        
        # Final data cleaning for quantity
        print(f"\nFinalizing data cleaning for quantity...")
        construction_data['Project Qty'] = pd.to_numeric(construction_data['Project Qty'], errors='coerce')
        
        # Remove invalid entries
        original_size = len(construction_data)
        construction_data = construction_data.dropna(subset=[' Bid Unit Price ', 'Project Qty'])
        construction_data = construction_data[
            (construction_data[' Bid Unit Price '] > 0) & 
            (construction_data['Project Qty'] > 0)
        ]
        removed = original_size - len(construction_data)
        print(f"After removing invalid prices/quantities: {len(construction_data):,} ({removed} removed)")
        
        # Target and quantity features
        construction_data['log_unit_price'] = np.log1p(construction_data[' Bid Unit Price '])
        construction_data['log_project_qty'] = np.log1p(construction_data['Project Qty'])
        
        # Extract universal text features - DISABLED FOR TESTING
        # universal_text_features = self.extract_universal_text_features(construction_data, 'Item Description')
        # construction_data = pd.concat([construction_data, universal_text_features], axis=1)
        print("⚠️  Universal text features disabled for testing")
        
        # Extract MPNet embeddings for item descriptions
        item_mpnet_embeddings = self.extract_mpnet_embeddings(construction_data['item_description_expanded'])
        construction_data = pd.concat([construction_data, item_mpnet_embeddings], axis=1)
        
        # Extract MPNet embeddings for project descriptions
        if 'project_description' in construction_data.columns:
            print(f"🏗️  Extracting project description embeddings...")
            project_descriptions = construction_data['project_description'].fillna('Unknown Project')
            project_mpnet_embeddings = self.extract_mpnet_embeddings(
                project_descriptions, 
                column_prefix='project_mpnet_dim_',
                reduce_dimensions=192  # Further reduced to quarter of original 768 dimensions
            )
            construction_data = pd.concat([construction_data, project_mpnet_embeddings], axis=1)
        else:
            print(f"⚠️  No project descriptions available for embeddings")
        
        print(f"Final enhanced dataset size: {len(construction_data):,}")
        
        # Show classification enhancement summary
        classified_records = construction_data['has_classification'].sum()
        classification_rate = (classified_records / len(construction_data)) * 100
        print(f"Records with item classifications: {classified_records:,} ({classification_rate:.1f}%)")
        
        # Train/test split
        print(f"\n" + "="*80)
        print("TIME-BASED DATA SPLIT FOR ENHANCED MODEL")
        print("="*80)
        
        train_data = construction_data[construction_data['year'] <= 2024]
        test_data = construction_data[construction_data['year'] > 2024]
        
        print(f"Train set (2014-2024): {len(train_data):,} records")
        print(f"Test set (2025): {len(test_data):,} records")
        
        # Enhanced feature selection including item classifications
        core_features = [
            'Primary County',
            'year', 'month', 'day', 'quarter', 'log_project_qty'
        ]
        
        # Add item classification features
        classification_features = [
            'item_class', 'item_type'
        ]
        
        # Add project features
        project_features = [
            'work_type', 'project_length', 'is_on_call', 'is_emergency'
        ]
        
        # Add universal text features - DISABLED FOR TESTING
        text_feature_cols = []
        # text_feature_cols = [
        #     'primary_dimension', 'line_width', 'material_concrete', 'material_steel', 'material_asphalt',
        #     'material_aggregate', 'material_plastic', 'material_thermoplastic', 'material_painted',
        #     'is_stop_line', 'is_arrow', 'is_striping', 'is_removal', 'is_installation', 
        #     'is_enhanced', 'has_line_width', 'has_grade_class'
        # ]
        
        # Get embedding column names (both item and project embeddings)
        item_embedding_cols = [col for col in construction_data.columns if col.startswith('mpnet_dim_')]
        project_embedding_cols = [col for col in construction_data.columns if col.startswith('project_mpnet_dim_')]
        all_embedding_cols = item_embedding_cols + project_embedding_cols
        
        # Add LLM-extracted features for 716 items
        llm_feature_cols = [col for col in construction_data.columns if col.startswith('ml_')]
        
        # Separate LLM features into categorical and numerical
        llm_categorical_features = []
        llm_numerical_features = []
        
        for col in llm_feature_cols:
            if construction_data[col].dtype == 'object':
                llm_categorical_features.append(col)
            else:
                llm_numerical_features.append(col)
        
        # Combine all features
        all_features = core_features + classification_features + project_features + text_feature_cols + all_embedding_cols + llm_feature_cols
        self.feature_names = all_features
        
        print(f"\nEnhanced feature breakdown:")
        print(f"  Core features: {len(core_features)}")
        print(f"  Classification features: {len(classification_features)}")
        print(f"  Project features: {len(project_features)}")  
        print(f"  Text features: {len(text_feature_cols)}")
        print(f"  Item MPNet embeddings: {len(item_embedding_cols)}")
        print(f"  Project MPNet embeddings: {len(project_embedding_cols)}")
        print(f"  LLM categorical features: {len(llm_categorical_features)} (NEW)")
        print(f"  LLM numerical features: {len(llm_numerical_features)} (NEW)")
        print(f"  Item Context features (Classification + LLM): {len(classification_features) + len(llm_categorical_features) + len(llm_numerical_features)}")
        print(f"  Total features: {len(all_features)}")
        print(f"  🎯 Testing 716-only model with LLM-extracted semantic features")
        
        # Prepare training data
        X_train = train_data[all_features]
        X_test = test_data[all_features]
        y_train = train_data['log_unit_price']
        y_test = test_data['log_unit_price']
        
        # Categorical features (embeddings are numerical, don't include them)
        cat_features = [
            'item_class', 'Primary County', 'item_type', 'work_type',
            'month', 'day', 'quarter', 'is_on_call', 'is_emergency'
        ] + llm_categorical_features
        # cat_features = [
        #     'item_class', 'Primary County', 'item_type', 'work_type',
        #     'month', 'day', 'quarter', 'is_on_call', 'is_emergency',
        #     'material_concrete', 'material_steel', 'material_asphalt', 'material_aggregate', 
        #     'material_plastic', 'material_thermoplastic', 'material_painted',
        #     'is_stop_line', 'is_arrow', 'is_striping', 'is_removal', 'is_installation', 
        #     'is_enhanced', 'has_line_width', 'has_grade_class'
        # ]
        
        print(f"\n" + "="*80)
        print("TRAINING ENHANCED MODEL WITH ITEM CLASSIFICATIONS")
        print("="*80)
        
        print(f"Features: {len(all_features)}")
        print(f"Categorical features: {len(cat_features)}")
        print(f"Numerical features: {len(all_features) - len(cat_features)}")
        print(f"Item Classes included: {list(train_data['item_class'].unique())}")
        
        # Train model
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        test_pool = Pool(X_test, y_test, cat_features=cat_features)
        
        model = CatBoostRegressor(
            iterations=1000, 
            learning_rate=0.1, 
            depth=6,
            random_seed=42, 
            verbose=100,
            early_stopping_rounds=50
        )
        
        model.fit(train_pool, eval_set=test_pool)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        # Calculate R² on log scale
        r2 = r2_score(y_test, y_pred)
        
        # Convert predictions back to original price scale for MAE/RMSE
        y_test_actual = np.expm1(y_test)
        y_pred_actual = np.expm1(y_pred)
        
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        
        print(f"\n" + "="*80)
        print("ENHANCED MODEL WITH ITEM CLASSIFICATIONS PERFORMANCE")
        print("="*80)
        
        print(f"\nTest Set (2025) - Enhanced Construction Model:")
        print(f"  Records: {len(test_data):,}")
        print(f"  Item Classes: {list(test_data['item_class'].unique())}")
        print(f"  With Classifications: {test_data['has_classification'].sum():,}")
        print(f"  MAE:  ${mae:.2f}")
        print(f"  RMSE: ${rmse:,.2f}")
        print(f"  R²:   {r2:.4f}")
        
        # Performance by item class
        print(f"\nPerformance by Item Class:")
        for i, item_class in enumerate(test_data['item_class'].unique()):
            class_mask = test_data['item_class'] == item_class
            if class_mask.sum() > 10:
                class_test_indices = test_data.index[class_mask]
                class_test_positions = [test_data.index.get_loc(idx) for idx in class_test_indices]
                
                class_y_test_log = y_test.iloc[class_test_positions] if hasattr(y_test, 'iloc') else y_test[class_test_positions]
                class_y_pred_log = y_pred[class_test_positions]
                class_y_test_actual = y_test_actual.iloc[class_test_positions] if hasattr(y_test_actual, 'iloc') else y_test_actual[class_test_positions]
                class_y_pred_actual = y_pred_actual[class_test_positions]
                
                if len(class_y_test_log) > 0:
                    class_r2 = r2_score(class_y_test_log, class_y_pred_log)
                    class_mae = mean_absolute_error(class_y_test_actual, class_y_pred_actual)
                    class_classified = test_data[class_mask]['has_classification'].sum()
                    print(f"  {item_class}: R²={class_r2:.3f}, MAE=${class_mae:.2f}, n={class_mask.sum()}, classified={class_classified}")
        
        # Feature importance analysis
        importance = model.get_feature_importance()
        importance_df = pd.DataFrame({
            'feature': all_features,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\n" + "="*80)
        print("ENHANCED FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        # Show top features with type indicators
        for _, row in importance_df.head(30).iterrows():
            if row['feature'] in classification_features:
                marker = "🏷️"  # Item classification features
            elif row['feature'] in project_features:
                marker = "🏗️"  # Project features
            elif row['feature'] in text_feature_cols:
                marker = "📝"  # Text features
            elif row['feature'].startswith('project_mpnet_dim_'):
                marker = "🏢"  # Project MPNet embeddings
            elif row['feature'].startswith('mpnet_dim_'):
                marker = "🤖"  # Item MPNet embeddings
            else:
                marker = "📊"  # Core features
            print(f"{marker} {row['feature']:<35} {row['importance']:>8.6f}")
        
        # Summary of feature group importance
        classification_importance = importance_df[importance_df['feature'].isin(classification_features)]['importance'].sum()
        item_embedding_importance = importance_df[importance_df['feature'].str.startswith('mpnet_dim_')]['importance'].sum()
        project_embedding_importance = importance_df[importance_df['feature'].str.startswith('project_mpnet_dim_')]['importance'].sum()
        project_feature_importance = importance_df[importance_df['feature'].isin(project_features)]['importance'].sum()
        total_importance = importance_df['importance'].sum()
        
        classification_pct = (classification_importance / total_importance) * 100
        item_embedding_pct = (item_embedding_importance / total_importance) * 100
        project_embedding_pct = (project_embedding_importance / total_importance) * 100
        project_feature_pct = (project_feature_importance / total_importance) * 100
        print(f"\n🏷️  Item Classification Features Summary:")
        print(f"  Total importance: {classification_importance:.2f} ({classification_pct:.1f}%)")
        top_classification = importance_df[importance_df['feature'].isin(classification_features)]
        if len(top_classification) > 0:
            top_classification_feature = top_classification.iloc[0]
            print(f"  Top classification feature: {top_classification_feature['feature']} (rank #{importance_df[importance_df['feature'] == top_classification_feature['feature']].index[0] + 1})")
        
        print(f"\n🏗️  Project Context Features Summary:")
        print(f"  Total importance: {project_feature_importance:.2f} ({project_feature_pct:.1f}%)")
        
        print(f"\n🤖 Item MPNet Embeddings Summary:")
        print(f"  Total embedding importance: {item_embedding_importance:.2f} ({item_embedding_pct:.1f}%)")
        top_item_embedding = importance_df[importance_df['feature'].str.startswith('mpnet_dim_')]
        if len(top_item_embedding) > 0:
            top_item_emb_feature = top_item_embedding.iloc[0]
            print(f"  Top item embedding feature: {top_item_emb_feature['feature']} (rank #{importance_df[importance_df['feature'] == top_item_emb_feature['feature']].index[0] + 1})")
        
        print(f"\n🏢 Project Description Embeddings Summary:")
        print(f"  Total embedding importance: {project_embedding_importance:.2f} ({project_embedding_pct:.1f}%)")
        top_project_embedding = importance_df[importance_df['feature'].str.startswith('project_mpnet_dim_')]
        if len(top_project_embedding) > 0:
            top_proj_emb_feature = top_project_embedding.iloc[0]
            print(f"  Top project embedding feature: {top_proj_emb_feature['feature']} (rank #{importance_df[importance_df['feature'] == top_proj_emb_feature['feature']].index[0] + 1})")
        
        # Save enhanced model
        try:
            model_path = os.path.join(self.output_dir, 'enhanced_universal_construction_model.cbm')
            model.save_model(model_path)
            print(f"\n✅ Enhanced model saved to {model_path}")
            
            # Save feature importance
            importance_path = os.path.join(self.output_dir, 'enhanced_model_feature_importance.csv')
            importance_df.to_csv(importance_path, index=False)
            print(f"✅ Feature importance saved to {importance_path}")
            
        except Exception as e:
            print(f"\n⚠️  Could not save model: {e}")
        
        # Generate comprehensive analysis
        try:
            self.generate_comprehensive_analysis(model, train_data, test_data, all_features)
        except Exception as e:
            print(f"\n⚠️  Could not generate comprehensive analysis: {e}")
        
        self.model = model
        return model, r2
    
    def generate_comprehensive_analysis(self, model, train_data, test_data, all_features, output_suffix=""):
        """Generate comprehensive analysis including plots and grouped feature importance"""
        print(f"\n" + "="*80)
        print("GENERATING COMPREHENSIVE ANALYSIS")
        print("="*80)
        
        # Set up matplotlib
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create output directory for plots
        plots_dir = os.path.join(self.output_dir, 'analysis_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate predictions for both train and test
        X_train = train_data[all_features]
        X_test = test_data[all_features]
        y_train = train_data['log_unit_price']
        y_test = test_data['log_unit_price']
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Convert back to original price scale
        y_train_orig = np.expm1(y_train)
        y_test_orig = np.expm1(y_test)
        train_pred_orig = np.expm1(train_pred)
        test_pred_orig = np.expm1(test_pred)
        
        # 1. Real vs Predicted scatter plots
        self._plot_real_vs_predicted(y_train_orig, train_pred_orig, "Training Set", plots_dir, output_suffix)
        self._plot_real_vs_predicted(y_test_orig, test_pred_orig, "Test Set", plots_dir, output_suffix)
        
        # 2. Residual plots
        self._plot_residuals(y_train_orig, train_pred_orig, "Training Set", plots_dir, output_suffix)
        self._plot_residuals(y_test_orig, test_pred_orig, "Test Set", plots_dir, output_suffix)
        
        # 3. Error distribution plots
        self._plot_error_distribution(y_train_orig, train_pred_orig, y_test_orig, test_pred_orig, plots_dir, output_suffix)
        
        # 4. Performance by item class
        self._plot_performance_by_class(train_data, test_data, y_train_orig, train_pred_orig, 
                                      y_test_orig, test_pred_orig, plots_dir, output_suffix)
        
        # 5. Grouped feature importance
        grouped_importance = self._analyze_grouped_feature_importance(model, all_features)
        self._plot_grouped_feature_importance(grouped_importance, plots_dir, output_suffix)
        
        # 6. Core features detailed breakdown
        self._plot_core_features_breakdown(model, all_features, plots_dir, output_suffix)
        
        # 7. Price distribution analysis
        self._plot_price_distributions(y_train_orig, y_test_orig, plots_dir, output_suffix)
        
        # 8. Export comprehensive results
        results_summary = self._generate_results_summary(
            train_data, test_data, y_train_orig, train_pred_orig, y_test_orig, test_pred_orig, 
            grouped_importance
        )
        
        # Save results summary
        results_path = os.path.join(self.output_dir, f'comprehensive_analysis{output_suffix}.json')
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print(f"\n✅ Comprehensive analysis completed!")
        print(f"📊 Plots saved to: {plots_dir}")
        print(f"📋 Results summary saved to: {results_path}")
        
        return results_summary
    
    def _plot_real_vs_predicted(self, y_real, y_pred, dataset_name, plots_dir, suffix=""):
        """Create real vs predicted scatter plot"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot with transparency
        ax.scatter(y_real, y_pred, alpha=0.5, s=20, c='blue')
        
        # Perfect prediction line
        min_val = min(min(y_real), min(y_pred))
        max_val = max(max(y_real), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Calculate R²
        r2 = r2_score(y_real, y_pred)
        mae = mean_absolute_error(y_real, y_pred)
        rmse = np.sqrt(mean_squared_error(y_real, y_pred))
        
        ax.set_xlabel('Actual Unit Price ($)', fontsize=12)
        ax.set_ylabel('Predicted Unit Price ($)', fontsize=12)
        ax.set_title(f'Real vs Predicted - {dataset_name}\nR² = {r2:.4f}, MAE = ${mae:.2f}, RMSE = ${rmse:.2f}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Log scale for better visualization
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        plt.tight_layout()
        filename = f'real_vs_predicted_{dataset_name.lower().replace(" ", "_")}{suffix}.png'
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_residuals(self, y_real, y_pred, dataset_name, plots_dir, suffix=""):
        """Create residual plots"""
        relative_error = (y_pred - y_real) / y_real * 100
        
        # Filter to ±100% range for better visualization
        mask = (relative_error >= -100) & (relative_error <= 100)
        filtered_pred = y_pred[mask]
        filtered_error = relative_error[mask]
        
        total_points = len(relative_error)
        filtered_points = len(filtered_error)
        excluded_points = total_points - filtered_points
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Percentage residuals vs predicted (filtered to ±100%)
        ax1.scatter(filtered_pred, filtered_error, alpha=0.5, s=20)
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel('Predicted Unit Price ($)')
        ax1.set_ylabel('Residuals (%)')
        ax1.set_title(f'Percentage Residuals vs Predicted - {dataset_name}\n({filtered_points:,} points, {excluded_points} excluded)')
        ax1.set_xscale('log')
        ax1.set_ylim(-100, 100)
        ax1.grid(True, alpha=0.3)
        
        # Relative error histogram (filtered to ±100%)
        ax2.hist(filtered_error, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.set_xlabel('Relative Error (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Relative Error Distribution - {dataset_name}\n(Range: ±100%)')
        ax2.set_xlim(-100, 100)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'residuals_{dataset_name.lower().replace(" ", "_")}{suffix}.png'
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_distribution(self, y_train, train_pred, y_test, test_pred, plots_dir, suffix=""):
        """Plot error distribution comparison"""
        train_error = np.abs(train_pred - y_train) / y_train * 100
        test_error = np.abs(test_pred - y_test) / y_test * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(train_error, bins=50, alpha=0.7, label=f'Training (Median: {np.median(train_error):.1f}%)', density=True)
        ax.hist(test_error, bins=50, alpha=0.7, label=f'Test (Median: {np.median(test_error):.1f}%)', density=True)
        
        ax.set_xlabel('Absolute Relative Error (%)')
        ax.set_ylabel('Density')
        ax.set_title('Error Distribution: Training vs Test Set')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)  # Focus on 0-100% error range
        
        plt.tight_layout()
        filename = f'error_distribution_comparison{suffix}.png'
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_by_class(self, train_data, test_data, y_train, train_pred, y_test, test_pred, plots_dir, suffix=""):
        """Plot performance metrics by item class"""
        
        # Calculate metrics by item class for both sets
        results = []
        
        for dataset_name, data, y_real, y_pred in [
            ("Training", train_data, y_train, train_pred),
            ("Test", test_data, y_test, test_pred)
        ]:
            for item_class in data['item_class'].unique():
                mask = data['item_class'] == item_class
                if mask.sum() > 10:  # Only classes with sufficient data
                    class_y_real = y_real[mask]
                    class_y_pred = y_pred[mask]
                    
                    r2 = r2_score(class_y_real, class_y_pred)
                    mae = mean_absolute_error(class_y_real, class_y_pred)
                    count = len(class_y_real)
                    
                    results.append({
                        'Dataset': dataset_name,
                        'Item_Class': item_class,
                        'R2': r2,
                        'MAE': mae,
                        'Count': count
                    })
        
        results_df = pd.DataFrame(results)
        
        # Create subplot for R² and MAE by class
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² by class
        pivot_r2 = results_df.pivot(index='Item_Class', columns='Dataset', values='R2')
        pivot_r2.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('R² Score by Item Class')
        ax1.set_ylabel('R² Score')
        ax1.set_xlabel('Item Class')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # MAE by class
        pivot_mae = results_df.pivot(index='Item_Class', columns='Dataset', values='MAE')
        pivot_mae.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Mean Absolute Error by Item Class')
        ax2.set_ylabel('MAE ($)')
        ax2.set_xlabel('Item Class')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        filename = f'performance_by_class{suffix}.png'
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        return results_df
    
    def _analyze_grouped_feature_importance(self, model, all_features):
        """Analyze feature importance with logical grouping"""
        feature_importance = model.get_feature_importance()
        
        # Group features
        grouped_importance = {
            'Item MPNet Embeddings': 0,
            'Project MPNet Embeddings': 0,
            'Project Context': 0,
            'Item Context': 0,
            'Core Features': 0
        }
        
        detailed_groups = {
            'Item MPNet Embeddings': [],
            'Project MPNet Embeddings': [],
            'Project Context': [],
            'Item Context': [],
            'Core Features': []
        }
        
        core_features = ['year', 'month', 'day', 'quarter', 'Primary County', 'log_project_qty']
        classification_features = ['item_class', 'item_type']
        project_features = ['work_type', 'project_length', 'is_on_call', 'is_emergency']
        
        for i, feature in enumerate(all_features):
            importance = feature_importance[i]
            
            if feature.startswith('mpnet_dim_'):
                grouped_importance['Item MPNet Embeddings'] += importance
                detailed_groups['Item MPNet Embeddings'].append((feature, importance))
            elif feature.startswith('project_mpnet_dim_'):
                grouped_importance['Project MPNet Embeddings'] += importance
                detailed_groups['Project MPNet Embeddings'].append((feature, importance))
            elif feature.startswith('ml_') or feature in classification_features:
                grouped_importance['Item Context'] += importance
                detailed_groups['Item Context'].append((feature, importance))
            elif feature in core_features:
                grouped_importance['Core Features'] += importance
                detailed_groups['Core Features'].append((feature, importance))
            elif feature in project_features:
                grouped_importance['Project Context'] += importance
                detailed_groups['Project Context'].append((feature, importance))
        
        # Sort individual features within each group
        for group in detailed_groups:
            detailed_groups[group].sort(key=lambda x: x[1], reverse=True)
        
        return {
            'grouped_totals': grouped_importance,
            'detailed_features': detailed_groups
        }
    
    def _plot_grouped_feature_importance(self, grouped_importance, plots_dir, suffix=""):
        """Plot grouped feature importance"""
        groups = list(grouped_importance['grouped_totals'].keys())
        importances = list(grouped_importance['grouped_totals'].values())
        total_importance = sum(importances)
        percentages = [imp/total_importance*100 for imp in importances]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        bars = ax1.bar(groups, percentages, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax1.set_title('Grouped Feature Importance', fontsize=14)
        ax1.set_ylabel('Importance (%)', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        ax2.pie(percentages, labels=groups, autopct='%1.1f%%', startangle=90, 
               colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax2.set_title('Feature Importance Distribution', fontsize=14)
        
        plt.tight_layout()
        filename = f'grouped_feature_importance{suffix}.png'
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed importance to CSV
        detailed_df = []
        for group, features in grouped_importance['detailed_features'].items():
            for feature, importance in features[:5]:  # Top 5 per group
                detailed_df.append({
                    'Group': group,
                    'Feature': feature,
                    'Importance': importance,
                    'Percentage': importance/total_importance*100
                })
        
        detailed_df = pd.DataFrame(detailed_df)
        csv_path = os.path.join(plots_dir, f'grouped_feature_importance{suffix}.csv')
        detailed_df.to_csv(csv_path, index=False)
    
    def _plot_core_features_breakdown(self, model, all_features, plots_dir, suffix=""):
        """Plot detailed breakdown of core features"""
        feature_importance = model.get_feature_importance()
        
        # Core features as defined in the model
        core_features = ['year', 'month', 'day', 'quarter', 'Primary County', 'log_project_qty']
        
        # Extract importance for core features only
        core_feature_data = []
        for i, feature in enumerate(all_features):
            if feature in core_features:
                importance = feature_importance[i]
                core_feature_data.append((feature, importance))
        
        # Sort by importance
        core_feature_data.sort(key=lambda x: x[1], reverse=True)
        
        if not core_feature_data:
            print("No core features found for breakdown")
            return
            
        features, importances = zip(*core_feature_data)
        total_core_importance = sum(importances)
        percentages = [imp/total_core_importance*100 for imp in importances]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        bars = ax1.bar(features, percentages, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83', '#0B6E4F'])
        ax1.set_title('Core Features Breakdown', fontsize=14)
        ax1.set_ylabel('Importance (%)', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels on bars
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        ax2.pie(percentages, labels=features, autopct='%1.1f%%', startangle=90,
                colors=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83', '#0B6E4F'])
        ax2.set_title('Core Features Distribution', fontsize=14)
        
        plt.tight_layout()
        filename = f'core_features_breakdown{suffix}.png'
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed breakdown to CSV
        core_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances,
            'Percentage': percentages
        })
        csv_path = os.path.join(plots_dir, f'core_features_breakdown{suffix}.csv')
        core_df.to_csv(csv_path, index=False)
        
        print(f"  📊 Core features breakdown:")
        for feature, imp, pct in zip(features, importances, percentages):
            print(f"    {feature}: {imp:.3f} ({pct:.1f}%)")
    
    def _plot_price_distributions(self, y_train, y_test, plots_dir, suffix=""):
        """Plot price distribution analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Price distributions
        ax1.hist(y_train, bins=50, alpha=0.7, label=f'Training (n={len(y_train):,})', density=True)
        ax1.hist(y_test, bins=50, alpha=0.7, label=f'Test (n={len(y_test):,})', density=True)
        ax1.set_xlabel('Unit Price ($)')
        ax1.set_ylabel('Density')
        ax1.set_title('Price Distribution: Training vs Test')
        ax1.legend()
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Log price distributions
        ax2.hist(np.log10(y_train), bins=50, alpha=0.7, label='Training', density=True)
        ax2.hist(np.log10(y_test), bins=50, alpha=0.7, label='Test', density=True)
        ax2.set_xlabel('Log10(Unit Price)')
        ax2.set_ylabel('Density')
        ax2.set_title('Log Price Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Box plots
        ax3.boxplot([y_train, y_test], labels=['Training', 'Test'])
        ax3.set_ylabel('Unit Price ($)')
        ax3.set_title('Price Distribution Box Plot')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Statistics comparison
        stats_data = {
            'Dataset': ['Training', 'Test'],
            'Mean': [np.mean(y_train), np.mean(y_test)],
            'Median': [np.median(y_train), np.median(y_test)],
            'Std': [np.std(y_train), np.std(y_test)],
            'Min': [np.min(y_train), np.min(y_test)],
            'Max': [np.max(y_train), np.max(y_test)]
        }
        
        ax4.axis('tight')
        ax4.axis('off')
        table_data = []
        for i, dataset in enumerate(['Training', 'Test']):
            table_data.append([
                dataset,
                f'${stats_data["Mean"][i]:.2f}',
                f'${stats_data["Median"][i]:.2f}',
                f'${stats_data["Std"][i]:.2f}',
                f'${stats_data["Min"][i]:.2f}',
                f'${stats_data["Max"][i]:.2f}'
            ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Dataset', 'Mean', 'Median', 'Std', 'Min', 'Max'],
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Price Statistics Comparison')
        
        plt.tight_layout()
        filename = f'price_distributions{suffix}.png'
        plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_results_summary(self, train_data, test_data, y_train, train_pred, y_test, test_pred, grouped_importance):
        """Generate comprehensive results summary"""
        
        # Calculate metrics
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        # Performance by class
        class_performance = {}
        for item_class in test_data['item_class'].unique():
            mask = test_data['item_class'] == item_class
            if mask.sum() > 0:
                class_y_test = y_test[mask]
                class_test_pred = test_pred[mask]
                class_performance[item_class] = {
                    'count': int(mask.sum()),
                    'r2': float(r2_score(class_y_test, class_test_pred)),
                    'mae': float(mean_absolute_error(class_y_test, class_test_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(class_y_test, class_test_pred)))
                }
        
        return {
            'model_performance': {
                'training': {
                    'r2': float(train_r2),
                    'mae': float(train_mae),
                    'rmse': float(train_rmse),
                    'samples': int(len(train_data))
                },
                'test': {
                    'r2': float(test_r2),
                    'mae': float(test_mae),
                    'rmse': float(test_rmse),
                    'samples': int(len(test_data))
                }
            },
            'class_performance': class_performance,
            'feature_importance': {
                'grouped_totals': {k: float(v) for k, v in grouped_importance['grouped_totals'].items()},
                'top_features_per_group': {
                    group: [(feat, float(imp)) for feat, imp in features[:3]]
                    for group, features in grouped_importance['detailed_features'].items()
                }
            },
            'data_summary': {
                'total_records': int(len(train_data) + len(test_data)),
                'training_records': int(len(train_data)),
                'test_records': int(len(test_data)),
                'item_classes': list(test_data['item_class'].unique()),
                'price_range': {
                    'min': float(min(min(y_train), min(y_test))),
                    'max': float(max(max(y_train), max(y_test))),
                    'median': float(np.median(np.concatenate([y_train, y_test])))
                }
            }
        }

def generate_input_statistics_only():
    """Generate input statistics plots without running full model training"""
    print("\n🔍 GENERATING INPUT STATISTICS ONLY (No Model Training)")
    print("="*60)
    
    # Create model instance for data loading methods
    enhanced_model = EnhancedUniversalTDOTModel()
    
    # Define target categories
    target_categories = ['302', '303', '401', '402', '403', '404', '405', '406', '407', '408', '409', '716']
    
    # Load configurations and data
    enhanced_model.load_configurations()
    enhanced_model.load_item_classifications()
    project_data = enhanced_model.load_project_data()
    
    # Load main data
    print(f"\nLoading main TDOT bidding data from {enhanced_model.data_file}...")
    df = pd.read_csv(enhanced_model.data_file, encoding='latin-1')
    print(f"Total bidding records: {len(df):,}")
    
    print(f"\nFiltering for target categories: {target_categories}")
    target_mask = df['Item No.'].str.match(r'^(' + '|'.join(target_categories) + ')-', na=False)
    construction_data = df[target_mask].copy()
    print(f"Filtered to target category records: {len(construction_data):,}")
    
    # Apply basic data processing needed for statistics
    construction_data = enhanced_model.merge_bidding_with_classifications(construction_data)
    construction_data = enhanced_model.merge_bidding_with_project_data(construction_data, project_data)
    
    # Add category prefix for analysis
    construction_data['category_prefix'] = construction_data['Item No.'].str.extract(r'^(\\d{3})')
    
    # Apply temporal features (needed for temporal plots)
    construction_data['Letting Date'] = pd.to_datetime(construction_data['Letting Date'].astype(str), format='%Y%m%d')
    construction_data['year'] = construction_data['Letting Date'].dt.year
    construction_data['month'] = construction_data['Letting Date'].dt.month
    construction_data['quarter'] = construction_data['Letting Date'].dt.quarter
    
    # Basic price cleaning for statistics
    print("Cleaning price and quantity data...")
    construction_data[' Bid Unit Price '] = construction_data[' Bid Unit Price '].astype(str).str.replace('$', '').str.replace(',', '').str.strip()
    construction_data[' Bid Unit Price '] = pd.to_numeric(construction_data[' Bid Unit Price '], errors='coerce')
    construction_data['Project Qty'] = pd.to_numeric(construction_data['Project Qty'], errors='coerce')
    
    # Remove invalid entries for clean statistics
    print("Removing invalid entries...")
    original_size = len(construction_data)
    construction_data = construction_data.dropna(subset=[' Bid Unit Price ', 'Project Qty'])
    construction_data = construction_data[
        (construction_data[' Bid Unit Price '] > 0) & 
        (construction_data['Project Qty'] > 0)
    ]
    removed = original_size - len(construction_data)
    print(f"Removed {removed} invalid entries")
    
    print(f"Final records for statistics: {len(construction_data):,}")
    
    # Generate comprehensive input statistics with plots
    stats_summary = enhanced_model.generate_input_statistics(construction_data)
    
    print(f"\n✅ Input statistics generation completed!")
    print(f"📊 Plots saved to: {stats_summary['plots_directory']}")
    print(f"📋 Total records analyzed: {stats_summary['total_records']:,}")
    
    return stats_summary

def main():
    """Run the enhanced universal TDOT model"""
    # Initialize the enhanced model
    enhanced_model = EnhancedUniversalTDOTModel()
    
    # Train with 716 pavement markings only
    target_categories = ['716']
    model, r2_score = enhanced_model.train_enhanced_model(target_categories)
    
    print(f"\n" + "="*80)
    print("ENHANCED MODEL TRAINING COMPLETED")
    print("="*80)
    print(f"Final R² Score: {r2_score:.4f}")
    print(f"Model saved with item classification enhancements")
    print(f"Categories included: 716 pavement markings only")
    print(f"Using Item Class and Item Type plus LLM-extracted features")

if __name__ == "__main__":
    import sys
    
    # Check for --stats-only command line argument
    if len(sys.argv) > 1 and sys.argv[1] == '--stats-only':
        generate_input_statistics_only()
    else:
        main()