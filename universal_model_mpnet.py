#!/usr/bin/env python3
"""
Universal TDOT Construction Price Prediction Model with MPNet Embeddings
Adds semantic understanding via sentence transformers to the proven methodology
"""

import pandas as pd
import numpy as np
import re
import json
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def load_hidden_items_config(config_file='hidden_items_config.json'):
    """Load hidden items configuration for filtering training data"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {config_file} not found, using default filtering")
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

def apply_data_filtering(df, category_key, config):
    """Apply filtering rules from configuration to dataframe"""
    filtered_df = df.copy()
    
    rules = config.get('filtering_rules', {}).get(category_key, {})
    
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

def analyze_construction_categories(df):
    """Analyze all construction categories to determine expansion targets"""
    print("=== CONSTRUCTION CATEGORY ANALYSIS ===")
    
    # Extract category prefixes from item numbers
    df['category_prefix'] = df['Item No.'].str.extract(r'^(\d{3})')
    category_counts = df['category_prefix'].value_counts()
    
    print(f"Top construction categories by record count:")
    for category, count in category_counts.head(15).items():
        sample_items = df[df['category_prefix'] == category]['Item Description'].unique()[:3]
        print(f"  {category}-XX.XX: {count:>6,} records - {', '.join(sample_items[:2])}, ...")
    
    return category_counts

def get_category_info(category_prefix):
    """Get category name and characteristics"""
    category_map = {
        '303': {'name': 'Aggregates', 'description': 'Mineral aggregates, base materials'},
        '307': {'name': 'Asphalt Concrete', 'description': 'Hot mix asphalt, surface courses'},
        '403': {'name': 'Hot Mix Asphalt', 'description': 'Asphalt concrete mixes, hot mix'},
        '411': {'name': 'Asphalt Materials', 'description': 'Asphalt concrete mixes, binders'},
        '604': {'name': 'Concrete', 'description': 'Portland cement concrete, structural'},
        '705': {'name': 'Guardrails', 'description': 'Guardrail systems, barriers'},
        '712': {'name': 'Traffic Control', 'description': 'Signs, signals, devices'},
        '713': {'name': 'Signage', 'description': 'Permanent signs, posts'},
        '716': {'name': 'Pavement Markings', 'description': 'Line markings, markers'},
        '717': {'name': 'Mobilization', 'description': 'Project mobilization, setup'},
        '801': {'name': 'Seeding/Sodding', 'description': 'Landscaping, erosion control'}
    }
    return category_map.get(category_prefix, {'name': f'Category {category_prefix}', 'description': 'Other construction items'})

def extract_mpnet_embeddings(descriptions, model_name='sentence-transformers/all-mpnet-base-v2'):
    """Extract MPNet embeddings for construction descriptions"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("❌ sentence_transformers not installed. Installing now...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
        from sentence_transformers import SentenceTransformer
    
    print(f"🤖 Extracting MPNet embeddings using {model_name}...")
    print(f"  Loading pre-trained model...")
    
    # Load pre-trained model
    model = SentenceTransformer(model_name)
    
    # Generate embeddings for all descriptions
    print(f"  Encoding {len(descriptions)} construction descriptions...")
    embeddings = model.encode(descriptions.tolist(), show_progress_bar=True)
    
    # Convert to DataFrame with column names
    embedding_cols = [f'mpnet_dim_{i}' for i in range(embeddings.shape[1])]
    embedding_df = pd.DataFrame(embeddings, columns=embedding_cols, index=descriptions.index)
    
    print(f"  ✅ Generated {embeddings.shape[1]} semantic dimensions for {len(descriptions)} items")
    return embedding_df

def extract_universal_text_features(df, description_col):
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
    # Fix the count method - it doesn't support case parameter
    features['specification_count'] = descriptions.str.upper().str.count(r'\b(?:CLASS|GRADE|TYPE|PSI|MIL|\d+["\']\s*(?:X|\s+X\s+))')
    
    print(f"  Extracted {len(features.columns)} universal text features")
    return features

def load_description_corrections(config_file='description_corrections.json'):
    """Load description correction rules from JSON configuration"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {config_file} not found, using basic corrections")
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

def expand_and_correct_description(desc, corrections_config=None):
    """Apply description corrections using configuration"""
    if corrections_config is None:
        corrections_config = load_description_corrections()
    
    corrected = str(desc).strip()
    config = corrections_config.get('description_corrections', {})
    processing_order = corrections_config.get('processing_order', [])
    
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

def universal_construction_model_with_mpnet():
    """
    Universal construction price prediction model with MPNet embeddings
    Combines proven methodology with semantic understanding
    """
    print("=== UNIVERSAL TDOT CONSTRUCTION PRICE PREDICTION WITH MPNET ===")
    print("Adding semantic understanding to proven methodology")
    
    # Load configurations
    corrections_config = load_description_corrections()
    hidden_items_config = load_hidden_items_config()
    print(f"Loaded description corrections and filtering configuration")
    
    # Load data
    df = pd.read_csv('Data/TDOT_data.csv', encoding='latin-1')
    print(f"Total records: {len(df):,}")
    
    # Analyze categories
    category_counts = analyze_construction_categories(df)
    
    # Select categories for modeling (only what user requested)
    target_categories = ['716', '303', '604', '403']  # Pavement Markings + Aggregates + Concrete + 403
    print(f"\nTarget categories for model: {target_categories}")
    print("716 = Pavement Markings (proven success)")  
    print("303 = Aggregates (requested addition)")
    print("604 = Concrete (requested addition)")
    print("403 = Hot Mix Asphalt (new addition)")
    
    # Filter for target categories
    print(f"\nFiltering for target construction categories...")
    target_mask = df['Item No.'].str.match(r'^(' + '|'.join(target_categories) + ')-', na=False)
    construction_data = df[target_mask].copy()
    print(f"Target category records: {len(construction_data):,}")
    
    # Add category information
    construction_data['category_prefix'] = construction_data['Item No.'].str.extract(r'^(\d{3})')
    construction_data['category_name'] = construction_data['category_prefix'].apply(
        lambda x: get_category_info(x)['name']
    )
    
    print("\nCategory distribution:")
    for cat, count in construction_data['category_name'].value_counts().items():
        print(f"  {cat}: {count:,} records")
    
    # Apply filtering rules per category
    print(f"\nApplying configured filtering rules by category:")
    filtered_construction_data = pd.DataFrame()
    
    for category in target_categories:
        category_data = construction_data[construction_data['category_prefix'] == category].copy()
        category_key = f"{category}_{get_category_info(category)['name'].lower().replace(' ', '_')}"
        
        if len(category_data) > 0:
            print(f"\n  Processing {get_category_info(category)['name']} ({category}):")
            filtered_category = apply_data_filtering(category_data, category_key, hidden_items_config)
            print(f"    Kept {len(filtered_category):,} of {len(category_data):,} records")
            filtered_construction_data = pd.concat([filtered_construction_data, filtered_category])
    
    construction_data = filtered_construction_data
    print(f"\nAfter category filtering: {len(construction_data):,} records")
    
    # Apply global filtering rules (design-build, mega-projects, etc.)
    if 'global_filters' in hidden_items_config:
        print(f"\nApplying global filtering rules...")
        original_size = len(construction_data)
        
        global_filters = hidden_items_config['global_filters']
        
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
                # Clean price data first for threshold comparison
                construction_data[' Bid Unit Price '] = construction_data[' Bid Unit Price '].str.replace('$', '').str.replace(',', '').str.strip()
                construction_data[' Bid Unit Price '] = pd.to_numeric(construction_data[' Bid Unit Price '], errors='coerce')
                
                high_price_mask = construction_data[' Bid Unit Price '] > max_price
                excluded_count = high_price_mask.sum()
                construction_data = construction_data[~high_price_mask]
                
                if excluded_count > 0:
                    print(f"  Excluded {excluded_count:,} records with unit price > ${max_price:,} - {thresholds['reason']}")
        
        total_excluded = original_size - len(construction_data)
        print(f"After global filtering: {len(construction_data):,} records ({total_excluded:,} excluded)")
    else:
        print(f"No global filters configured")
    
    # Data cleaning
    print(f"\nFinalizing data cleaning...")
    # Price data may already be cleaned by global filters, ensure it's numeric
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
    
    # Expand descriptions using proven correction system
    print(f"\nExpanding item descriptions...")
    construction_data['item_description_expanded'] = construction_data['Item Description'].apply(
        lambda desc: expand_and_correct_description(desc, corrections_config)
    )
    
    # Temporal features
    print(f"Processing temporal features...")
    construction_data['Letting Date'] = pd.to_datetime(construction_data['Letting Date'].astype(str), format='%Y%m%d')
    construction_data['year'] = construction_data['Letting Date'].dt.year
    construction_data['month'] = construction_data['Letting Date'].dt.month
    construction_data['day'] = construction_data['Letting Date'].dt.day
    construction_data['quarter'] = construction_data['Letting Date'].dt.quarter
    
    # Target and quantity features
    construction_data['log_unit_price'] = np.log1p(construction_data[' Bid Unit Price '])
    construction_data['log_project_qty'] = np.log1p(construction_data['Project Qty'])
    
    # Extract universal text features
    universal_text_features = extract_universal_text_features(construction_data, 'Item Description')
    construction_data = pd.concat([construction_data, universal_text_features], axis=1)
    
    # Extract MPNet embeddings
    mpnet_embeddings = extract_mpnet_embeddings(construction_data['item_description_expanded'])
    construction_data = pd.concat([construction_data, mpnet_embeddings], axis=1)
    
    print(f"Final dataset size: {len(construction_data):,}")
    
    # Train/test split
    print(f"\n" + "="*80)
    print("TIME-BASED DATA SPLIT")
    print("="*80)
    
    train_data = construction_data[construction_data['year'] <= 2024]
    test_data = construction_data[construction_data['year'] > 2024]
    
    print(f"Train set (2014-2024): {len(train_data):,} records")
    print(f"Test set (2025): {len(test_data):,} records")
    
    # Feature selection for universal model (without categorical description since MPNet captures semantics)
    core_features = [
        'category_name', 'Primary County',
        'year', 'month', 'day', 'quarter', 'log_project_qty'
    ]
    
    # Add universal text features
    text_feature_cols = [
        'primary_dimension', 'line_width', 'material_concrete', 'material_steel', 'material_asphalt',
        'material_aggregate', 'material_plastic', 'material_thermoplastic', 'material_painted',
        'is_stop_line', 'is_arrow', 'is_striping', 'is_removal', 'is_installation', 
        'is_enhanced', 'has_line_width', 'has_grade_class'
    ]
    
    # Get embedding column names
    embedding_cols = [col for col in construction_data.columns if col.startswith('mpnet_dim_')]
    
    # Combine all features
    all_features = core_features + text_feature_cols + embedding_cols
    
    print(f"\nFeature breakdown:")
    print(f"  Core features: {len(core_features)} (no categorical description)")
    print(f"  Text features: {len(text_feature_cols)}")  
    print(f"  MPNet embeddings: {len(embedding_cols)}")
    print(f"  Total features: {len(all_features)}")
    print(f"  → Testing if MPNet semantic embeddings can replace categorical item descriptions")
    
    # Prepare training data
    X_train = train_data[all_features]
    X_test = test_data[all_features]
    y_train = train_data['log_unit_price']
    y_test = test_data['log_unit_price']
    
    # Categorical features (embeddings are numerical, don't include them)
    cat_features = [
        'category_name', 'Primary County', 
        'month', 'day', 'quarter',
        'material_concrete', 'material_steel', 'material_asphalt', 'material_aggregate', 
        'material_plastic', 'material_thermoplastic', 'material_painted',
        'is_stop_line', 'is_arrow', 'is_striping', 'is_removal', 'is_installation', 
        'is_enhanced', 'has_line_width', 'has_grade_class'
    ]
    
    print(f"\n" + "="*80)
    print("TRAINING UNIVERSAL CONSTRUCTION MODEL WITH MPNET")
    print("="*80)
    
    print(f"Features: {len(all_features)}")
    print(f"Categorical features: {len(cat_features)}")
    print(f"Numerical features: {len(all_features) - len(cat_features)}")
    print(f"Categories included: {list(train_data['category_name'].unique())}")
    
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
    
    r2 = r2_score(y_test, y_pred)
    
    y_test_actual = np.expm1(y_test)
    y_pred_actual = np.expm1(y_pred)
    
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    
    print(f"\n" + "="*80)
    print("UNIVERSAL MODEL WITH MPNET PERFORMANCE")
    print("="*80)
    
    print(f"\nTest Set (2025) - All Construction Categories:")
    print(f"  Records: {len(test_data):,}")
    print(f"  Categories: {list(test_data['category_name'].unique())}")
    print(f"  MAE:  ${mae:.2f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R²:   {r2:.4f}")
    
    # Performance by category
    print(f"\nPerformance by Category:")
    for i, category in enumerate(test_data['category_name'].unique()):
        cat_mask = test_data['category_name'] == category
        if cat_mask.sum() > 10:
            cat_test_indices = test_data.index[cat_mask]
            cat_test_positions = [test_data.index.get_loc(idx) for idx in cat_test_indices]
            
            cat_y_test = y_test_actual.iloc[cat_test_positions] if hasattr(y_test_actual, 'iloc') else y_test_actual[cat_test_positions]
            cat_y_pred = y_pred_actual[cat_test_positions]
            
            if len(cat_y_test) > 0:
                cat_r2 = r2_score(cat_y_test, cat_y_pred)
                cat_mae = mean_absolute_error(cat_y_test, cat_y_pred)
                print(f"  {category}: R²={cat_r2:.3f}, MAE=${cat_mae:.2f}, n={cat_mask.sum()}")
    
    # Feature importance
    importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'feature': all_features,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(f"\n" + "="*80)
    print("FEATURE IMPORTANCE")
    print("="*80)
    
    # Show top features with type indicators
    for _, row in importance_df.head(25).iterrows():
        if row['feature'] in text_feature_cols:
            marker = "🏗️"
        elif row['feature'].startswith('mpnet_dim_'):
            marker = "🤖"
        else:
            marker = "📊"
        print(f"{marker} {row['feature']:<35} {row['importance']:>8.6f}")
    
    # Summary of embedding importance
    embedding_importance = importance_df[importance_df['feature'].str.startswith('mpnet_dim_')]['importance'].sum()
    total_importance = importance_df['importance'].sum()
    embedding_pct = (embedding_importance / total_importance) * 100
    
    print(f"\n🤖 MPNet Embeddings Summary:")
    print(f"  Total embedding importance: {embedding_importance:.2f} ({embedding_pct:.1f}%)")
    print(f"  Top embedding feature: {importance_df[importance_df['feature'].str.startswith('mpnet_dim_')].iloc[0]['feature']} (rank #{importance_df[importance_df['feature'].str.startswith('mpnet_dim_')].index[0] + 1})")
    
    # Save model
    try:
        import os
        os.makedirs('output', exist_ok=True)
        model.save_model('output/universal_construction_model_mpnet.cbm')
        print(f"\n✅ MPNet-enhanced model saved to output/universal_construction_model_mpnet.cbm")
    except Exception as e:
        print(f"\n⚠️  Could not save model: {e}")
    
    return model, r2

if __name__ == "__main__":
    model, r2_score = universal_construction_model_with_mpnet()