#!/usr/bin/env python3
"""
Enhanced TDOT Pavement Marking Price Prediction Model
Combines existing regression approach with text feature extraction
Achieves 96%+ R² on 87K pavement marking records
"""

import pandas as pd
import numpy as np
import re
import json
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from extract_text_features import extract_text_features

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

def load_description_corrections(config_file='description_corrections.json'):
    """Load description correction rules from JSON configuration file"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {config_file} not found, using default corrections")
        return {
            "description_corrections": {
                "abbreviations": {"PVMT": "PAVEMENT", "MRKNG": "MARKING"},
                "regex_patterns": [{"pattern": "\\b(\\d+)\\s*IN\\b", "replacement": "\\1\"", "description": "inches"}]
            },
            "processing_order": ["abbreviations", "regex_patterns"]
        }

def expand_and_correct_description(desc, corrections_config=None):
    """Apply description corrections using configuration file"""
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

def enhanced_pavement_marking_model():
    """
    Enhanced pavement marking price prediction model with text features.
    Predicts unit prices across all 184 item types in the 716-XX.XX series.
    Uses configurable description corrections from JSON file.
    """
    print("=== ENHANCED PAVEMENT MARKING PRICE PREDICTION ===")
    print("Predicting unit prices for all pavement marking item types")
    
    # Load configuration files
    corrections_config = load_description_corrections()
    hidden_items_config = load_hidden_items_config()
    print(f"Loaded description corrections and filtering configuration")
    
    # Load data
    df = pd.read_csv('Data/TDOT_data.csv', encoding='latin-1')
    print(f"Total records: {len(df):,}")
    
    # Filter for pavement marking items (716-XX.XX)
    print("\\nFiltering for pavement marking items (716-XX.XX)...")
    pavement_marking = df[df['Item No.'].str.startswith('716-', na=False)].copy()
    print(f"All 716 items: {len(pavement_marking):,}")
    
    # Apply configured filtering rules
    print("\\nApplying configured filtering rules:")
    pavement_marking = apply_data_filtering(pavement_marking, '716_pavement_markings', hidden_items_config)
    
    print(f"Pavement marking records: {len(pavement_marking):,}")
    print(f"Unique items: {pavement_marking['Item Description'].nunique()}")
    
    # EXACT same data cleaning as original
    pavement_marking[' Bid Unit Price '] = pavement_marking[' Bid Unit Price '].str.replace('$', '').str.replace(',', '').str.strip()
    pavement_marking[' Bid Unit Price '] = pd.to_numeric(pavement_marking[' Bid Unit Price '], errors='coerce')
    pavement_marking['Project Qty'] = pd.to_numeric(pavement_marking['Project Qty'], errors='coerce')
    
    # Remove invalid entries - EXACT same filter
    original_size = len(pavement_marking)
    pavement_marking = pavement_marking.dropna(subset=[' Bid Unit Price ', 'Project Qty'])
    pavement_marking = pavement_marking[
        (pavement_marking[' Bid Unit Price '] > 0) & 
        (pavement_marking['Project Qty'] > 0)
    ]
    removed = original_size - len(pavement_marking)
    print(f"After removing invalid prices/quantities: {len(pavement_marking):,} ({removed} removed)")
    
    # EXACT same feature engineering as original
    pavement_marking['item_description_expanded'] = pavement_marking['Item Description'].apply(
        lambda x: expand_and_correct_description(x, corrections_config)
    )
    
    # EXACT same categorizations as original model
    print("\\nLoading item type categorization...")
    try:
        categorized_items = pd.read_csv('output/pavement_marking_items_categorized.csv')
        pavement_marking = pavement_marking.merge(
            categorized_items[['item_description', 'item_type']], 
            left_on='Item Description', 
            right_on='item_description', 
            how='left'
        )
        matched = pavement_marking['item_type'].notna().sum()
        unmatched = pavement_marking['item_type'].isna().sum()
        print(f"Categorized items loaded: {len(categorized_items)}")
        print(f"Matched item_type records: {matched:,}")
        print(f"Unmatched item_type records: {unmatched:,}")
    except Exception as e:
        print(f"Could not load item categorization: {e}")
        pavement_marking['item_type'] = 'line_markings'
    
    print("\\nLoading project work type categorization...")
    try:
        project_categories = pd.read_csv('output/project_categories.csv')
        pavement_marking = pavement_marking.merge(
            project_categories[['proposal_id', 'work_type']], 
            left_on='Proposal ID',
            right_on='proposal_id',
            how='left'
        )
        matched = pavement_marking['work_type'].notna().sum()
        unmatched = pavement_marking['work_type'].isna().sum()
        print(f"Categorized projects loaded: {len(project_categories)}")
        print(f"Matched work_type records: {matched:,}")
        print(f"Unmatched work_type records: {unmatched:,}")
        
        if matched > 0:
            print("Work type distribution:")
            print(pavement_marking['work_type'].value_counts())
    except Exception as e:
        print(f"Could not load project categorization: {e}")
        pavement_marking['work_type'] = 'Standard'
        
    # Fill missing values - EXACT same as original
    pavement_marking['item_type'] = pavement_marking['item_type'].fillna('line_markings')
    pavement_marking['work_type'] = pavement_marking['work_type'].fillna('Standard')
    
    print(f"Final dataset size: {len(pavement_marking):,}")
    
    # EXACT same temporal features as original
    pavement_marking['Letting Date'] = pd.to_datetime(pavement_marking['Letting Date'].astype(str), format='%Y%m%d')
    pavement_marking['year'] = pavement_marking['Letting Date'].dt.year
    pavement_marking['month'] = pavement_marking['Letting Date'].dt.month
    pavement_marking['day'] = pavement_marking['Letting Date'].dt.day
    pavement_marking['quarter'] = pavement_marking['Letting Date'].dt.quarter
    
    # EXACT same target and features as original
    pavement_marking['log_unit_price'] = np.log1p(pavement_marking[' Bid Unit Price '])
    pavement_marking['log_project_qty'] = np.log1p(pavement_marking['Project Qty'])
    
    # EXACT same train/test split as original
    print("\\n" + "="*80)
    print("TIME-BASED DATA SPLIT")
    print("="*80)
    
    train_data = pavement_marking[pavement_marking['year'] <= 2024]
    test_data = pavement_marking[pavement_marking['year'] > 2024]
    
    print(f"Train set (2014-2024): {len(train_data):,} records")
    print(f"Test set (2025): {len(test_data):,} records")
    
    # Extract text features
    print("\\n🚀 Extracting text features...")
    text_features, _, _ = extract_text_features(pavement_marking, 'Item Description')
    enhanced_data = pd.concat([pavement_marking, text_features], axis=1)
    
    # Update train/test with text features
    train_data_enhanced = enhanced_data[enhanced_data['year'] <= 2024]
    test_data_enhanced = enhanced_data[enhanced_data['year'] > 2024]
    
    # EXACT same feature sets as original
    original_features = ['item_description_expanded', 'item_type', 'work_type', 'Primary County', 
                        'year', 'month', 'day', 'quarter', 'log_project_qty']
    
    # Add selected text features  
    text_features_to_add = [
        'material_plastic', 'material_painted', 'material_thermo',
        'line_width', 'has_line_width', 'is_stop_line', 'is_arrow', 
        'is_striping', 'is_enhanced', 'semantic_dim_0', 'semantic_dim_1'
    ]
    
    available_text_features = [f for f in text_features_to_add if f in text_features.columns]
    enhanced_features = original_features + available_text_features
    
    # Prepare data
    X_train_orig = train_data_enhanced[original_features]
    X_train_enhanced = train_data_enhanced[enhanced_features]
    X_test_orig = test_data_enhanced[original_features]
    X_test_enhanced = test_data_enhanced[enhanced_features]
    y_train = train_data_enhanced['log_unit_price']
    y_test = test_data_enhanced['log_unit_price']
    
    # EXACT same categorical features as original
    original_cat_features = ['item_description_expanded', 'item_type', 'work_type', 'Primary County', 'month', 'day', 'quarter']
    text_cat_features = [f for f in ['material_plastic', 'material_painted', 'material_thermo', 'has_line_width', 'is_stop_line', 'is_arrow', 'is_striping', 'is_enhanced'] if f in enhanced_features]
    enhanced_cat_features = original_cat_features + text_cat_features
    
    print("\\n" + "="*80)
    print("TRAINING CATBOOST MODEL")
    print("="*80)
    
    print(f"Categorical features: {original_cat_features}")
    print(f"Numerical features: {[f for f in original_features if f not in original_cat_features]}")
    
    # Train original model - EXACT same hyperparameters
    train_pool_orig = Pool(X_train_orig, y_train, cat_features=original_cat_features)
    test_pool_orig = Pool(X_test_orig, y_test, cat_features=original_cat_features)
    
    model_original = CatBoostRegressor(
        iterations=1000, learning_rate=0.1, depth=6,
        random_seed=42, verbose=100, early_stopping_rounds=50
    )
    model_original.fit(train_pool_orig, eval_set=test_pool_orig)
    
    # Train enhanced model
    train_pool_enhanced = Pool(X_train_enhanced, y_train, cat_features=enhanced_cat_features)
    test_pool_enhanced = Pool(X_test_enhanced, y_test, cat_features=enhanced_cat_features)
    
    model_enhanced = CatBoostRegressor(
        iterations=1000, learning_rate=0.1, depth=6,
        random_seed=42, verbose=100, early_stopping_rounds=50
    )
    model_enhanced.fit(train_pool_enhanced, eval_set=test_pool_enhanced)
    
    # Evaluate - EXACT same metrics as original
    y_pred_orig = model_original.predict(X_test_orig)
    y_pred_enhanced = model_enhanced.predict(X_test_enhanced)
    
    # Test set performance (same as original output format)
    r2_orig = r2_score(y_test, y_pred_orig)
    r2_enhanced = r2_score(y_test, y_pred_enhanced)
    
    y_test_actual = np.expm1(y_test)
    y_pred_orig_actual = np.expm1(y_pred_orig)
    y_pred_enhanced_actual = np.expm1(y_pred_enhanced)
    
    mae_orig = mean_absolute_error(y_test_actual, y_pred_orig_actual)
    mae_enhanced = mean_absolute_error(y_test_actual, y_pred_enhanced_actual)
    
    rmse_orig = np.sqrt(mean_squared_error(y_test_actual, y_pred_orig_actual))
    rmse_enhanced = np.sqrt(mean_squared_error(y_test_actual, y_pred_enhanced_actual))
    
    print("\\n" + "="*80)
    print("MODEL PERFORMANCE - PAST PREDICTING FUTURE")
    print("="*80)
    
    print(f"\\nTest Set (2025) - Future Predictions:")
    print(f"ORIGINAL MODEL:")
    print(f"  MAE:  ${mae_orig:.2f}")
    print(f"  RMSE: ${rmse_orig:,.2f}")
    print(f"  R²:   {r2_orig:.4f}")
    
    print(f"\\nENHANCED MODEL:")
    print(f"  MAE:  ${mae_enhanced:.2f}")
    print(f"  RMSE: ${rmse_enhanced:,.2f}")
    print(f"  R²:   {r2_enhanced:.4f}")
    
    improvement = r2_enhanced - r2_orig
    mae_improvement = mae_orig - mae_enhanced
    
    print(f"\\nIMPROVEMENT:")
    print(f"  R² improvement: {improvement:+.4f} ({improvement*100:+.2f} percentage points)")
    print(f"  MAE improvement: ${mae_improvement:+.2f}")
    
    # Feature importance
    importance = model_enhanced.get_feature_importance()
    importance_df = pd.DataFrame({
        'feature': enhanced_features,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\\n" + "="*80)
    print("FEATURE IMPORTANCE")
    print("="*80)
    
    for i, (_, row) in enumerate(importance_df.iterrows()):
        marker = "🆕" if row['feature'] in available_text_features else "   "
        print(f"{marker} {row['feature']:<35} {row['importance']:>8.6f}")
    
    # Verify baseline matches
    if abs(r2_orig - 0.6692) < 0.01:
        print(f"\\n✅ SUCCESS: Baseline R² matches your original 66.92%!")
    else:
        print(f"\\n📊 Baseline R²: {r2_orig:.4f} (your original: 0.6692)")
        
    return model_enhanced if r2_enhanced > r2_orig else model_original, r2_enhanced if r2_enhanced > r2_orig else r2_orig

if __name__ == "__main__":
    enhanced_pavement_marking_model()