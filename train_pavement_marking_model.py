import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings('ignore')

def expand_and_correct_description(desc):
    """Expand abbreviations, correct spelling errors, and normalize descriptions"""
    # First, fix common spelling errors
    corrections = {
        'PAVMENT': 'PAVEMENT',
        'PAVEMEMT': 'PAVEMENT',
        'REFL': 'REFLECTIVE',
        'SYMB': 'SYMBOL',
        'SNWPLWBLE': 'SNOWPLOWABLE',
        'MRKRS': 'MARKERS',
    }
    
    corrected = desc
    for error, correct in corrections.items():
        corrected = re.sub(r'\b' + error + r'\b', correct, corrected)
    
    # Normalize inch measurements - convert both "IN" and "INCH" to inches symbol
    # Match patterns like 4IN, 6IN, 8IN, 12IN, 24IN, etc.
    corrected = re.sub(r'\b(\d+)\s*IN\b', r'\1"', corrected)
    corrected = re.sub(r'\b(\d+)\s*INCH\b', r'\1"', corrected)
    
    # Normalize millimeter measurements
    corrected = re.sub(r'\b(\d+)\s*MM\b', r'\1mm', corrected, flags=re.IGNORECASE)
    
    # Fix parentheses spacing - but first handle ")(" -> ") ("
    corrected = re.sub(r'\)\(', ') (', corrected)
    
    # Normalize spacing around parentheses
    corrected = re.sub(r'\s*\(\s*', ' (', corrected)
    corrected = re.sub(r'\s*\)\s*', ') ', corrected)
    corrected = corrected.strip()
    
    # Remove empty or whitespace-only parentheses
    corrected = re.sub(r'\(\s*\)', '', corrected)
    
    # Then expand abbreviations
    abbreviations = {
        'PVMT': 'PAVEMENT',
        'MRKNG': 'MARKING',
        'THERMO': 'THERMOPLASTIC',
        'PLSTC': 'PLASTIC',
        'PREFRMD': 'PREFORMED',
        'MRKING': 'MARKING',
        'BI-DIR': 'BI-DIRECTIONAL',
        'MONO-DIR': 'MONO-DIRECTIONAL',
        'W/': 'WITH ',
        'TEMP': 'TEMPORARY',
        'CHAN': 'CHANNELIZATION',
        'PROFILD': 'PROFILED',
        'THERMOPLST': 'THERMOPLASTIC',
        'LN': 'LINE',
        'FLTLNE': 'FLATLINE',
        'ENHNCD': 'ENHANCED',
    }
    
    expanded = corrected
    for abbr, full in abbreviations.items():
        expanded = re.sub(r'\b' + abbr + r'\b', full, expanded)
    
    # Final cleanup - remove extra spaces
    expanded = re.sub(r'\s+', ' ', expanded).strip()
    
    return expanded

print("="*80)
print("PAVEMENT MARKING PRICE PREDICTION - CATBOOST MODEL")
print("="*80)

# Load TDOT data
print("\nLoading TDOT_data.csv...")
df = pd.read_csv('Data/TDOT_data.csv', encoding='latin-1')
print(f"Total records: {len(df):,}")

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Filter for pavement marking items (716- series), excluding 716-99 items and removal items
print("\nFiltering for pavement marking items (716-XX.XX)...")
pavement_marking = df[
    (df['item_no.'].str.startswith('716-', na=False)) & 
    (~df['item_no.'].str.startswith('716-99', na=False))
].copy()

# Remove specific item ranges: 716-08.31 to 716-08.34 and 716-08.11 to 716-08.17
# These are removal items (hydroblast removal, word removal)
items_to_remove = [f'716-08.{i:02d}' for i in list(range(31, 35)) + list(range(11, 18))]
for item in items_to_remove:
    pavement_marking = pavement_marking[~pavement_marking['item_no.'].str.startswith(item, na=False)]

# Remove items with "(DESCRIPTION)" placeholder - these are generic/unspecified items
pavement_marking = pavement_marking[~pavement_marking['item_description'].str.contains(r'\(DESCRIPTION\)', na=False, regex=True)]

print(f"Pavement marking records: {len(pavement_marking):,}")
print(f"Unique items: {pavement_marking['item_no.'].nunique()}")
print(f"Contracts: {pavement_marking['proposal_id'].nunique()}")

# Convert numeric columns (remove spaces, dollar signs, and commas, convert to float)
numeric_cols = ['bid_unit_price', 'bid_extended_amount', 'project_qty']
for col in numeric_cols:
    pavement_marking[col] = pd.to_numeric(
        pavement_marking[col].astype(str).str.strip().str.replace('$', '').str.replace(',', ''), 
        errors='coerce'
    )

# Remove records with missing or zero prices
initial_count = len(pavement_marking)
pavement_marking = pavement_marking[
    (pavement_marking['bid_unit_price'].notna()) & 
    (pavement_marking['bid_unit_price'] > 0) &
    (pavement_marking['project_qty'].notna()) &
    (pavement_marking['project_qty'] > 0)
].copy()
print(f"After removing invalid prices/quantities: {len(pavement_marking):,} ({initial_count - len(pavement_marking):,} removed)")

# Add date features
print("\nEngineering features...")
# Parse letting date - format is YYYYMMDD
pavement_marking['letting_date'] = pd.to_datetime(pavement_marking['letting_date'], format='%Y%m%d', errors='coerce')
pavement_marking['year'] = pavement_marking['letting_date'].dt.year
pavement_marking['month'] = pavement_marking['letting_date'].dt.month
pavement_marking['day'] = pavement_marking['letting_date'].dt.day
pavement_marking['quarter'] = pavement_marking['letting_date'].dt.quarter
pavement_marking['day_of_week'] = pavement_marking['letting_date'].dt.dayofweek  # Monday=0, Sunday=6

# Expand abbreviations and correct spelling in item descriptions
print("Expanding abbreviations and correcting spelling...")
pavement_marking['item_description_expanded'] = pavement_marking['item_description'].apply(expand_and_correct_description)

# Load categorized items with item_type
print("Loading item type categorization...")
categorized = pd.read_csv('output/pavement_marking_items_categorized.csv')
print(f"Categorized items loaded: {len(categorized)}")

# Merge with pavement_marking data using item_no
pavement_marking = pavement_marking.merge(
    categorized[['item_no', 'item_type']],
    left_on='item_no.',
    right_on='item_no',
    how='left'
)

# Check merge results
matched = pavement_marking['item_type'].notna().sum()
unmatched = pavement_marking['item_type'].isna().sum()
print(f"Matched item_type records: {matched:,}")
print(f"Unmatched item_type records: {unmatched:,}")

# Load project categorization with work_type
print("\nLoading project work type categorization...")
projects = pd.read_csv('output/bid_tabs_combined_categorized.csv')
print(f"Categorized projects loaded: {len(projects)}")

# Merge with pavement_marking data using proposal_id = contract_number
pavement_marking = pavement_marking.merge(
    projects[['contract_number', 'work_type']],
    left_on='proposal_id',
    right_on='contract_number',
    how='left'
)

# Check merge results
matched_work = pavement_marking['work_type'].notna().sum()
unmatched_work = pavement_marking['work_type'].isna().sum()
print(f"Matched work_type records: {matched_work:,}")
print(f"Unmatched work_type records: {unmatched_work:,}")
if matched_work > 0:
    print(f"Work type distribution:")
    print(pavement_marking['work_type'].value_counts())

# Log transform the target (unit_price) and project_qty for better prediction
pavement_marking['log_unit_price'] = np.log1p(pavement_marking['bid_unit_price'])
pavement_marking['log_project_qty'] = np.log1p(pavement_marking['project_qty'])

# Select features for modeling (use expanded description + item_type + work_type + log_project_qty)
# Note: 'units' removed - it was acting as a proxy for item type, dominating predictions
# Note: using log_project_qty instead of project_qty to handle skewness (skew=30.00)
feature_cols = ['item_description_expanded', 'item_type', 'work_type', 'primary_county', 'year', 'month', 'day', 'quarter', 
                'log_project_qty']
target_col = 'log_unit_price'

# Remove any rows with missing features
model_data = pavement_marking[feature_cols + [target_col, 'bid_unit_price', 'project_qty']].dropna()
print(f"Final dataset size: {len(model_data):,}")

# Show data distribution
print("\n" + "="*80)
print("DATA SUMMARY")
print("="*80)
print(f"\nYear range: {model_data['year'].min()} - {model_data['year'].max()}")
print(f"\nTop 10 counties by frequency:")
print(model_data['primary_county'].value_counts().head(10))
print(f"\nTop 10 items by frequency:")
print(model_data['item_description_expanded'].value_counts().head(10))

print(f"\nUnit Price Statistics:")
print(model_data['bid_unit_price'].describe())

# Time-based split: use past data to predict future
# Train on data up to 2024, test on 2025
print("\n" + "="*80)
print("TIME-BASED DATA SPLIT")
print("="*80)

train_cutoff_year = 2024
train_mask = model_data['year'] <= train_cutoff_year
test_mask = model_data['year'] > train_cutoff_year

train_data = model_data[train_mask]
test_data = model_data[test_mask]

X_train = train_data[feature_cols]
y_train = train_data[target_col]
y_actual_train = train_data['bid_unit_price']

X_test = test_data[feature_cols]
y_test = test_data[target_col]
y_actual_test = test_data['bid_unit_price']

print(f"\nTrain set (2014-{train_cutoff_year}): {len(X_train):,} records")
print(f"  Year range: {train_data['year'].min():.0f} - {train_data['year'].max():.0f}")
print(f"\nTest set ({train_cutoff_year + 1}): {len(X_test):,} records")
print(f"  Year range: {test_data['year'].min():.0f} - {test_data['year'].max():.0f}")

# Define categorical features
cat_features = ['item_description_expanded', 'item_type', 'work_type', 'primary_county', 'month', 'day', 'quarter']

print("\n" + "="*80)
print("TRAINING CATBOOST MODEL")
print("="*80)
print(f"\nCategorical features: {cat_features}")
print(f"Numerical features: {[col for col in feature_cols if col not in cat_features]}")

# Train CatBoost model
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)

# Create pools for CatBoost
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

# Train
model.fit(train_pool, eval_set=test_pool, plot=False)

# Predictions (convert back from log scale)
y_pred_train_log = model.predict(X_train)
y_pred_test_log = model.predict(X_test)

y_pred_train = np.expm1(y_pred_train_log)
y_pred_test = np.expm1(y_pred_test_log)

# Evaluation
print("\n" + "="*80)
print("MODEL PERFORMANCE - PAST PREDICTING FUTURE")
print("="*80)

print(f"\nTrain Set (2014-2024):")
print(f"  MAE:  ${mean_absolute_error(y_actual_train, y_pred_train):,.2f}")
print(f"  RMSE: ${np.sqrt(mean_squared_error(y_actual_train, y_pred_train)):,.2f}")
print(f"  R²:   {r2_score(y_actual_train, y_pred_train):.4f}")

print(f"\nTest Set (2025) - Future Predictions:")
print(f"  MAE:  ${mean_absolute_error(y_actual_test, y_pred_test):,.2f}")
print(f"  RMSE: ${np.sqrt(mean_squared_error(y_actual_test, y_pred_test)):,.2f}")
print(f"  R²:   {r2_score(y_actual_test, y_pred_test):.4f}")

# Show price trends
print("\n" + "="*80)
print("PRICE TRENDS BY YEAR")
print("="*80)
year_stats = model_data.groupby('year')['bid_unit_price'].agg(['count', 'mean', 'median', 'std'])
print("\n" + year_stats.to_string())

# Feature importance
print("\n" + "="*80)
print("FEATURE IMPORTANCE")
print("="*80)
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + feature_importance.to_string(index=False))

# Save predictions for analysis
results = X_test.copy()
results['year'] = test_data['year'].values
results['project_qty'] = test_data['project_qty'].values  # Add back original quantity for reporting
results['actual_price'] = y_actual_test.values
results['predicted_price'] = y_pred_test
results['error'] = results['actual_price'] - results['predicted_price']
results['percent_error'] = (results['error'] / results['actual_price']) * 100

results.to_csv('output/pavement_marking_predictions.csv', index=False)
print("\n✓ Predictions saved to: output/pavement_marking_predictions.csv")

# Show sample predictions
print("\n" + "="*80)
print("SAMPLE FUTURE PREDICTIONS (2025)")
print("="*80)
sample = results[['year', 'primary_county', 'item_description_expanded', 'project_qty', 'actual_price', 'predicted_price', 'percent_error']].head(20)
print("\n" + sample.to_string(index=False))

# Error analysis by item
print("\n" + "="*80)
print("ERROR ANALYSIS BY ITEM (Top 10 by frequency)")
print("="*80)
top_items = results['item_description_expanded'].value_counts().head(10).index
item_errors = results[results['item_description_expanded'].isin(top_items)].groupby('item_description_expanded').agg({
    'actual_price': 'count',
    'error': 'mean',
    'percent_error': lambda x: abs(x).mean()
}).rename(columns={
    'actual_price': 'count',
    'error': 'avg_error',
    'percent_error': 'avg_abs_pct_error'
}).sort_values('count', ascending=False)

print("\n" + item_errors.to_string())

# Save model
model.save_model('output/pavement_marking_catboost_model.cbm')
print("\n✓ Model saved to: output/pavement_marking_catboost_model.cbm")

# ============================================================================
# GENERATE VISUALIZATION PLOTS
# ============================================================================
print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)

import matplotlib.pyplot as plt
plt.style.use('default')
fig_size = (12, 8)

# 1. Feature Importance Plot
plt.figure(figsize=fig_size)
feature_importance_plot = feature_importance.head(10)
plt.barh(range(len(feature_importance_plot)), feature_importance_plot['importance'])
plt.yticks(range(len(feature_importance_plot)), feature_importance_plot['feature'])
plt.xlabel('Importance (%)')
plt.title('Feature Importance - Pavement Marking Price Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('output/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Feature importance plot saved to: output/feature_importance.png")

# 2. Price Trends by Year
plt.figure(figsize=fig_size)
year_stats_plot = year_stats.reset_index()
plt.subplot(2, 1, 1)
plt.plot(year_stats_plot['year'], year_stats_plot['mean'], 'o-', linewidth=2, markersize=6, label='Mean Price')
plt.plot(year_stats_plot['year'], year_stats_plot['median'], 's--', linewidth=2, markersize=6, label='Median Price')
plt.xlabel('Year')
plt.ylabel('Price ($)')
plt.title('Pavement Marking Price Trends by Year')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.bar(year_stats_plot['year'], year_stats_plot['count'], alpha=0.7)
plt.xlabel('Year')
plt.ylabel('Number of Records')
plt.title('Number of Records by Year')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/price_trends_by_year.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Price trends plot saved to: output/price_trends_by_year.png")

# 3. Model Performance Visualization
plt.figure(figsize=fig_size)
metrics = ['MAE', 'RMSE', 'R²']
train_scores = [
    mean_absolute_error(y_actual_train, y_pred_train),
    np.sqrt(mean_squared_error(y_actual_train, y_pred_train)),
    r2_score(y_actual_train, y_pred_train)
]
test_scores = [
    mean_absolute_error(y_actual_test, y_pred_test),
    np.sqrt(mean_squared_error(y_actual_test, y_pred_test)),
    r2_score(y_actual_test, y_pred_test)
]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, train_scores, width, label='Train (2014-2024)', alpha=0.8)
plt.bar(x + width/2, test_scores, width, label='Test (2025)', alpha=0.8)

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Performance: Train vs Test')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, (train_val, test_val) in enumerate(zip(train_scores, test_scores)):
    plt.text(i - width/2, train_val + train_val*0.01, f'{train_val:.2f}', ha='center', va='bottom')
    plt.text(i + width/2, test_val + test_val*0.01, f'{test_val:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('output/model_performance.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Model performance plot saved to: output/model_performance.png")

# 4. Actual vs Predicted Scatter Plot
plt.figure(figsize=fig_size)
sample_size = min(2000, len(y_actual_test))  # Limit points for readability
indices = np.random.choice(len(y_actual_test), sample_size, replace=False)
actual_sample = y_actual_test.iloc[indices]
pred_sample = y_pred_test[indices]

plt.scatter(actual_sample, pred_sample, alpha=0.5, s=20)
max_val = max(actual_sample.max(), pred_sample.max())
plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title(f'Actual vs Predicted Prices (Sample of {sample_size} points)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Actual vs predicted plot saved to: output/actual_vs_predicted.png")

# 5. Error Analysis by Top Items
plt.figure(figsize=(14, 8))
item_errors_plot = item_errors.head(10)
x_pos = np.arange(len(item_errors_plot))

plt.subplot(1, 2, 1)
plt.barh(x_pos, item_errors_plot['count'])
plt.yticks(x_pos, [item[:40] + '...' if len(item) > 40 else item for item in item_errors_plot.index])
plt.xlabel('Count')
plt.title('Top 10 Items by Frequency')
plt.gca().invert_yaxis()

plt.subplot(1, 2, 2)
plt.barh(x_pos, item_errors_plot['avg_abs_pct_error'])
plt.yticks(x_pos, [item[:40] + '...' if len(item) > 40 else item for item in item_errors_plot.index])
plt.xlabel('Average Absolute % Error')
plt.title('Prediction Error by Item')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('output/item_error_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Item error analysis plot saved to: output/item_error_analysis.png")

# 6. Work Type Distribution
if 'work_type' in model_data.columns:
    work_type_stats = model_data.groupby('work_type').agg({
        'bid_unit_price': ['count', 'mean']
    }).reset_index()
    work_type_stats.columns = ['work_type', 'count', 'mean_price']
    work_type_stats = work_type_stats.sort_values('count', ascending=True)
    
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.barh(work_type_stats['work_type'], work_type_stats['count'])
    plt.xlabel('Number of Records')
    plt.title('Distribution of Records by Work Type')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.barh(work_type_stats['work_type'], work_type_stats['mean_price'])
    plt.xlabel('Average Price ($)')
    plt.title('Average Price by Work Type')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/work_type_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Work type distribution plot saved to: output/work_type_distribution.png")

# 7. Item Type Distribution  
if 'item_type' in model_data.columns:
    item_type_stats = model_data.groupby('item_type').agg({
        'bid_unit_price': ['count', 'mean']
    }).reset_index()
    item_type_stats.columns = ['item_type', 'count', 'mean_price']
    
    plt.figure(figsize=fig_size)
    plt.subplot(1, 2, 1)
    plt.pie(item_type_stats['count'], labels=item_type_stats['item_type'], autopct='%1.1f%%')
    plt.title('Distribution by Item Type')
    
    plt.subplot(1, 2, 2)
    plt.bar(item_type_stats['item_type'], item_type_stats['mean_price'])
    plt.xlabel('Item Type')
    plt.ylabel('Average Price ($)')
    plt.title('Average Price by Item Type')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/item_type_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Item type distribution plot saved to: output/item_type_distribution.png")

# 8. County Statistics (Top 15 counties)
county_stats = model_data.groupby('primary_county').agg({
    'bid_unit_price': ['count', 'mean']
}).reset_index()
county_stats.columns = ['county', 'count', 'mean_price']
county_stats = county_stats.sort_values('count', ascending=True).tail(15)

plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
plt.barh(county_stats['county'], county_stats['count'])
plt.xlabel('Number of Records')
plt.title('Top 15 Counties by Number of Records')
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.barh(county_stats['county'], county_stats['mean_price'])
plt.xlabel('Average Price ($)')
plt.title('Average Price by County (Top 15)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/county_statistics.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ County statistics plot saved to: output/county_statistics.png")

# Export JSON metrics (keep these for reference)
import json
performance_metrics = {
    'model_name': 'Pavement Marking Price Prediction - CatBoost',
    'train_period': f"2014-{train_cutoff_year}",
    'test_period': f"{train_cutoff_year + 1}",
    'total_records': len(model_data),
    'train_records': len(X_train),
    'test_records': len(X_test),
    'train_metrics': {
        'mae': float(mean_absolute_error(y_actual_train, y_pred_train)),
        'rmse': float(np.sqrt(mean_squared_error(y_actual_train, y_pred_train))),
        'r2': float(r2_score(y_actual_train, y_pred_train))
    },
    'test_metrics': {
        'mae': float(mean_absolute_error(y_actual_test, y_pred_test)),
        'rmse': float(np.sqrt(mean_squared_error(y_actual_test, y_pred_test))),
        'r2': float(r2_score(y_actual_test, y_pred_test))
    },
    'model_parameters': {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'early_stopping_rounds': 50,
        'best_iteration': int(model.get_best_iteration())
    }
}

with open('output/model_performance_metrics.json', 'w') as f:
    json.dump(performance_metrics, f, indent=2)
print("✓ Performance metrics saved to: output/model_performance_metrics.json")

print("\n" + "="*80)
print("ANALYSIS COMPLETE - ALL PLOTS GENERATED")
print("="*80)
