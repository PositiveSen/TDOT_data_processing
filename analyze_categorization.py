import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load categorized data
print("Loading categorized pavement marking items...")
items = pd.read_csv('output/pavement_marking_items_categorized.csv')

# Load actual bid data to get pricing info
print("Loading TDOT data for pricing analysis...")
df = pd.read_csv('Data/TDOT_data.csv', encoding='latin-1')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Filter for pavement marking
pavement_marking = df[
    (df['item_no.'].str.startswith('716-', na=False)) & 
    (~df['item_no.'].str.startswith('716-99', na=False))
].copy()

# Convert prices
numeric_cols = ['bid_unit_price', 'project_qty']
for col in numeric_cols:
    pavement_marking[col] = pd.to_numeric(
        pavement_marking[col].astype(str).str.strip().str.replace('$', '').str.replace(',', ''), 
        errors='coerce'
    )

# Remove invalid data
pavement_marking = pavement_marking[
    (pavement_marking['bid_unit_price'].notna()) & 
    (pavement_marking['bid_unit_price'] > 0)
].copy()

# Merge categorization with pricing data
print("Merging categorization with pricing data...")
pavement_marking['item_no_clean'] = pavement_marking['item_no.'].str.strip()
items['item_no_clean'] = items['item_no'].str.strip()

categorized_pricing = pavement_marking.merge(
    items[['item_no_clean', 'material_type', 'marking_type', 'line_width', 'application_type', 
           'is_temporary', 'is_snowplowable', 'is_removal', 'is_contrast', 'is_wet_reflective']],
    on='item_no_clean',
    how='left'
)

print(f"\nMerged records: {len(categorized_pricing):,}")
print(f"Records with categories: {categorized_pricing['material_type'].notna().sum():,}")

# Save enriched dataset
categorized_pricing.to_csv('output/pavement_marking_with_categories.csv', index=False)
print("✓ Saved enriched dataset to: output/pavement_marking_with_categories.csv")

# Analysis
print("\n" + "="*80)
print("PRICING ANALYSIS BY CATEGORY")
print("="*80)

print("\n1. Average Unit Price by Material Type:")
print("-" * 80)
material_pricing = categorized_pricing.groupby('material_type')['bid_unit_price'].agg([
    'count', 'mean', 'median', 'std'
]).sort_values('mean', ascending=False)
print(material_pricing.to_string())

print("\n2. Average Unit Price by Marking Type (Top 15):")
print("-" * 80)
marking_pricing = categorized_pricing.groupby('marking_type')['bid_unit_price'].agg([
    'count', 'mean', 'median', 'std'
]).sort_values('mean', ascending=False).head(15)
print(marking_pricing.to_string())

print("\n3. Average Unit Price by Line Width:")
print("-" * 80)
width_pricing = categorized_pricing.groupby('line_width')['bid_unit_price'].agg([
    'count', 'mean', 'median', 'std'
]).sort_values('mean', ascending=False)
print(width_pricing.to_string())

print("\n4. Average Unit Price by Application Type:")
print("-" * 80)
app_pricing = categorized_pricing.groupby('application_type')['bid_unit_price'].agg([
    'count', 'mean', 'median', 'std'
]).sort_values('mean', ascending=False)
print(app_pricing.to_string())

print("\n5. Impact of Binary Features on Price:")
print("-" * 80)
binary_features = ['is_temporary', 'is_snowplowable', 'is_removal', 'is_contrast', 'is_wet_reflective']
for feature in binary_features:
    feature_pricing = categorized_pricing.groupby(feature)['bid_unit_price'].agg(['count', 'mean', 'median'])
    print(f"\n{feature}:")
    print(feature_pricing.to_string())

# Create visualizations
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(16, 12))

# 1. Material Type Distribution
plt.subplot(2, 3, 1)
material_counts = categorized_pricing['material_type'].value_counts().head(10)
plt.barh(range(len(material_counts)), material_counts.values)
plt.yticks(range(len(material_counts)), material_counts.index)
plt.xlabel('Count')
plt.title('Top 10 Material Types by Frequency')
plt.tight_layout()

# 2. Marking Type Distribution
plt.subplot(2, 3, 2)
marking_counts = categorized_pricing['marking_type'].value_counts().head(10)
plt.barh(range(len(marking_counts)), marking_counts.values)
plt.yticks(range(len(marking_counts)), marking_counts.index)
plt.xlabel('Count')
plt.title('Top 10 Marking Types by Frequency')
plt.tight_layout()

# 3. Price by Material Type
plt.subplot(2, 3, 3)
material_avg = material_pricing['mean'].sort_values(ascending=True).tail(10)
plt.barh(range(len(material_avg)), material_avg.values)
plt.yticks(range(len(material_avg)), material_avg.index)
plt.xlabel('Average Price ($)')
plt.title('Average Price by Material Type')
plt.tight_layout()

# 4. Price by Line Width
plt.subplot(2, 3, 4)
width_avg = width_pricing['mean'].sort_values(ascending=True)
plt.barh(range(len(width_avg)), width_avg.values)
plt.yticks(range(len(width_avg)), width_avg.index)
plt.xlabel('Average Price ($)')
plt.title('Average Price by Line Width')
plt.tight_layout()

# 5. Application Type Distribution
plt.subplot(2, 3, 5)
app_counts = categorized_pricing['application_type'].value_counts()
plt.barh(range(len(app_counts)), app_counts.values)
plt.yticks(range(len(app_counts)), app_counts.index)
plt.xlabel('Count')
plt.title('Application Type Distribution')
plt.tight_layout()

# 6. Price distribution boxplot by top material types
plt.subplot(2, 3, 6)
top_materials = material_counts.head(5).index
plot_data = categorized_pricing[categorized_pricing['material_type'].isin(top_materials)]
plot_data_clean = plot_data[plot_data['bid_unit_price'] < plot_data['bid_unit_price'].quantile(0.95)]
sns.boxplot(data=plot_data_clean, y='material_type', x='bid_unit_price')
plt.xlabel('Unit Price ($)')
plt.ylabel('')
plt.title('Price Distribution by Top 5 Material Types\n(95th percentile cap)')
plt.tight_layout()

plt.savefig('output/pavement_marking_category_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization to: output/pavement_marking_category_analysis.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nKey Findings:")
print(f"- Most common material: {material_counts.index[0]} ({material_counts.values[0]:,} records)")
print(f"- Most common marking: {marking_counts.index[0]} ({marking_counts.values[0]:,} records)")
print(f"- Highest avg price material: {material_pricing.index[0]} (${material_pricing['mean'].iloc[0]:,.2f})")
print(f"- Most common line width: {categorized_pricing['line_width'].value_counts().index[0]}")
print(f"- Total unique item numbers: {categorized_pricing['item_no.'].nunique()}")
