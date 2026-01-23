import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

def extract_text_features(df, text_column='item_description'):
    """
    Extract comprehensive features from item description text
    """
    
    # Create a copy to work with
    text_df = df.copy()
    
    print("=== TEXT FEATURE EXTRACTION ===")
    print(f"Processing {len(text_df)} item descriptions...")
    
    # 1. Basic text statistics
    print("\n1. Extracting basic text features...")
    text_df['desc_length'] = text_df[text_column].str.len()
    text_df['desc_word_count'] = text_df[text_column].str.split().str.len()
    text_df['desc_upper_ratio'] = text_df[text_column].str.count(r'[A-Z]') / text_df['desc_length']
    
    # 2. Pattern-based features (VERY useful for your data)
    print("2. Extracting pattern-based features...")
    
    # Material type
    text_df['material_plastic'] = text_df[text_column].str.contains('PLASTIC', case=False).astype(int)
    text_df['material_painted'] = text_df[text_column].str.contains('PAINTED', case=False).astype(int)
    text_df['material_thermo'] = text_df[text_column].str.contains('THERMO', case=False).astype(int)
    
    # Line specifications
    text_df['line_width'] = text_df[text_column].str.extract(r'(\d+)"?\s*(?:IN|INCH)?\s*LINE', flags=re.IGNORECASE)
    text_df['line_width'] = pd.to_numeric(text_df['line_width'], errors='coerce')
    text_df['has_line_width'] = text_df['line_width'].notna().astype(int)
    
    # Marking types
    text_df['is_stop_line'] = text_df[text_column].str.contains('STOP LINE', case=False).astype(int)
    text_df['is_arrow'] = text_df[text_column].str.contains('ARROW', case=False).astype(int)
    text_df['is_crosswalk'] = text_df[text_column].str.contains('CROSS.?WALK', case=False).astype(int)
    text_df['is_striping'] = text_df[text_column].str.contains('STRIP', case=False).astype(int)
    text_df['is_dotted'] = text_df[text_column].str.contains('DOTTED', case=False).astype(int)
    text_df['is_word_marking'] = text_df[text_column].str.contains('WORD', case=False).astype(int)
    
    # Enhanced/special features
    text_df['is_enhanced'] = text_df[text_column].str.contains('ENHANCED', case=False).astype(int)
    text_df['is_removable'] = text_df[text_column].str.contains('REMOVABLE', case=False).astype(int)
    text_df['has_thickness'] = text_df[text_column].str.contains(r'\d+\s*mil', case=False).astype(int)
    
    # 3. TF-IDF Features for semantic similarity
    print("3. Creating TF-IDF features...")
    
    # Preprocess text for TF-IDF
    processed_text = (text_df[text_column]
                     .str.lower()
                     .str.replace(r'[^\w\s]', ' ', regex=True)
                     .str.replace(r'\s+', ' ', regex=True)
                     .str.strip())
    
    # Create TF-IDF features
    tfidf = TfidfVectorizer(
        max_features=50,  # Top 50 most important terms
        stop_words='english',
        ngram_range=(1, 2),  # Include bigrams
        min_df=2,  # Must appear in at least 2 documents
        max_df=0.8  # Remove terms in >80% of documents
    )
    
    tfidf_matrix = tfidf.fit_transform(processed_text)
    
    # Convert to DataFrame
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f'tfidf_{term}' for term in tfidf.get_feature_names_out()],
        index=text_df.index
    )
    
    # 4. Semantic clusters using SVD
    print("4. Creating semantic clusters...")
    svd = TruncatedSVD(n_components=10, random_state=42)
    semantic_features = svd.fit_transform(tfidf_matrix)
    
    semantic_df = pd.DataFrame(
        semantic_features,
        columns=[f'semantic_dim_{i}' for i in range(10)],
        index=text_df.index
    )
    
    # 5. Combine all features
    print("5. Combining all text features...")
    
    # Pattern features
    pattern_features = [
        'desc_length', 'desc_word_count', 'desc_upper_ratio',
        'material_plastic', 'material_painted', 'material_thermo',
        'line_width', 'has_line_width',
        'is_stop_line', 'is_arrow', 'is_crosswalk', 'is_striping', 
        'is_dotted', 'is_word_marking',
        'is_enhanced', 'is_removable', 'has_thickness'
    ]
    
    # Combine all features
    final_df = pd.concat([
        text_df[pattern_features],
        tfidf_df,
        semantic_df
    ], axis=1)
    
    # Fill missing values
    final_df = final_df.fillna(0)
    
    print(f"\n✅ Created {final_df.shape[1]} text features!")
    print(f"   - {len(pattern_features)} pattern-based features")
    print(f"   - {tfidf_df.shape[1]} TF-IDF features") 
    print(f"   - {semantic_df.shape[1]} semantic features")
    
    # Feature importance analysis
    print("\n=== PATTERN FEATURE ANALYSIS ===")
    for feature in pattern_features[:10]:  # Show first 10
        if feature in ['desc_length', 'desc_word_count', 'desc_upper_ratio', 'line_width']:
            print(f"{feature}: mean={final_df[feature].mean():.2f}, std={final_df[feature].std():.2f}")
        else:
            print(f"{feature}: {final_df[feature].sum()} items ({final_df[feature].mean()*100:.1f}%)")
    
    return final_df, tfidf, svd

def analyze_text_patterns(df, text_column='item_description'):
    """
    Analyze common patterns in text data
    """
    print("=== TEXT PATTERN ANALYSIS ===")
    
    # Most common terms
    all_text = ' '.join(df[text_column].str.lower())
    words = re.findall(r'\b\w+\b', all_text)
    word_counts = pd.Series(words).value_counts()
    
    print("Most common terms:")
    for word, count in word_counts.head(15).items():
        if len(word) > 3:  # Skip short words
            print(f"  {word}: {count}")
    
    # Common patterns
    print("\nCommon patterns:")
    patterns = {
        'Line width': r'\b(\d+)"?\s*(?:IN|INCH)?\s*LINE',
        'Thickness': r'(\d+)\s*mil',
        'Material': r'(PLASTIC|PAINTED|THERMO)',
        'Type': r'(STOP LINE|ARROW|CROSSWALK|STRIPING)'
    }
    
    for pattern_name, pattern in patterns.items():
        matches = df[text_column].str.extractall(pattern, flags=re.IGNORECASE)
        if not matches.empty:
            print(f"  {pattern_name}: {matches[0].value_counts().head(5).to_dict()}")

if __name__ == "__main__":
    # Load your data
    print("Loading TDOT data...")
    df = pd.read_csv('Data/TDOT_data.csv', encoding='latin-1')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Filter for pavement marking items
    pavement_items = df[df['item_description'].str.contains('PAVEMENT MARKING|MARKING|STRIPE|LINE', case=False, na=False)]
    
    print(f"Found {len(pavement_items)} pavement marking items")
    
    # Analyze patterns
    analyze_text_patterns(pavement_items, 'item_description')
    
    # Extract features
    text_features, tfidf_model, svd_model = extract_text_features(pavement_items, 'item_description')
    
    # Save results
    text_features.to_csv('output/item_description_text_features.csv', index=False)
    
    print(f"\n✅ Text features saved to 'output/item_description_text_features.csv'")
    print(f"   Shape: {text_features.shape}")
    print(f"   Ready to use in your ML model!")