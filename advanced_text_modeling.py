"""
Advanced Text Understanding Models for Item Descriptions
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer
import re

def advanced_text_modeling(df, text_column='item_description'):
    """
    Apply advanced NLP techniques to understand item descriptions
    """
    
    print("=== ADVANCED TEXT MODELING ===")
    
    # 1. Topic Modeling with LDA
    print("1. Topic Modeling (LDA)...")
    
    # Preprocess for topic modeling
    processed_docs = preprocess_for_topics(df[text_column])
    
    # Create document-term matrix
    vectorizer = TfidfVectorizer(
        max_features=200,
        stop_words='english',
        min_df=5,
        max_df=0.7,
        ngram_range=(1,2)
    )
    doc_term_matrix = vectorizer.fit_transform(processed_docs)
    
    # Fit LDA model
    n_topics = 8  # Good for your pavement marking categories
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20
    )
    
    # Get topic distributions for each document
    topic_distributions = lda.fit_transform(doc_term_matrix)
    
    # Create topic features
    topic_df = pd.DataFrame(
        topic_distributions,
        columns=[f'topic_{i}' for i in range(n_topics)],
        index=df.index
    )
    
    # Print top words for each topic
    print("Discovered Topics:")
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-10:]]
        print(f"  Topic {topic_idx}: {', '.join(top_words[:5])}")
    
    # 2. Semantic Embeddings (Advanced)
    print("\n2. Semantic Embeddings...")
    
    # Use sentence transformer for semantic understanding
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load a lightweight model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Sample for speed (full dataset would be slow)
        sample_size = min(1000, len(df))
        sample_idx = np.random.choice(len(df), sample_size, replace=False)
        sample_descriptions = df.iloc[sample_idx][text_column].tolist()
        
        # Generate embeddings
        embeddings = model.encode(sample_descriptions)
        
        # Reduce dimensionality for model features
        from sklearn.decomposition import PCA
        pca = PCA(n_components=20)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # Create embedding features for sample
        embedding_df = pd.DataFrame(
            reduced_embeddings,
            columns=[f'embed_{i}' for i in range(20)],
            index=df.iloc[sample_idx].index
        )
        
        print(f"   Created semantic embeddings for {sample_size} samples")
        
    except ImportError:
        print("   Sentence transformers not available - install with: pip install sentence-transformers")
        embedding_df = pd.DataFrame()
    
    # 3. Text Clustering
    print("\n3. Text Clustering...")
    
    # Cluster similar item descriptions
    n_clusters = 15  # Good number for pavement marking varieties
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(doc_term_matrix)
    
    # Add cluster assignments
    cluster_df = pd.DataFrame({
        'text_cluster': clusters
    }, index=df.index)
    
    # Analyze clusters
    print("Text Clusters Found:")
    for cluster_id in range(min(5, n_clusters)):  # Show first 5
        cluster_samples = df[clusters == cluster_id][text_column].head(3)
        print(f"  Cluster {cluster_id}: {len(df[clusters == cluster_id])} items")
        for sample in cluster_samples:
            print(f"    - {sample}")
    
    return topic_df, embedding_df, cluster_df, lda, vectorizer

def preprocess_for_topics(text_series):
    """
    Preprocess text for topic modeling
    """
    processed = (
        text_series
        .str.lower()
        .str.replace(r'[^\w\s]', ' ', regex=True)
        .str.replace(r'\d+', 'NUM', regex=True)  # Replace numbers with NUM
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    return processed.tolist()

def create_smart_categories(df, text_column='item_description'):
    """
    Create intelligent categories based on text patterns
    """
    print("=== SMART CATEGORIZATION ===")
    
    categories = []
    
    for desc in df[text_column]:
        desc_lower = desc.lower()
        
        # Multi-level categorization
        if 'stop line' in desc_lower:
            categories.append('stop_line')
        elif 'arrow' in desc_lower:
            categories.append('arrow_marking')
        elif 'crosswalk' in desc_lower:
            categories.append('crosswalk')
        elif 'word' in desc_lower:
            categories.append('word_marking')
        elif 'stripe' in desc_lower or 'striping' in desc_lower:
            categories.append('striping')
        elif 'dotted' in desc_lower:
            categories.append('dotted_line')
        elif 'barrier' in desc_lower:
            categories.append('barrier_line')
        elif any(width in desc_lower for width in ['4"', '6"', '8"', '12"', '4 in', '6 in']):
            categories.append('standard_line')
        else:
            categories.append('other_marking')
    
    category_counts = pd.Series(categories).value_counts()
    print("Smart Categories:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count} items ({count/len(categories)*100:.1f}%)")
    
    return pd.Series(categories, index=df.index, name='smart_category')

if __name__ == "__main__":
    # Load sample of your data
    df = pd.read_csv('Data/TDOT_data.csv', encoding='latin-1')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Filter for pavement marking items (sample for speed)
    pavement_items = df[df['item_description'].str.contains('PAVEMENT MARKING', case=False, na=False)]
    sample = pavement_items.head(5000)  # Use sample for demonstration
    
    print(f"Analyzing {len(sample)} pavement marking descriptions...")
    
    # Create smart categories
    smart_categories = create_smart_categories(sample, 'item_description')
    
    # Apply advanced modeling
    topic_features, embedding_features, cluster_features, lda_model, vectorizer = advanced_text_modeling(sample, 'item_description')
    
    # Combine all features
    all_text_features = pd.concat([
        smart_categories,
        topic_features,
        cluster_features
    ], axis=1)
    
    if not embedding_features.empty:
        all_text_features = pd.concat([all_text_features, embedding_features], axis=1)
    
    print(f"\n✅ Advanced text analysis complete!")
    print(f"   Created {all_text_features.shape[1]} additional features")
    print(f"   Combined with pattern features = {77 + all_text_features.shape[1]} total text features")