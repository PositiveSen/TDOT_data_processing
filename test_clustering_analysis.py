#!/usr/bin/env python3
"""
Test Clustering Analysis for TDOT Construction Items
Standalone tool to explore semantic clustering of construction item descriptions
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from typing import Dict, List, Tuple
from datetime import datetime

# ML Libraries
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not available. Install with: pip install sentence-transformers")

class TDOTClusteringAnalyzer:
    """
    Analyze TDOT construction items using semantic clustering
    """
    
    def __init__(self, 
                 data_file: str = 'Data/TDOT_data.csv',
                 output_dir: str = 'clustering_analysis',
                 model_name: str = 'sentence-transformers/all-mpnet-base-v2'):
        
        self.data_file = data_file
        self.output_dir = output_dir
        self.model_name = model_name
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Data storage
        self.df = None
        self.embeddings = None
        self.clusters = None
        self.cluster_info = {}
        self.sentence_model = None
    
    def load_sentence_transformer(self) -> bool:
        """Load the sentence transformer model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return False
        
        print(f"🤖 Loading sentence transformer: {self.model_name}...")
        try:
            self.sentence_model = SentenceTransformer(self.model_name)
            print(f"  ✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"  ❌ Error loading model: {e}")
            return False
    
    def load_data(self) -> bool:
        """Load TDOT construction data with same filtering as universal model"""
        print(f"📊 Loading TDOT data from {self.data_file}...")
        
        try:
            self.df = pd.read_csv(self.data_file, encoding='latin-1')
            print(f"  Loaded {len(self.df):,} total records")
            
            # Filter for target categories (same as universal model)
            target_categories = ['302', '303', '401', '402', '403', '404', '405', '406', '407', '408', '409', '716']
            category_pattern = '|'.join(f'^{cat}' for cat in target_categories)
            mask = self.df['Item No.'].str.match(category_pattern + '-', na=False)
            self.df = self.df[mask].copy()
            
            print(f"  Filtered to target categories: {len(self.df):,} records")
            
            # Apply same data cleaning as universal model
            print(f"  Applying data cleaning...")
            
            # Clean price data
            if ' Bid Unit Price ' in self.df.columns:
                self.df[' Bid Unit Price '] = self.df[' Bid Unit Price '].astype(str).str.replace('$', '').str.replace(',', '').str.strip()
                self.df[' Bid Unit Price '] = pd.to_numeric(self.df[' Bid Unit Price '], errors='coerce')
            
            # Clean quantity data  
            self.df['Project Qty'] = pd.to_numeric(self.df['Project Qty'], errors='coerce')
            
            # Remove invalid entries (same logic as universal model)
            original_size = len(self.df)
            self.df = self.df.dropna(subset=[' Bid Unit Price ', 'Project Qty', 'Item Description'])
            self.df = self.df[
                (self.df[' Bid Unit Price '] > 0) & 
                (self.df['Project Qty'] > 0)
            ]
            removed = original_size - len(self.df)
            print(f"  Removed {removed:,} invalid entries")
            print(f"  Final dataset: {len(self.df):,} records for clustering")
            
            return True
            
        except FileNotFoundError:
            print(f"❌ Data file not found: {self.data_file}")
            return False
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def create_embeddings(self) -> bool:
        """Create embeddings for item descriptions"""
        if self.sentence_model is None:
            print("❌ No sentence transformer model available")
            return False
        
        print(f"🔄 Creating embeddings for {len(self.df):,} descriptions...")
        
        # Get unique descriptions to reduce computation
        unique_descriptions = self.df['Item Description'].unique()
        print(f"  Processing {len(unique_descriptions):,} unique descriptions")
        
        # Create embeddings
        try:
            unique_embeddings = self.sentence_model.encode(
                unique_descriptions.tolist(), 
                show_progress_bar=True,
                batch_size=32
            )
            
            # Map embeddings back to all records
            desc_to_embedding = dict(zip(unique_descriptions, unique_embeddings))
            self.embeddings = np.array([desc_to_embedding[desc] for desc in self.df['Item Description']])
            
            print(f"  ✅ Created {self.embeddings.shape} embedding matrix")
            return True
            
        except Exception as e:
            print(f"  ❌ Error creating embeddings: {e}")
            return False
    
    def perform_clustering(self, n_clusters: int = 25) -> bool:
        """Perform K-means clustering on embeddings"""
        if self.embeddings is None:
            print("❌ No embeddings available for clustering")
            return False
        
        print(f"📊 Performing K-means clustering with {n_clusters} clusters...")
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.clusters = kmeans.fit_predict(self.embeddings)
            
            print(f"  ✅ Clustering completed")
            
            # Add cluster assignments to dataframe
            self.df['cluster'] = self.clusters
            
            # Show cluster distribution
            cluster_counts = pd.Series(self.clusters).value_counts().sort_index()
            print(f"  📈 Cluster size distribution:")
            for cluster_id, count in cluster_counts.items():
                print(f"    Cluster {cluster_id:2d}: {count:,} items")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error in clustering: {e}")
            return False
    
    def analyze_clusters(self) -> Dict:
        """Analyze clusters and assign meaningful names"""
        if self.clusters is None:
            print("❌ No clusters available for analysis")
            return {}
        
        print(f"🔍 Analyzing clusters and assigning meaningful names...")
        
        cluster_analysis = {}
        
        for cluster_id in range(max(self.clusters) + 1):
            mask = self.clusters == cluster_id
            cluster_data = self.df[mask]
            
            # Get descriptions for this cluster
            descriptions = cluster_data['Item Description'].tolist()
            
            # Analyze text patterns
            all_text = ' '.join(descriptions).upper()
            
            # Find common terms
            common_terms = self.extract_common_terms(all_text)
            
            # Calculate price statistics
            if ' Bid Unit Price ' in cluster_data.columns:
                prices = cluster_data[' Bid Unit Price '].dropna()
                price_stats = {
                    'mean': float(prices.mean()) if len(prices) > 0 else 0,
                    'median': float(prices.median()) if len(prices) > 0 else 0,
                    'min': float(prices.min()) if len(prices) > 0 else 0,
                    'max': float(prices.max()) if len(prices) > 0 else 0,
                    'std': float(prices.std()) if len(prices) > 0 else 0
                }
            else:
                price_stats = {}
            
            # Generate meaningful cluster name
            cluster_name = self.generate_cluster_name(common_terms, descriptions[0] if descriptions else f"Cluster_{cluster_id}")
            
            # Estimate complexity and characteristics
            complexity = self.estimate_complexity(common_terms, price_stats)
            material_type = self.identify_material_type(common_terms)
            work_type = self.identify_work_type(common_terms)
            
            cluster_analysis[cluster_id] = {
                'name': cluster_name,
                'size': len(cluster_data),
                'common_terms': common_terms[:10],  # Top 10 terms
                'price_stats': price_stats,
                'complexity_score': complexity,
                'material_type': material_type,
                'work_type': work_type,
                'sample_descriptions': self.get_diverse_samples(descriptions, 5),  # Get diverse samples
                'item_numbers': cluster_data['Item No.'].unique()[:10].tolist()  # Sample item numbers
            }
            
            print(f"  Cluster {cluster_id:2d}: {cluster_name} ({len(cluster_data):,} items)")
        
        self.cluster_info = cluster_analysis
        return cluster_analysis
    
    def extract_common_terms(self, text: str) -> List[str]:
        """Extract common construction terms from text"""
        # Remove special characters and split into words
        words = re.findall(r'\b[A-Z]{2,}\b', text)  # Get uppercase words (construction terms)
        
        # Count word frequency
        word_counts = Counter(words)
        
        # Filter out very common but not meaningful words
        stop_words = {'THE', 'AND', 'OR', 'OF', 'FOR', 'WITH', 'BY', 'TO', 'IN', 'ON', 'AT'}
        filtered_counts = {word: count for word, count in word_counts.items() if word not in stop_words}
        
        # Return most common terms
        return [word for word, count in sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)]
    
    def get_diverse_samples(self, descriptions: List[str], n_samples: int = 5) -> List[str]:
        """Get maximally diverse unique sample descriptions from a cluster"""
        # Get unique descriptions first
        unique_descriptions = list(set(descriptions))
        
        if len(unique_descriptions) <= n_samples:
            return unique_descriptions
        
        # Try to maximize diversity by selecting descriptions with different key words
        diverse_samples = []
        used_keywords = set()
        
        # Sort by length for variety
        unique_descriptions.sort(key=lambda x: (len(x), x))
        
        for desc in unique_descriptions:
            if len(diverse_samples) >= n_samples:
                break
                
            # Extract key words from this description
            words = set(desc.upper().split())
            # Remove very common construction words to focus on differentiating terms
            common_words = {'PAVEMENT', 'MARKING', 'LINE', 'PLASTIC', 'THE', 'AND', 'OR', 'OF', 'FOR'}
            key_words = words - common_words
            
            # Check if this description adds new key words
            if not key_words or not key_words.issubset(used_keywords):
                diverse_samples.append(desc)
                used_keywords.update(key_words)
        
        # If we still don't have enough samples, fill with remaining unique descriptions
        remaining = [d for d in unique_descriptions if d not in diverse_samples]
        diverse_samples.extend(remaining[:n_samples - len(diverse_samples)])
        
        return diverse_samples[:n_samples]
    
    def generate_cluster_name(self, common_terms: List[str], sample_description: str) -> str:
        """Generate meaningful cluster name based on analysis"""
        if not common_terms:
            return "General_Construction"
        
        # Common construction naming patterns
        naming_patterns = {
            # Materials
            'CONCRETE': 'Concrete_Work',
            'ASPHALT': 'Asphalt_Work', 
            'STEEL': 'Steel_Work',
            'PLASTIC': 'Plastic_Materials',
            'THERMOPLASTIC': 'Thermoplastic_Materials',
            'AGGREGATE': 'Aggregate_Materials',
            
            # Work types
            'MARKING': 'Pavement_Marking',
            'STRIPE': 'Striping_Work',
            'PAINT': 'Paint_Materials',
            'BRIDGE': 'Bridge_Work',
            'DRAINAGE': 'Drainage_Systems',
            'PIPE': 'Pipe_Work',
            'REMOVAL': 'Removal_Work',
            'INSTALLATION': 'Installation_Work',
            
            # Specific items
            'STOP': 'Stop_Line_Markings',
            'ARROW': 'Arrow_Markings',
            'BARRIER': 'Barrier_Systems',
            'SIGN': 'Signage_Work'
        }
        
        # Try to match patterns
        for term in common_terms[:5]:  # Check top 5 terms
            if term in naming_patterns:
                return naming_patterns[term]
        
        # Fallback: use most common term
        primary_term = common_terms[0] if common_terms else 'UNKNOWN'
        return f"{primary_term.title()}_Work"
    
    def estimate_complexity(self, common_terms: List[str], price_stats: Dict) -> int:
        """Estimate complexity score (1-5) based on terms and pricing"""
        complexity_score = 1  # Base complexity
        
        # High complexity indicators
        high_complexity_terms = ['BRIDGE', 'PRESTRESSED', 'REINFORCED', 'CUSTOM', 'SPECIALIZED', 'ENHANCED']
        medium_complexity_terms = ['CLASS', 'GRADE', 'SPECIAL', 'MODIFIED', 'INSTALLATION']
        
        for term in common_terms[:10]:
            if term in high_complexity_terms:
                complexity_score += 2
            elif term in medium_complexity_terms:
                complexity_score += 1
        
        # Adjust based on price if available
        if price_stats and 'mean' in price_stats:
            mean_price = price_stats['mean']
            if mean_price > 500:
                complexity_score += 2
            elif mean_price > 100:
                complexity_score += 1
        
        return min(complexity_score, 5)  # Cap at 5
    
    def identify_material_type(self, common_terms: List[str]) -> str:
        """Identify primary material type"""
        material_terms = {
            'CONCRETE': 'Concrete',
            'ASPHALT': 'Asphalt', 
            'STEEL': 'Steel',
            'PLASTIC': 'Plastic',
            'THERMOPLASTIC': 'Thermoplastic',
            'AGGREGATE': 'Aggregate',
            'ALUMINUM': 'Aluminum'
        }
        
        for term in common_terms[:5]:
            if term in material_terms:
                return material_terms[term]
        
        return 'Mixed/Other'
    
    def identify_work_type(self, common_terms: List[str]) -> str:
        """Identify primary work type"""
        work_terms = {
            'MARKING': 'Marking',
            'REMOVAL': 'Removal',
            'INSTALLATION': 'Installation',
            'REPAIR': 'Repair',
            'MAINTENANCE': 'Maintenance',
            'BRIDGE': 'Bridge_Construction',
            'DRAINAGE': 'Drainage',
            'STRIPE': 'Striping',
            'PAINT': 'Painting'
        }
        
        for term in common_terms[:5]:
            if term in work_terms:
                return work_terms[term]
        
        return 'General_Construction'
    
    def create_visualizations(self):
        """Create comprehensive visualizations of clusters"""
        if self.embeddings is None or self.clusters is None:
            print("❌ No clustering data available for visualization")
            return
        
        print(f"📊 Creating cluster visualizations...")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. PCA visualization
        self.plot_pca_clusters()
        
        # 2. t-SNE visualization (for smaller datasets)
        if len(self.df) <= 5000:
            self.plot_tsne_clusters()
        else:
            print("  ⚠️  Skipping t-SNE (dataset too large)")
        
        # 3. Cluster characteristics
        self.plot_cluster_characteristics()
        
        # 4. Price distribution by cluster
        if ' Bid Unit Price ' in self.df.columns:
            self.plot_price_by_cluster()
    
    def plot_pca_clusters(self):
        """Create PCA plot of clusters"""
        print("  Creating PCA visualization...")
        
        # Reduce to 2D with PCA
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(self.embeddings)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot each cluster with different colors
        unique_clusters = np.unique(self.clusters)
        colors = sns.color_palette("husl", len(unique_clusters))
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = self.clusters == cluster_id
            cluster_name = self.cluster_info.get(cluster_id, {}).get('name', f'Cluster_{cluster_id}')
            
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                      c=[colors[i]], label=f'{cluster_name} ({mask.sum()})', 
                      alpha=0.6, s=30)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title('TDOT Construction Items - Semantic Clustering (PCA)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pca_clusters.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_tsne_clusters(self):
        """Create t-SNE plot of clusters"""
        print("  Creating t-SNE visualization...")
        
        # Reduce to 2D with t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.df)//4))
        embeddings_2d = tsne.fit_transform(self.embeddings)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot each cluster
        unique_clusters = np.unique(self.clusters)
        colors = sns.color_palette("husl", len(unique_clusters))
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = self.clusters == cluster_id
            cluster_name = self.cluster_info.get(cluster_id, {}).get('name', f'Cluster_{cluster_id}')
            
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                      c=[colors[i]], label=f'{cluster_name} ({mask.sum()})', 
                      alpha=0.6, s=30)
        
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('TDOT Construction Items - Semantic Clustering (t-SNE)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'tsne_clusters.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_cluster_characteristics(self):
        """Plot cluster characteristics"""
        print("  Creating cluster characteristics plot...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        cluster_ids = list(self.cluster_info.keys())
        cluster_names = [self.cluster_info[cid]['name'] for cid in cluster_ids]
        cluster_sizes = [self.cluster_info[cid]['size'] for cid in cluster_ids]
        complexity_scores = [self.cluster_info[cid]['complexity_score'] for cid in cluster_ids]
        
        # Cluster sizes
        ax1.barh(cluster_names, cluster_sizes)
        ax1.set_xlabel('Number of Items')
        ax1.set_title('Cluster Sizes')
        ax1.tick_params(axis='y', labelsize=8)
        
        # Complexity scores
        ax2.barh(cluster_names, complexity_scores)
        ax2.set_xlabel('Complexity Score (1-5)')
        ax2.set_title('Estimated Complexity by Cluster')
        ax2.tick_params(axis='y', labelsize=8)
        
        # Material type distribution
        material_types = [self.cluster_info[cid]['material_type'] for cid in cluster_ids]
        material_counts = Counter(material_types)
        
        ax3.pie(material_counts.values(), labels=material_counts.keys(), autopct='%1.1f%%')
        ax3.set_title('Material Type Distribution')
        
        # Work type distribution  
        work_types = [self.cluster_info[cid]['work_type'] for cid in cluster_ids]
        work_counts = Counter(work_types)
        
        ax4.pie(work_counts.values(), labels=work_counts.keys(), autopct='%1.1f%%')
        ax4.set_title('Work Type Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cluster_characteristics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_price_by_cluster(self):
        """Plot price distribution by cluster"""
        print("  Creating price distribution plot...")
        
        # Prepare price data by cluster
        cluster_prices = []
        cluster_labels = []
        
        for cluster_id in sorted(self.cluster_info.keys()):
            mask = self.clusters == cluster_id
            prices = self.df[mask][' Bid Unit Price '].dropna()
            
            if len(prices) > 5:  # Only include clusters with enough price data
                cluster_prices.append(prices)
                cluster_labels.append(self.cluster_info[cluster_id]['name'])
        
        if not cluster_prices:
            print("    ⚠️  No price data available for plotting")
            return
        
        # Create box plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Box plot (normal scale)
        ax1.boxplot(cluster_prices, labels=cluster_labels)
        ax1.set_ylabel('Unit Price ($)')
        ax1.set_title('Price Distribution by Cluster')
        ax1.tick_params(axis='x', rotation=45, labelsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Box plot (log scale)
        ax2.boxplot(cluster_prices, labels=cluster_labels)
        ax2.set_ylabel('Unit Price ($)')
        ax2.set_yscale('log')
        ax2.set_title('Price Distribution by Cluster (Log Scale)')
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'price_by_cluster.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save clustering results to files"""
        print(f"💾 Saving clustering results...")
        
        # Save detailed cluster analysis
        with open(os.path.join(self.output_dir, 'cluster_analysis.json'), 'w') as f:
            # Convert numpy types to regular Python types for JSON serialization
            json_friendly_analysis = {}
            for cluster_id, info in self.cluster_info.items():
                json_friendly_analysis[str(cluster_id)] = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in info.items()
                }
            
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_items': len(self.df),
                'num_clusters': len(self.cluster_info),
                'clusters': json_friendly_analysis
            }, f, indent=2, default=str)
        
        # Save cluster assignments with item details
        cluster_df = self.df.copy()
        cluster_df['cluster_id'] = self.clusters
        cluster_df['cluster_name'] = [self.cluster_info[cid]['name'] for cid in self.clusters]
        cluster_df['complexity_score'] = [self.cluster_info[cid]['complexity_score'] for cid in self.clusters]
        cluster_df['material_type'] = [self.cluster_info[cid]['material_type'] for cid in self.clusters]
        cluster_df['work_type'] = [self.cluster_info[cid]['work_type'] for cid in self.clusters]
        
        cluster_df.to_csv(os.path.join(self.output_dir, 'clustered_items.csv'), index=False)
        
        # Save cluster summary
        summary_rows = []
        for cluster_id, info in self.cluster_info.items():
            summary_rows.append({
                'cluster_id': cluster_id,
                'cluster_name': info['name'],
                'size': info['size'],
                'complexity_score': info['complexity_score'],
                'material_type': info['material_type'],
                'work_type': info['work_type'],
                'avg_price': info['price_stats'].get('mean', 0),
                'median_price': info['price_stats'].get('median', 0),
                'common_terms': ', '.join(info['common_terms'][:5])
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(self.output_dir, 'cluster_summary.csv'), index=False)
        
        print(f"  ✅ Results saved to {self.output_dir}/")
        print(f"    - cluster_analysis.json: Detailed analysis")
        print(f"    - clustered_items.csv: All items with cluster assignments")  
        print(f"    - cluster_summary.csv: Cluster summary table")
    
    def print_cluster_examples(self):
        """Print examples from each cluster"""
        print(f"\n" + "="*80)
        print("CLUSTER EXAMPLES AND ANALYSIS")
        print("="*80)
        
        for cluster_id in sorted(self.cluster_info.keys()):
            info = self.cluster_info[cluster_id]
            
            print(f"\n🏷️  CLUSTER {cluster_id}: {info['name']}")
            print(f"   Size: {info['size']:,} items")
            print(f"   Complexity: {info['complexity_score']}/5")
            print(f"   Material: {info['material_type']}")
            print(f"   Work Type: {info['work_type']}")
            
            if info['price_stats']:
                print(f"   Avg Price: ${info['price_stats']['mean']:.2f}")
                print(f"   Price Range: ${info['price_stats']['min']:.2f} - ${info['price_stats']['max']:.2f}")
            
            print(f"   Common Terms: {', '.join(info['common_terms'][:5])}")
            print(f"   Examples:")
            for i, desc in enumerate(info['sample_descriptions'][:3], 1):
                print(f"     {i}. {desc}")


def main():
    """Run clustering analysis"""
    print("TDOT Construction Items - Clustering Analysis")
    print("="*50)
    
    # Initialize analyzer
    analyzer = TDOTClusteringAnalyzer(
        data_file='Data/TDOT_data.csv',  # Same path as universal model
        output_dir='clustering_analysis'
    )
    
    # Load sentence transformer
    if not analyzer.load_sentence_transformer():
        print("❌ Cannot proceed without sentence transformer model")
        print("💡 Install with: pip install sentence-transformers")
        return
    
    # Load data (full 112k dataset with same filtering as universal model)
    if not analyzer.load_data():
        return
    
    # Create embeddings
    if not analyzer.create_embeddings():
        return
    
    # Perform clustering
    n_clusters = 20  # Experiment with different numbers
    if not analyzer.perform_clustering(n_clusters=n_clusters):
        return
    
    # Analyze clusters
    cluster_analysis = analyzer.analyze_clusters()
    
    # Print examples
    analyzer.print_cluster_examples()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Save results
    analyzer.save_results()
    
    print(f"\n" + "="*50)
    print("CLUSTERING ANALYSIS COMPLETED")
    print("="*50)
    print(f"📊 Analyzed {len(analyzer.df):,} construction items")
    print(f"🏷️  Created {len(cluster_analysis)} semantic clusters")
    print(f"📁 Results saved to: clustering_analysis/")
    print(f"🎨 Visualizations created")
    
    # Show cluster summary
    print(f"\n📋 CLUSTER SUMMARY:")
    for cluster_id, info in cluster_analysis.items():
        print(f"  {cluster_id:2d}. {info['name']:<25} ({info['size']:>4,} items, {info['complexity_score']}/5 complexity)")


if __name__ == "__main__":
    main()