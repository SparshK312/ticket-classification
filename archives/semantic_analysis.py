#!/usr/bin/env python3
"""
SEMANTIC ANALYSIS - Phase 2 of Ticket Classification

This script processes the 3,208 tickets that couldn't be classified by hardcoded rules
and uses sentence-transformers + clustering to find semantic groups representing 
core problem statements.

Process:
1. Load unclassified tickets from improved classification results
2. Encode ticket descriptions using sentence-transformers
3. Apply DBSCAN clustering to find semantic groups
4. Analyze clusters to identify core problem statements
5. Generate final semantic taxonomy
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Semantic analysis libraries
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

class SemanticAnalyzer:
    """Semantic analysis for unclassified tickets using sentence transformers."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize the semantic analyzer."""
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        
        # Load sentence transformer model
        self.logger.info(f"Loading sentence transformer model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.logger.info("Sentence transformer model loaded successfully")
        
        # Clustering parameters (will be tuned)
        self.dbscan_eps = 0.3
        self.dbscan_min_samples = 3
        
    def load_unclassified_tickets(self, results_file: Path) -> pd.DataFrame:
        """Load unclassified tickets from classification results."""
        self.logger.info(f"Loading unclassified tickets from {results_file}")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        unclassified_tickets = results['unclassified_tickets']
        self.logger.info(f"Loaded {len(unclassified_tickets)} unclassified tickets")
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(unclassified_tickets)
        
        # Load original ticket data to get full descriptions
        consolidated_file = Path('data/processed/consolidated_tickets.csv')
        if consolidated_file.exists():
            df_original = pd.read_csv(consolidated_file)
            
            # Merge to get full ticket information
            df = df.merge(df_original, 
                         left_on='ticket_index', 
                         right_index=True, 
                         how='left',
                         suffixes=('', '_orig'))
            
            self.logger.info("Successfully merged with original ticket data")
        else:
            self.logger.warning("Original ticket data not found - using limited information")
        
        return df
    
    def prepare_text_for_encoding(self, row: pd.Series) -> str:
        """Prepare text for semantic encoding."""
        # Get short description and full description
        short_desc = str(row.get('Short description', ''))
        description = str(row.get('Description', ''))
        
        # Clean up
        short_desc = short_desc.strip()
        description = description.strip()
        
        # Strategy: Give more weight to short description (more focused)
        # but include full description for context
        if len(description) > 0 and description.lower() != 'nan':
            # Combine with emphasis on short description
            combined = f"{short_desc}. {short_desc}. {description}"
        else:
            # Only short description available
            combined = f"{short_desc}. {short_desc}."
        
        return combined.strip()
    
    def encode_tickets(self, df: pd.DataFrame) -> np.ndarray:
        """Encode ticket descriptions using sentence transformers."""
        self.logger.info("Preparing text for encoding")
        
        # Prepare texts for encoding
        texts = df.apply(self.prepare_text_for_encoding, axis=1).tolist()
        
        self.logger.info(f"Encoding {len(texts)} ticket descriptions...")
        
        # Encode all texts
        embeddings = self.encoder.encode(texts, 
                                       show_progress_bar=True,
                                       convert_to_tensor=False)
        
        # Normalize embeddings to unit length for better cosine similarity
        embeddings = normalize(embeddings, norm='l2')
        
        self.logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        
        return embeddings, texts
    
    def find_optimal_clustering_parameters(self, embeddings: np.ndarray) -> dict:
        """Find optimal DBSCAN parameters using various metrics."""
        self.logger.info("Finding optimal clustering parameters")
        
        # Calculate pairwise cosine distances for parameter tuning
        # Since embeddings are normalized, cosine distance = 1 - cosine_similarity
        from sklearn.metrics.pairwise import cosine_distances
        distances = cosine_distances(embeddings)
        
        # Test different eps values
        eps_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        min_samples_values = [2, 3, 4, 5]
        
        results = []
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                clustering = DBSCAN(eps=eps, 
                                  min_samples=min_samples, 
                                  metric='precomputed')
                labels = clustering.fit_predict(distances)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                if n_clusters > 0:
                    results.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'noise_ratio': n_noise / len(labels),
                        'avg_cluster_size': (len(labels) - n_noise) / n_clusters if n_clusters > 0 else 0
                    })
        
        # Find best parameters (balance cluster count, noise ratio, cluster size)
        results_df = pd.DataFrame(results)
        
        # Score based on: reasonable number of clusters, low noise, good cluster sizes
        results_df['score'] = (
            np.log(results_df['n_clusters'] + 1) * 0.3 -  # Prefer more clusters (but not too many)
            results_df['noise_ratio'] * 0.4 +  # Penalize high noise
            np.log(results_df['avg_cluster_size'] + 1) * 0.3  # Prefer reasonable cluster sizes
        )
        
        best_params = results_df.loc[results_df['score'].idxmax()]
        
        self.logger.info(f"Optimal parameters found:")
        self.logger.info(f"  eps: {best_params['eps']}")
        self.logger.info(f"  min_samples: {best_params['min_samples']}")
        self.logger.info(f"  Expected clusters: {best_params['n_clusters']}")
        self.logger.info(f"  Expected noise ratio: {best_params['noise_ratio']:.2%}")
        
        return {
            'eps': best_params['eps'],
            'min_samples': int(best_params['min_samples']),
            'expected_clusters': int(best_params['n_clusters']),
            'expected_noise_ratio': best_params['noise_ratio'],
            'parameter_search_results': results_df
        }
    
    def apply_clustering(self, embeddings: np.ndarray, eps: float = None, min_samples: int = None) -> np.ndarray:
        """Apply DBSCAN clustering to embeddings."""
        if eps is None:
            eps = self.dbscan_eps
        if min_samples is None:
            min_samples = self.dbscan_min_samples
            
        self.logger.info(f"Applying DBSCAN clustering (eps={eps}, min_samples={min_samples})")
        
        # Calculate cosine distance matrix
        from sklearn.metrics.pairwise import cosine_distances
        distance_matrix = cosine_distances(embeddings)
        
        # Apply DBSCAN with precomputed distance matrix
        clustering = DBSCAN(eps=eps, 
                          min_samples=min_samples, 
                          metric='precomputed')
        labels = clustering.fit_predict(distance_matrix)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        self.logger.info(f"Clustering results:")
        self.logger.info(f"  Clusters found: {n_clusters}")
        self.logger.info(f"  Noise points: {n_noise} ({n_noise/len(labels):.1%})")
        
        return labels
    
    def analyze_semantic_clusters(self, df: pd.DataFrame, labels: np.ndarray, texts: list) -> dict:
        """Analyze the semantic clusters to identify core problem statements."""
        self.logger.info("Analyzing semantic clusters")
        
        df_analysis = df.copy()
        df_analysis['cluster_label'] = labels
        df_analysis['prepared_text'] = texts
        
        clusters = {}
        
        # Analyze each cluster
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Handle noise separately
        
        for cluster_id in sorted(unique_labels):
            cluster_tickets = df_analysis[df_analysis['cluster_label'] == cluster_id]
            
            if len(cluster_tickets) == 0:
                continue
            
            # Extract key information about this cluster
            cluster_info = {
                'cluster_id': int(cluster_id),
                'size': len(cluster_tickets),
                'categories': cluster_tickets['category'].value_counts().to_dict(),
                'sample_descriptions': cluster_tickets['short_description'].head(10).tolist(),
                'sample_full_text': cluster_tickets['prepared_text'].head(5).tolist(),
                'ticket_indices': cluster_tickets['ticket_index'].tolist()
            }
            
            # Extract common keywords/patterns
            all_keywords = []
            for keywords in cluster_tickets['keywords']:
                if isinstance(keywords, list):
                    all_keywords.extend(keywords)
            
            if all_keywords:
                cluster_info['common_keywords'] = dict(Counter(all_keywords).most_common(15))
            else:
                cluster_info['common_keywords'] = {}
            
            clusters[f"semantic_cluster_{cluster_id}"] = cluster_info
        
        # Handle noise points
        noise_tickets = df_analysis[df_analysis['cluster_label'] == -1]
        if len(noise_tickets) > 0:
            clusters['noise'] = {
                'cluster_id': -1,
                'size': len(noise_tickets),
                'categories': noise_tickets['category'].value_counts().to_dict(),
                'sample_descriptions': noise_tickets['short_description'].head(20).tolist(),
                'ticket_indices': noise_tickets['ticket_index'].tolist(),
                'note': 'These tickets could not be grouped semantically - may represent unique issues'
            }
        
        # Overall statistics
        cluster_stats = {
            'total_clusters': len(unique_labels),
            'total_tickets_clustered': len(df_analysis[df_analysis['cluster_label'] != -1]),
            'noise_points': len(noise_tickets),
            'clustering_rate': (len(df_analysis[df_analysis['cluster_label'] != -1]) / len(df_analysis)) * 100,
            'avg_cluster_size': np.mean([info['size'] for info in clusters.values() if info.get('cluster_id', -1) != -1])
        }
        
        return {
            'clusters': clusters,
            'statistics': cluster_stats,
            'cluster_analysis_df': df_analysis[['ticket_index', 'category', 'short_description', 'cluster_label']]
        }
    
    def create_visualization(self, embeddings: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
        """Create visualizations of the semantic clusters."""
        self.logger.info("Creating cluster visualizations")
        
        # Use t-SNE for 2D visualization
        # Perplexity should be smaller than number of samples and typically between 5-50
        perplexity = min(30, max(5, len(embeddings) // 3))
        if len(embeddings) <= 5:
            perplexity = len(embeddings) - 1
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels, cmap='tab20', alpha=0.7)
        plt.title('Semantic Clusters of Unclassified Tickets (t-SNE)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        
        # Add legend for cluster IDs
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        if len(unique_labels) <= 20:  # Only show legend if reasonable number of clusters
            plt.colorbar(scatter, label='Cluster ID')
        
        plt.tight_layout()
        viz_file = output_dir / 'semantic_clusters_visualization.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualization saved to: {viz_file}")

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def main():
    """Main semantic analysis pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # File paths
    results_file = Path('outputs/improved_classification_results.json')
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    if not results_file.exists():
        print(f"❌ Error: Classification results file not found at {results_file}")
        print("   Please run improved_semantic_grouping_clean.py first.")
        return
    
    print("="*80)
    print("SEMANTIC ANALYSIS - PHASE 2: UNCLASSIFIED TICKET CLUSTERING")
    print("="*80)
    
    # Initialize semantic analyzer
    analyzer = SemanticAnalyzer()
    
    # Step 1: Load unclassified tickets
    print(f"\n🔄 STEP 1: Loading unclassified tickets")
    df_unclassified = analyzer.load_unclassified_tickets(results_file)
    print(f"   📊 Loaded {len(df_unclassified)} unclassified tickets")
    
    # Show category breakdown
    print(f"\n📋 CATEGORY BREAKDOWN:")
    category_counts = df_unclassified['category'].value_counts()
    for category, count in category_counts.head(10).items():
        print(f"   {category}: {count:,} tickets")
    
    # Step 2: Encode tickets using sentence transformers
    print(f"\n🔄 STEP 2: Encoding tickets with sentence transformers")
    embeddings, texts = analyzer.encode_tickets(df_unclassified)
    
    # Step 3: Find optimal clustering parameters
    print(f"\n🔄 STEP 3: Finding optimal clustering parameters")
    optimal_params = analyzer.find_optimal_clustering_parameters(embeddings)
    
    # Step 4: Apply clustering
    print(f"\n🔄 STEP 4: Applying semantic clustering")
    labels = analyzer.apply_clustering(embeddings, 
                                     eps=optimal_params['eps'],
                                     min_samples=optimal_params['min_samples'])
    
    # Step 5: Analyze clusters
    print(f"\n🔄 STEP 5: Analyzing semantic clusters")
    cluster_analysis = analyzer.analyze_semantic_clusters(df_unclassified, labels, texts)
    
    # Step 6: Create visualizations
    print(f"\n🔄 STEP 6: Creating visualizations")
    analyzer.create_visualization(embeddings, labels, output_dir)
    
    # Display results
    print(f"\n📊 SEMANTIC CLUSTERING RESULTS:")
    stats = cluster_analysis['statistics']
    print(f"   Total semantic clusters: {stats['total_clusters']}")
    print(f"   Tickets successfully clustered: {stats['total_tickets_clustered']:,} ({stats['clustering_rate']:.1f}%)")
    print(f"   Noise/unique tickets: {stats['noise_points']:,}")
    print(f"   Average cluster size: {stats['avg_cluster_size']:.1f} tickets")
    
    # Show top clusters
    clusters = cluster_analysis['clusters']
    semantic_clusters = {k: v for k, v in clusters.items() if k != 'noise'}
    
    if semantic_clusters:
        print(f"\n🎯 TOP SEMANTIC CLUSTERS:")
        sorted_clusters = sorted(semantic_clusters.items(), key=lambda x: x[1]['size'], reverse=True)
        
        for i, (cluster_name, cluster_info) in enumerate(sorted_clusters[:10], 1):
            print(f"   {i:2d}. {cluster_name}: {cluster_info['size']} tickets")
            print(f"       Categories: {list(cluster_info['categories'].keys())}")
            print(f"       Sample: '{cluster_info['sample_descriptions'][0]}'")
            if cluster_info['common_keywords']:
                top_keywords = list(cluster_info['common_keywords'].keys())[:5]
                print(f"       Keywords: {top_keywords}")
    
    # Save results
    print(f"\n💾 SAVING SEMANTIC ANALYSIS RESULTS")
    
    # Save complete analysis
    semantic_results_file = output_dir / 'semantic_analysis_results.json'
    with open(semantic_results_file, 'w') as f:
        json.dump({
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'methodology': {
                'model': analyzer.model_name,
                'clustering_algorithm': 'DBSCAN',
                'parameters': optimal_params
            },
            'cluster_analysis': {k: v for k, v in cluster_analysis.items() if k != 'cluster_analysis_df'},
            'summary': stats
        }, f, indent=2, default=str)
    print(f"   📄 Complete analysis saved to: {semantic_results_file}")
    
    # Save cluster assignments
    cluster_assignments_file = output_dir / 'semantic_cluster_assignments.csv'
    cluster_analysis['cluster_analysis_df'].to_csv(cluster_assignments_file, index=False)
    print(f"   📄 Cluster assignments saved to: {cluster_assignments_file}")
    
    print(f"\n✅ SEMANTIC ANALYSIS COMPLETE")
    print(f"   🎯 {stats['clustering_rate']:.1f}% of unclassified tickets grouped into semantic clusters")
    print(f"   🔍 {stats['total_clusters']} distinct problem patterns identified")
    print(f"   📁 Results saved to: {output_dir}")
    
    print(f"\n🔄 NEXT STEPS:")
    print(f"   1. Review semantic clusters to validate problem groupings")
    print(f"   2. Create core problem statements for each cluster")
    print(f"   3. Combine with hardcoded classifications for final taxonomy")
    print(f"   4. Calculate total automation potential")
    
    logger.info("Semantic analysis completed successfully!")

if __name__ == "__main__":
    main()