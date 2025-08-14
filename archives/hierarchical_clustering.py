#!/usr/bin/env python3
"""
HIERARCHICAL CLUSTERING - Two-Level Clustering System

This script implements a hierarchical clustering approach:
1. Level 1: 10 main clusters for routing/dashboards/high-level analysis
2. Level 2: HDBSCAN sub-clustering on large clusters (>10% of tickets)
3. Edge-case handling for tickets that don't fit into sub-clusters

Strategy:
- Use proven 10-cluster Agglomerative solution as base
- Apply HDBSCAN to clusters with >320 tickets (10% threshold)
- Min cluster size 15-20 tickets for business relevance
- Label remaining points as "edge-cases" within their parent cluster
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
import logging
from datetime import datetime
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_distances
import hdbscan
import warnings
warnings.filterwarnings('ignore')

# Import our proven embedding pipeline
from semantic_analysis import SemanticAnalyzer

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

class HierarchicalClustering:
    """Hierarchical clustering with two-level structure."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results = {}
        self.threshold_percentage = 10.0  # 10% threshold for sub-clustering
        self.min_subcluster_size = 18  # Minimum 18 tickets per sub-cluster
        
    def load_embeddings_and_data(self, results_file: Path):
        """Load embeddings using proven pipeline from semantic_analysis.py."""
        self.logger.info("Loading embeddings using proven semantic_analysis pipeline...")
        
        analyzer = SemanticAnalyzer()
        df_unclassified = analyzer.load_unclassified_tickets(results_file)
        embeddings, texts = analyzer.encode_tickets(df_unclassified)
        
        self.logger.info(f"Loaded {len(embeddings)} embeddings with shape: {embeddings.shape}")
        return embeddings, df_unclassified, texts
    
    def apply_top_level_clustering(self, embeddings: np.ndarray):
        """Apply the proven 10-cluster Agglomerative solution."""
        self.logger.info("🎯 Applying top-level 10-cluster Agglomerative clustering...")
        
        # Use proven parameters from cluster_experiments.py
        clusterer = AgglomerativeClustering(
            n_clusters=10,
            linkage='ward'
        )
        
        top_level_labels = clusterer.fit_predict(embeddings)
        
        # Calculate metrics
        metrics = {
            'davies_bouldin': davies_bouldin_score(embeddings, top_level_labels),
            'calinski_harabasz': calinski_harabasz_score(embeddings, top_level_labels)
        }
        
        self.logger.info(f"   Top-level clustering complete:")
        self.logger.info(f"   - Clusters: 10")
        self.logger.info(f"   - Davies-Bouldin: {metrics['davies_bouldin']:.3f}")
        self.logger.info(f"   - Calinski-Harabasz: {metrics['calinski_harabasz']:.2f}")
        
        return top_level_labels, metrics
    
    def identify_large_clusters(self, labels: np.ndarray, df: pd.DataFrame):
        """Identify clusters that exceed the threshold for sub-clustering."""
        total_tickets = len(labels)
        threshold_size = int(total_tickets * (self.threshold_percentage / 100))
        
        self.logger.info(f"🔍 Identifying large clusters (>{self.threshold_percentage}% = >{threshold_size} tickets)")
        
        cluster_sizes = {}
        large_clusters = []
        
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            cluster_sizes[cluster_id] = cluster_size
            
            if cluster_size > threshold_size:
                large_clusters.append(cluster_id)
                self.logger.info(f"   Cluster {cluster_id}: {cluster_size} tickets ({cluster_size/total_tickets*100:.1f}%) - LARGE")
            else:
                self.logger.info(f"   Cluster {cluster_id}: {cluster_size} tickets ({cluster_size/total_tickets*100:.1f}%)")
        
        self.logger.info(f"   → {len(large_clusters)} clusters selected for sub-clustering: {large_clusters}")
        
        return large_clusters, cluster_sizes
    
    def apply_hdbscan_subclustering(self, embeddings: np.ndarray, cluster_mask: np.ndarray, 
                                  cluster_id: int, cluster_texts: list):
        """Apply HDBSCAN sub-clustering to a specific cluster."""
        cluster_embeddings = embeddings[cluster_mask]
        
        self.logger.info(f"   🔬 Sub-clustering cluster {cluster_id} ({len(cluster_embeddings)} tickets)")
        
        # Calculate distance matrix for HDBSCAN - ensure double dtype
        distance_matrix = cosine_distances(cluster_embeddings).astype(np.float64)
        
        # Apply HDBSCAN with multiple parameter combinations
        best_result = None
        best_score = -1
        
        # Test different min_cluster_size values
        min_sizes = [15, 18, 20, 25]
        
        for min_size in min_sizes:
            if min_size >= len(cluster_embeddings) // 3:  # Skip if too large
                continue
                
            try:
                # HDBSCAN with cosine distance
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_size,
                    metric='precomputed',
                    cluster_selection_epsilon=0.1
                )
                
                sub_labels = clusterer.fit_predict(distance_matrix)
                
                # Count clusters and noise
                n_clusters = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
                n_noise = list(sub_labels).count(-1)
                noise_ratio = n_noise / len(sub_labels)
                
                # Score based on: cluster count, low noise, reasonable sizes
                if n_clusters > 0:
                    clustered_points = len(sub_labels) - n_noise
                    avg_cluster_size = clustered_points / n_clusters if n_clusters > 0 else 0
                    
                    # Prefer solutions with 2-6 sub-clusters, low noise, good sizes
                    score = 0
                    if 2 <= n_clusters <= 6:
                        score += 1.0
                    elif 1 <= n_clusters <= 8:
                        score += 0.7
                    else:
                        score += 0.3
                    
                    # Reward low noise (but accept some noise as edge cases)
                    if noise_ratio <= 0.3:
                        score += 1.0
                    elif noise_ratio <= 0.5:
                        score += 0.7
                    else:
                        score += 0.3
                    
                    # Reward reasonable cluster sizes
                    if 20 <= avg_cluster_size <= 100:
                        score += 1.0
                    elif 15 <= avg_cluster_size <= 150:
                        score += 0.8
                    else:
                        score += 0.5
                    
                    if score > best_score:
                        best_score = score
                        best_result = {
                            'labels': sub_labels,
                            'n_clusters': n_clusters,
                            'n_noise': n_noise,
                            'noise_ratio': noise_ratio,
                            'min_cluster_size': min_size,
                            'avg_cluster_size': avg_cluster_size,
                            'score': score
                        }
                        
                        self.logger.info(f"      min_size={min_size}: {n_clusters} clusters, {n_noise} noise ({noise_ratio:.1%}), score={score:.2f}")
                
            except Exception as e:
                self.logger.warning(f"      HDBSCAN failed for min_size={min_size}: {str(e)}")
        
        if best_result is None:
            # Fallback: no sub-clustering, treat all as one group
            self.logger.info(f"      → No valid sub-clustering found, keeping as single cluster")
            return np.full(len(cluster_embeddings), 0, dtype=int), {
                'n_clusters': 1,
                'n_noise': 0,
                'noise_ratio': 0.0,
                'method': 'fallback_single'
            }
        
        self.logger.info(f"      → Best: {best_result['n_clusters']} sub-clusters, {best_result['n_noise']} edge-cases")
        
        return best_result['labels'], best_result
    
    def build_hierarchical_structure(self, embeddings: np.ndarray, df: pd.DataFrame, 
                                   texts: list, top_level_labels: np.ndarray):
        """Build the complete hierarchical clustering structure."""
        self.logger.info("🏗️  Building hierarchical clustering structure...")
        
        # Identify large clusters for sub-clustering
        large_clusters, cluster_sizes = self.identify_large_clusters(top_level_labels, df)
        
        # Initialize hierarchical labels
        # Format: "L1_X" for top-level only, "L1_X_L2_Y" for sub-clustered
        hierarchical_labels = []
        sub_clustering_results = {}
        
        for i, top_cluster in enumerate(top_level_labels):
            if top_cluster in large_clusters:
                # This cluster needs sub-clustering - will be updated later
                hierarchical_labels.append(f"L1_{top_cluster}_TEMP")
            else:
                # Small cluster - stays at top level
                hierarchical_labels.append(f"L1_{top_cluster}")
        
        # Apply sub-clustering to large clusters
        for cluster_id in large_clusters:
            cluster_mask = top_level_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_texts_subset = [texts[i] for i in cluster_indices]
            
            # Apply HDBSCAN sub-clustering
            sub_labels, sub_results = self.apply_hdbscan_subclustering(
                embeddings, cluster_mask, cluster_id, cluster_texts_subset
            )
            
            # Update hierarchical labels for this cluster
            for idx, sub_label in enumerate(sub_labels):
                global_idx = cluster_indices[idx]
                if sub_label == -1:
                    # Edge case within this cluster
                    hierarchical_labels[global_idx] = f"L1_{cluster_id}_edge-case"
                else:
                    # Sub-cluster - ensure integer conversion
                    hierarchical_labels[global_idx] = f"L1_{cluster_id}_L2_{int(sub_label)}"
            
            # Store sub-clustering results - convert numpy int64 to int
            sub_clustering_results[int(cluster_id)] = sub_results
        
        self.logger.info("   Hierarchical structure complete!")
        
        return hierarchical_labels, sub_clustering_results
    
    def analyze_hierarchical_clusters(self, df: pd.DataFrame, texts: list, 
                                    hierarchical_labels: list, sub_clustering_results: dict):
        """Analyze the hierarchical clustering results."""
        self.logger.info("📊 Analyzing hierarchical clustering results...")
        
        df_analysis = df.copy()
        df_analysis['hierarchical_label'] = hierarchical_labels
        df_analysis['prepared_text'] = texts
        
        # Parse hierarchical structure
        hierarchy = defaultdict(lambda: defaultdict(list))
        label_counts = Counter(hierarchical_labels)
        
        for label in hierarchical_labels:
            parts = label.split('_')
            l1_cluster = int(parts[1])  # L1_X
            
            if len(parts) == 2:  # Top-level only: L1_X
                hierarchy[l1_cluster]['top_level_only'].append(label)
            elif parts[2] == 'edge-case':  # Edge case: L1_X_edge-case
                hierarchy[l1_cluster]['edge_cases'].append(label)
            else:  # Sub-cluster: L1_X_L2_Y
                l2_cluster = int(parts[3])
                hierarchy[l1_cluster]['sub_clusters'].append(label)
        
        # Analyze each top-level cluster
        cluster_analysis = {}
        
        for l1_cluster in sorted(hierarchy.keys()):
            l1_mask = df_analysis['hierarchical_label'].str.startswith(f'L1_{l1_cluster}')
            l1_tickets = df_analysis[l1_mask]
            
            if len(l1_tickets) == 0:
                continue
            
            cluster_info = {
                'l1_cluster_id': l1_cluster,
                'total_size': len(l1_tickets),
                'percentage': len(l1_tickets) / len(df_analysis) * 100,
                'categories': l1_tickets['category'].value_counts().to_dict(),
                'sample_descriptions': l1_tickets['Short description'].head(5).tolist(),
                'sub_structure': {}
            }
            
            # Analyze sub-structure
            unique_labels = l1_tickets['hierarchical_label'].unique()
            
            for label in sorted(unique_labels):
                label_mask = df_analysis['hierarchical_label'] == label
                label_tickets = df_analysis[label_mask]
                
                if len(label_tickets) == 0:
                    continue
                
                sub_info = {
                    'size': len(label_tickets),
                    'percentage_of_l1': len(label_tickets) / len(l1_tickets) * 100,
                    'sample_descriptions': label_tickets['Short description'].head(3).tolist(),
                    'top_categories': label_tickets['category'].value_counts().head(3).to_dict()
                }
                
                cluster_info['sub_structure'][label] = sub_info
            
            cluster_analysis[f"L1_{l1_cluster}"] = cluster_info
        
        # Overall statistics
        edge_cases = [label for label in hierarchical_labels if 'edge-case' in label]
        sub_clusters = [label for label in hierarchical_labels if 'L2_' in label and 'edge-case' not in label]
        top_level_only = [label for label in hierarchical_labels if '_L2_' not in label and 'edge-case' not in label]
        
        statistics = {
            'total_tickets': len(hierarchical_labels),
            'l1_clusters': len(set([label.split('_')[1] for label in hierarchical_labels])),
            'total_sub_clusters': len(set(sub_clusters)),
            'edge_cases': len(edge_cases),
            'edge_case_percentage': len(edge_cases) / len(hierarchical_labels) * 100,
            'top_level_only': len(top_level_only),
            'sub_clustered': len(sub_clusters),
            'sub_clustering_results': sub_clustering_results
        }
        
        return {
            'cluster_analysis': cluster_analysis,
            'statistics': statistics,
            'hierarchy_structure': dict(hierarchy),
            'analysis_df': df_analysis[['ticket_index', 'category', 'Short description', 'hierarchical_label']]
        }
    
    def create_hierarchical_visualization(self, analysis: dict, output_dir: Path):
        """Create visualization of the hierarchical clustering."""
        self.logger.info("📊 Creating hierarchical clustering visualization...")
        
        # Create summary visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Top-level cluster sizes
        cluster_analysis = analysis['cluster_analysis']
        l1_clusters = []
        l1_sizes = []
        
        for cluster_name, info in cluster_analysis.items():
            l1_clusters.append(cluster_name)
            l1_sizes.append(info['total_size'])
        
        bars1 = ax1.bar(l1_clusters, l1_sizes, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Top-Level Clusters')
        ax1.set_ylabel('Number of Tickets')
        ax1.set_title('Top-Level Cluster Distribution')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, size in zip(bars1, l1_sizes):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{size}', ha='center', va='bottom')
        
        # Plot 2: Sub-clustering breakdown
        stats = analysis['statistics']
        categories = ['Top-level Only', 'Sub-clustered', 'Edge Cases']
        counts = [stats['top_level_only'], stats['sub_clustered'], stats['edge_cases']]
        colors = ['lightgreen', 'lightblue', 'lightcoral']
        
        ax2.pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Hierarchical Structure Breakdown')
        
        # Plot 3: Sub-cluster distribution for large clusters
        large_cluster_data = []
        for cluster_name, info in cluster_analysis.items():
            if len(info['sub_structure']) > 1:  # Has sub-structure
                for sub_label, sub_info in info['sub_structure'].items():
                    if 'L2_' in sub_label:
                        large_cluster_data.append({
                            'parent': cluster_name,
                            'sub_cluster': sub_label,
                            'size': sub_info['size']
                        })
        
        if large_cluster_data:
            sub_df = pd.DataFrame(large_cluster_data)
            parent_clusters = sub_df['parent'].unique()
            
            x_pos = 0
            colors_sub = plt.cm.Set3(np.linspace(0, 1, len(parent_clusters)))
            
            for i, parent in enumerate(parent_clusters):
                parent_data = sub_df[sub_df['parent'] == parent]
                bars = ax3.bar(range(x_pos, x_pos + len(parent_data)), 
                              parent_data['size'], 
                              color=colors_sub[i], 
                              alpha=0.7,
                              label=parent)
                x_pos += len(parent_data) + 1
            
            ax3.set_xlabel('Sub-clusters')
            ax3.set_ylabel('Number of Tickets')
            ax3.set_title('Sub-cluster Sizes')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No sub-clusters created', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Sub-cluster Sizes')
        
        # Plot 4: Edge case distribution
        edge_case_data = []
        for cluster_name, info in cluster_analysis.items():
            edge_count = sum(1 for label in info['sub_structure'].keys() if 'edge-case' in label)
            if edge_count > 0:
                edge_case_data.append({
                    'cluster': cluster_name,
                    'edge_cases': sum(sub_info['size'] for label, sub_info in info['sub_structure'].items() if 'edge-case' in label)
                })
        
        if edge_case_data:
            edge_df = pd.DataFrame(edge_case_data)
            bars4 = ax4.bar(edge_df['cluster'], edge_df['edge_cases'], 
                           color='lightcoral', alpha=0.7)
            ax4.set_xlabel('Top-Level Clusters')
            ax4.set_ylabel('Edge Cases')
            ax4.set_title('Edge Cases by Cluster')
            ax4.tick_params(axis='x', rotation=45)
            
            for bar, count in zip(bars4, edge_df['edge_cases']):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{count}', ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, 'No edge cases found', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Edge Cases by Cluster')
        
        plt.tight_layout()
        
        # Save visualization
        viz_file = output_dir / 'hierarchical_clustering_analysis.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Hierarchical visualization saved to: {viz_file}")

def main():
    """Main hierarchical clustering pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # File paths
    results_file = Path('outputs/improved_classification_results.json')
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    if not results_file.exists():
        print(f"❌ Error: Classification results file not found at {results_file}")
        return
    
    print("="*80)
    print("HIERARCHICAL CLUSTERING - TWO-LEVEL SYSTEM")
    print("="*80)
    print("Strategy:")
    print("  Level 1: 10 main clusters for routing/dashboards")
    print("  Level 2: HDBSCAN sub-clustering on large clusters (>10%)")
    print("  Edge-cases: Tickets that don't fit sub-clusters")
    print("="*80)
    
    # Initialize hierarchical clustering
    hierarchical_clusterer = HierarchicalClustering()
    
    # Step 1: Load embeddings
    print(f"\n🔄 STEP 1: Loading embeddings and data")
    embeddings, df, texts = hierarchical_clusterer.load_embeddings_and_data(results_file)
    
    # Step 2: Apply top-level clustering (10 clusters)
    print(f"\n🔄 STEP 2: Applying top-level clustering (10 clusters)")
    top_level_labels, top_level_metrics = hierarchical_clusterer.apply_top_level_clustering(embeddings)
    
    # Step 3: Build hierarchical structure
    print(f"\n🔄 STEP 3: Building hierarchical structure")
    hierarchical_labels, sub_clustering_results = hierarchical_clusterer.build_hierarchical_structure(
        embeddings, df, texts, top_level_labels
    )
    
    # Step 4: Analyze hierarchical results
    print(f"\n🔄 STEP 4: Analyzing hierarchical clustering")
    analysis = hierarchical_clusterer.analyze_hierarchical_clusters(
        df, texts, hierarchical_labels, sub_clustering_results
    )
    
    # Step 5: Create visualizations
    print(f"\n🔄 STEP 5: Creating visualizations")
    hierarchical_clusterer.create_hierarchical_visualization(analysis, output_dir)
    
    # Display results
    print(f"\n📊 HIERARCHICAL CLUSTERING RESULTS:")
    stats = analysis['statistics']
    print(f"   Top-level clusters: {stats['l1_clusters']}")
    print(f"   Total sub-clusters: {stats['total_sub_clusters']}")
    print(f"   Edge cases: {stats['edge_cases']} ({stats['edge_case_percentage']:.1f}%)")
    print(f"   Tickets with sub-clustering: {stats['sub_clustered']:,}")
    print(f"   Tickets staying top-level: {stats['top_level_only']:,}")
    
    # Show top-level cluster breakdown
    print(f"\n🎯 TOP-LEVEL CLUSTER BREAKDOWN:")
    cluster_analysis = analysis['cluster_analysis']
    sorted_clusters = sorted(cluster_analysis.items(), 
                           key=lambda x: x[1]['total_size'], reverse=True)
    
    for cluster_name, info in sorted_clusters:
        print(f"   {cluster_name}: {info['total_size']} tickets ({info['percentage']:.1f}%)")
        print(f"      Categories: {list(info['categories'].keys())[:3]}")
        print(f"      Sample: '{info['sample_descriptions'][0]}'")
        
        # Show sub-structure
        sub_structure = info['sub_structure']
        if len(sub_structure) > 1:
            print(f"      Sub-structure:")
            for sub_label, sub_info in sorted(sub_structure.items(), 
                                            key=lambda x: x[1]['size'], reverse=True):
                if 'edge-case' in sub_label:
                    print(f"        └─ Edge cases: {sub_info['size']} tickets ({sub_info['percentage_of_l1']:.1f}%)")
                elif 'L2_' in sub_label:
                    sub_id = sub_label.split('_')[-1]
                    print(f"        └─ Sub-cluster {sub_id}: {sub_info['size']} tickets ({sub_info['percentage_of_l1']:.1f}%)")
                    print(f"           Sample: '{sub_info['sample_descriptions'][0]}'")
        print()
    
    # Save comprehensive results
    results_summary = {
        'experiment_timestamp': datetime.now().isoformat(),
        'methodology': {
            'level_1': 'Agglomerative clustering with 10 clusters',
            'level_2': 'HDBSCAN sub-clustering on clusters >10%',
            'threshold_percentage': hierarchical_clusterer.threshold_percentage,
            'min_subcluster_size': hierarchical_clusterer.min_subcluster_size
        },
        'top_level_metrics': top_level_metrics,
        'analysis': {k: v for k, v in analysis.items() if k != 'analysis_df'},
        'summary_statistics': stats
    }
    
    results_file = output_dir / 'hierarchical_clustering_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, cls=NumpyEncoder)
    
    # Save cluster assignments
    assignments_file = output_dir / 'hierarchical_cluster_assignments.csv'
    analysis['analysis_df'].to_csv(assignments_file, index=False)
    
    print(f"\n💾 RESULTS SAVED:")
    print(f"   📄 Complete analysis: {results_file}")
    print(f"   📄 Cluster assignments: {assignments_file}")
    print(f"   📊 Visualization: {output_dir}/hierarchical_clustering_analysis.png")
    
    print(f"\n🎯 BUSINESS IMPACT:")
    print(f"   🎛️  Use 10 top-level clusters for dashboards and routing")
    print(f"   🔍 Drill down into {stats['total_sub_clusters']} sub-clusters for detailed analysis")
    print(f"   ⚠️  Monitor {stats['edge_cases']} edge cases for new pattern discovery")
    print(f"   📈 {100 - stats['edge_case_percentage']:.1f}% of tickets successfully clustered")
    
    print(f"\n🔄 NEXT STEPS:")
    print(f"   1. Review hierarchical structure and validate business relevance")
    print(f"   2. Create routing rules based on top-level clusters")
    print(f"   3. Build sub-cluster analysis for detailed problem identification")
    print(f"   4. Monitor edge cases for emerging ticket patterns")
    print(f"   5. Integrate into semantic_analysis.py for production use")
    
    logger.info("Hierarchical clustering analysis completed successfully!")

if __name__ == "__main__":
    main()