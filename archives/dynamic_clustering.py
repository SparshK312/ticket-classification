#!/usr/bin/env python3
"""
DYNAMIC CLUSTERING - Automatic Optimal Cluster Determination

This script implements multiple methods to automatically determine the optimal
number of clusters for agglomerative clustering without pre-setting k.

Methods implemented:
1. Calinski-Harabasz Index progression
2. Silhouette Analysis with sampling
3. Dendrogram Gap Analysis  
4. Distance Threshold method
5. Consensus clustering approach
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
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# Import our proven embedding pipeline
from semantic_analysis import SemanticAnalyzer

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

class DynamicClustering:
    """Dynamic clustering with automatic optimal cluster determination."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results = {}
        
    def load_embeddings_and_data(self, results_file: Path):
        """Load embeddings using proven pipeline from semantic_analysis.py."""
        self.logger.info("Loading embeddings using proven semantic_analysis pipeline...")
        
        analyzer = SemanticAnalyzer()
        df_unclassified = analyzer.load_unclassified_tickets(results_file)
        embeddings, texts = analyzer.encode_tickets(df_unclassified)
        
        self.logger.info(f"Loaded {len(embeddings)} embeddings with shape: {embeddings.shape}")
        return embeddings, df_unclassified, texts
    
    def calculate_linkage_matrix(self, embeddings: np.ndarray):
        """Calculate linkage matrix for hierarchical clustering."""
        self.logger.info("Calculating linkage matrix (Ward method)...")
        
        # Calculate pairwise distances
        distances = pdist(embeddings, metric='euclidean')
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(distances, method='ward')
        
        self.logger.info(f"Linkage matrix calculated with shape: {linkage_matrix.shape}")
        return linkage_matrix
    
    def method_1_calinski_harabasz_progression(self, embeddings: np.ndarray):
        """Method 1: Find optimal clusters using Calinski-Harabasz index progression."""
        self.logger.info("🔍 Method 1: Calinski-Harabasz Index Progression")
        
        # Test range of cluster numbers
        cluster_range = range(2, min(51, len(embeddings)//10))  # Reasonable upper bound
        ch_scores = []
        
        for n_clusters in cluster_range:
            try:
                clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='ward'
                )
                labels = clusterer.fit_predict(embeddings)
                
                # Calculate Calinski-Harabasz score
                ch_score = calinski_harabasz_score(embeddings, labels)
                ch_scores.append((n_clusters, ch_score))
                
                if n_clusters <= 20:  # Log first 20 for monitoring
                    self.logger.info(f"   n_clusters={n_clusters}: CH_score={ch_score:.2f}")
                    
            except Exception as e:
                self.logger.warning(f"Failed for n_clusters={n_clusters}: {str(e)}")
                ch_scores.append((n_clusters, 0))
        
        # Find optimal using elbow method on CH scores
        if len(ch_scores) < 3:
            return None
            
        scores = [score for _, score in ch_scores]
        optimal_idx = self._find_elbow_point(scores)
        optimal_clusters = ch_scores[optimal_idx][0]
        
        result = {
            'method': 'Calinski-Harabasz',
            'optimal_clusters': optimal_clusters,
            'score': ch_scores[optimal_idx][1],
            'all_scores': ch_scores,
            'confidence': self._calculate_elbow_confidence(scores, optimal_idx)
        }
        
        self.logger.info(f"   → Optimal clusters: {optimal_clusters} (score: {result['score']:.2f})")
        return result
    
    def method_2_silhouette_analysis(self, embeddings: np.ndarray, sample_size: int = 1000):
        """Method 2: Find optimal clusters using silhouette analysis with sampling."""
        self.logger.info("🔍 Method 2: Silhouette Analysis (with sampling)")
        
        # Sample for efficiency if dataset is large
        if len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sample_embeddings = embeddings[indices]
        else:
            sample_embeddings = embeddings
        
        cluster_range = range(2, min(31, len(sample_embeddings)//5))
        silhouette_scores = []
        
        for n_clusters in cluster_range:
            try:
                clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='ward'
                )
                labels = clusterer.fit_predict(sample_embeddings)
                
                # Calculate silhouette score
                sil_score = silhouette_score(sample_embeddings, labels)
                silhouette_scores.append((n_clusters, sil_score))
                
                if n_clusters <= 20:
                    self.logger.info(f"   n_clusters={n_clusters}: Silhouette={sil_score:.3f}")
                    
            except Exception as e:
                self.logger.warning(f"Silhouette failed for n_clusters={n_clusters}: {str(e)}")
                silhouette_scores.append((n_clusters, -1))
        
        # Find maximum silhouette score
        if not silhouette_scores:
            return None
            
        optimal_clusters, max_score = max(silhouette_scores, key=lambda x: x[1])
        
        result = {
            'method': 'Silhouette',
            'optimal_clusters': optimal_clusters,
            'score': max_score,
            'all_scores': silhouette_scores,
            'sample_size': len(sample_embeddings),
            'confidence': max_score  # Silhouette score is itself a confidence measure
        }
        
        self.logger.info(f"   → Optimal clusters: {optimal_clusters} (score: {max_score:.3f})")
        return result
    
    def method_3_dendrogram_gap_analysis(self, linkage_matrix: np.ndarray):
        """Method 3: Find optimal clusters using dendrogram gap analysis."""
        self.logger.info("🔍 Method 3: Dendrogram Gap Analysis")
        
        # Extract merge distances from linkage matrix
        merge_distances = linkage_matrix[:, 2]
        
        # Calculate gaps between consecutive merges
        gaps = np.diff(merge_distances)
        
        # Find largest gaps (indicating natural stopping points)
        # The number of clusters is n_samples - merge_index
        n_samples = len(linkage_matrix) + 1
        
        # Get indices of largest gaps
        gap_indices = np.argsort(gaps)[-10:]  # Top 10 gaps
        potential_clusters = [n_samples - idx - 1 for idx in gap_indices]
        
        # Filter to reasonable range (2-50 clusters)
        reasonable_clusters = [c for c in potential_clusters if 2 <= c <= 50]
        
        if not reasonable_clusters:
            return None
        
        # Select the cluster count with largest gap in reasonable range
        largest_gap_idx = None
        largest_gap = 0
        
        for cluster_count in reasonable_clusters:
            merge_idx = n_samples - cluster_count - 1
            if 0 <= merge_idx < len(gaps):
                gap = gaps[merge_idx]
                if gap > largest_gap:
                    largest_gap = gap
                    largest_gap_idx = merge_idx
        
        if largest_gap_idx is None:
            optimal_clusters = reasonable_clusters[0]
        else:
            optimal_clusters = n_samples - largest_gap_idx - 1
        
        result = {
            'method': 'Dendrogram_Gap',
            'optimal_clusters': optimal_clusters,
            'largest_gap': largest_gap,
            'potential_clusters': sorted(reasonable_clusters, reverse=True),
            'confidence': min(1.0, largest_gap / np.mean(gaps)) if np.mean(gaps) > 0 else 0.5
        }
        
        self.logger.info(f"   → Optimal clusters: {optimal_clusters} (gap: {largest_gap:.3f})")
        return result
    
    def method_4_distance_threshold(self, embeddings: np.ndarray, linkage_matrix: np.ndarray):
        """Method 4: Use distance threshold to determine clusters automatically."""
        self.logger.info("🔍 Method 4: Distance Threshold Method")
        
        # Extract merge distances
        merge_distances = linkage_matrix[:, 2]
        
        # Method A: Use standard deviation threshold
        mean_distance = np.mean(merge_distances)
        std_distance = np.std(merge_distances)
        threshold_std = mean_distance + 0.5 * std_distance
        
        # Method B: Use percentile threshold
        threshold_percentile = np.percentile(merge_distances, 75)
        
        # Method C: Use knee detection
        threshold_knee = self._find_knee_threshold(merge_distances)
        
        # Test all three thresholds
        thresholds = {
            'std_based': threshold_std,
            'percentile_75': threshold_percentile,
            'knee_detection': threshold_knee
        }
        
        results = {}
        for method, threshold in thresholds.items():
            try:
                # Create clustering with distance threshold
                clusterer = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=threshold,
                    linkage='ward'
                )
                labels = clusterer.fit_predict(embeddings)
                n_clusters = len(set(labels))
                
                results[method] = {
                    'threshold': threshold,
                    'n_clusters': n_clusters,
                    'labels': labels
                }
                
                self.logger.info(f"   {method}: threshold={threshold:.3f} → {n_clusters} clusters")
                
            except Exception as e:
                self.logger.warning(f"Distance threshold {method} failed: {str(e)}")
                results[method] = {'threshold': threshold, 'n_clusters': 1, 'labels': np.zeros(len(embeddings))}
        
        # Select best threshold (prefer reasonable cluster counts)
        best_method = None
        best_score = 0
        
        for method, result in results.items():
            n_clusters = result['n_clusters']
            # Score based on reasonable cluster count (prefer 5-30)
            if 5 <= n_clusters <= 30:
                score = 1.0
            elif 2 <= n_clusters <= 50:
                score = 0.7
            else:
                score = 0.3
            
            if score > best_score:
                best_score = score
                best_method = method
        
        if best_method is None:
            best_method = 'std_based'  # Fallback
        
        optimal_result = results[best_method]
        
        result = {
            'method': 'Distance_Threshold',
            'optimal_clusters': optimal_result['n_clusters'],
            'threshold': optimal_result['threshold'],
            'threshold_method': best_method,
            'all_results': results,
            'confidence': best_score,
            'labels': optimal_result['labels']
        }
        
        self.logger.info(f"   → Optimal: {best_method} with {optimal_result['n_clusters']} clusters")
        return result
    
    def method_5_consensus_clustering(self, all_results: list):
        """Method 5: Consensus approach combining all methods."""
        self.logger.info("🔍 Method 5: Consensus Clustering")
        
        if not all_results:
            return None
        
        # Extract cluster counts and confidence scores
        cluster_counts = []
        weighted_counts = []
        
        for result in all_results:
            if result is not None:
                n_clusters = result['optimal_clusters']
                confidence = result.get('confidence', 0.5)
                
                cluster_counts.append(n_clusters)
                weighted_counts.extend([n_clusters] * int(confidence * 10))  # Weight by confidence
        
        if not cluster_counts:
            return None
        
        # Calculate consensus statistics
        mean_clusters = np.mean(cluster_counts)
        median_clusters = np.median(cluster_counts)
        mode_clusters = Counter(weighted_counts).most_common(1)[0][0] if weighted_counts else median_clusters
        
        # Calculate agreement score
        cluster_counter = Counter(cluster_counts)
        max_agreement = max(cluster_counter.values())
        agreement_ratio = max_agreement / len(cluster_counts)
        
        # Select optimal based on weighted consensus
        if agreement_ratio >= 0.6:  # Strong agreement
            optimal_clusters = int(mode_clusters)
        elif abs(mean_clusters - median_clusters) <= 2:  # Mean and median close
            optimal_clusters = int(np.round(mean_clusters))
        else:  # Fall back to median
            optimal_clusters = int(median_clusters)
        
        result = {
            'method': 'Consensus',
            'optimal_clusters': optimal_clusters,
            'mean_clusters': mean_clusters,
            'median_clusters': median_clusters,
            'mode_clusters': mode_clusters,
            'agreement_ratio': agreement_ratio,
            'confidence': agreement_ratio,
            'individual_results': {r['method']: r['optimal_clusters'] for r in all_results if r is not None}
        }
        
        self.logger.info(f"   → Consensus optimal: {optimal_clusters} clusters (agreement: {agreement_ratio:.1%})")
        return result
    
    def _find_elbow_point(self, scores: list):
        """Find elbow point in score progression using second derivative."""
        if len(scores) < 3:
            return 0
        
        # Calculate second derivative
        first_derivative = np.diff(scores)
        second_derivative = np.diff(first_derivative)
        
        # Find maximum curvature (elbow point)
        elbow_idx = np.argmax(np.abs(second_derivative)) + 1
        
        # Ensure reasonable bounds
        elbow_idx = max(1, min(len(scores) - 2, elbow_idx))
        
        return elbow_idx
    
    def _calculate_elbow_confidence(self, scores: list, elbow_idx: int):
        """Calculate confidence in elbow detection."""
        if len(scores) < 3 or elbow_idx <= 0 or elbow_idx >= len(scores) - 1:
            return 0.5
        
        # Calculate curvature at elbow point
        before = scores[elbow_idx - 1]
        at_elbow = scores[elbow_idx]
        after = scores[elbow_idx + 1]
        
        # Measure how pronounced the elbow is
        curvature = abs(2 * at_elbow - before - after)
        max_score = max(scores)
        
        # Normalize confidence
        confidence = min(1.0, curvature / (max_score * 0.1))
        return confidence
    
    def _find_knee_threshold(self, distances: np.ndarray):
        """Find knee point in distance progression."""
        if len(distances) < 3:
            return np.mean(distances)
        
        # Sort distances
        sorted_distances = np.sort(distances)
        
        # Calculate knee using second derivative
        x = np.arange(len(sorted_distances))
        first_derivative = np.diff(sorted_distances)
        
        if len(first_derivative) < 2:
            return np.percentile(sorted_distances, 75)
        
        second_derivative = np.diff(first_derivative)
        knee_idx = np.argmax(second_derivative)
        
        # Return distance at knee point
        knee_threshold = sorted_distances[min(knee_idx + 1, len(sorted_distances) - 1)]
        
        return knee_threshold
    
    def run_final_clustering(self, embeddings: np.ndarray, df: pd.DataFrame, texts: list, 
                           optimal_clusters: int, method_name: str):
        """Run final clustering with optimal number of clusters."""
        self.logger.info(f"🎯 Running final clustering with {optimal_clusters} clusters ({method_name})")
        
        try:
            # Perform final clustering
            clusterer = AgglomerativeClustering(
                n_clusters=optimal_clusters,
                linkage='ward'
            )
            labels = clusterer.fit_predict(embeddings)
            
            # Calculate final metrics
            metrics = {
                'davies_bouldin': davies_bouldin_score(embeddings, labels),
                'calinski_harabasz': calinski_harabasz_score(embeddings, labels)
            }
            
            # Add silhouette if dataset not too large
            if len(embeddings) <= 2000:
                metrics['silhouette'] = silhouette_score(embeddings, labels)
            
            # Analyze clusters
            cluster_analysis = self._analyze_final_clusters(labels, df, texts)
            
            result = {
                'method_used': method_name,
                'optimal_clusters': optimal_clusters,
                'labels': labels,
                'metrics': metrics,
                'cluster_analysis': cluster_analysis,
                'n_noise': 0,  # Agglomerative doesn't produce noise
                'clustered_ratio': 1.0
            }
            
            self.logger.info(f"   Final clustering complete:")
            self.logger.info(f"   - Clusters: {optimal_clusters}")
            self.logger.info(f"   - Davies-Bouldin: {metrics['davies_bouldin']:.3f}")
            self.logger.info(f"   - Calinski-Harabasz: {metrics['calinski_harabasz']:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Final clustering failed: {str(e)}")
            return None
    
    def _analyze_final_clusters(self, labels: np.ndarray, df: pd.DataFrame, texts: list):
        """Analyze final cluster composition."""
        unique_labels = set(labels)
        clusters = {}
        
        for cluster_id in sorted(unique_labels):
            cluster_mask = labels == cluster_id
            cluster_tickets = df[cluster_mask]
            cluster_texts = [texts[i] for i in range(len(texts)) if cluster_mask[i]]
            
            if len(cluster_tickets) == 0:
                continue
            
            # Sample tickets for review
            sample_indices = [0, len(cluster_tickets)//2, len(cluster_tickets)-1]
            sample_indices = list(set(sample_indices))
            
            cluster_info = {
                'size': len(cluster_tickets),
                'sample_descriptions': [
                    cluster_tickets.iloc[i]['Short description'] 
                    for i in sample_indices[:3]
                ],
                'category_distribution': cluster_tickets['category'].value_counts().head(5).to_dict(),
                'percentage': len(cluster_tickets) / len(df) * 100
            }
            
            clusters[f"cluster_{cluster_id}"] = cluster_info
        
        return clusters
    
    def create_comparison_visualization(self, all_results: list, output_dir: Path):
        """Create visualization comparing all methods."""
        self.logger.info("📊 Creating method comparison visualization...")
        
        # Extract data for plotting
        methods = []
        cluster_counts = []
        confidences = []
        
        for result in all_results:
            if result is not None:
                methods.append(result['method'])
                cluster_counts.append(result['optimal_clusters'])
                confidences.append(result.get('confidence', 0.5))
        
        if not methods:
            return
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Cluster counts by method
        bars1 = ax1.bar(methods, cluster_counts, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Method')
        ax1.set_ylabel('Optimal Clusters')
        ax1.set_title('Optimal Cluster Count by Method')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars1, cluster_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom')
        
        # Plot 2: Confidence scores
        bars2 = ax2.bar(methods, confidences, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Method')
        ax2.set_ylabel('Confidence Score')
        ax2.set_title('Method Confidence Scores')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, conf in zip(bars2, confidences):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{conf:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_dir / 'dynamic_clustering_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Comparison visualization saved to: {plot_file}")

def main():
    """Main dynamic clustering pipeline."""
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
    print("DYNAMIC CLUSTERING - AUTOMATIC OPTIMAL CLUSTER DETERMINATION")
    print("="*80)
    
    # Initialize dynamic clustering
    dynamic_clusterer = DynamicClustering()
    
    # Step 1: Load embeddings
    print(f"\n🔄 STEP 1: Loading embeddings and data")
    embeddings, df, texts = dynamic_clusterer.load_embeddings_and_data(results_file)
    
    # Step 2: Calculate linkage matrix
    print(f"\n🔄 STEP 2: Calculating linkage matrix")
    linkage_matrix = dynamic_clusterer.calculate_linkage_matrix(embeddings)
    
    # Step 3: Run all optimization methods
    print(f"\n🔄 STEP 3: Running optimization methods")
    
    all_results = []
    
    # Method 1: Calinski-Harabasz progression
    result1 = dynamic_clusterer.method_1_calinski_harabasz_progression(embeddings)
    if result1: all_results.append(result1)
    
    # Method 2: Silhouette analysis
    result2 = dynamic_clusterer.method_2_silhouette_analysis(embeddings)
    if result2: all_results.append(result2)
    
    # Method 3: Dendrogram gap analysis
    result3 = dynamic_clusterer.method_3_dendrogram_gap_analysis(linkage_matrix)
    if result3: all_results.append(result3)
    
    # Method 4: Distance threshold
    result4 = dynamic_clusterer.method_4_distance_threshold(embeddings, linkage_matrix)
    if result4: all_results.append(result4)
    
    # Method 5: Consensus
    consensus_result = dynamic_clusterer.method_5_consensus_clustering(all_results)
    if consensus_result: all_results.append(consensus_result)
    
    # Step 4: Select best method and run final clustering
    print(f"\n🔄 STEP 4: Selecting optimal approach and running final clustering")
    
    if not all_results:
        print("❌ No methods produced valid results")
        return
    
    # Select consensus if available, otherwise highest confidence
    if consensus_result:
        best_result = consensus_result
    else:
        best_result = max(all_results, key=lambda x: x.get('confidence', 0))
    
    optimal_clusters = best_result['optimal_clusters']
    method_name = best_result['method']
    
    # Run final clustering
    final_result = dynamic_clusterer.run_final_clustering(
        embeddings, df, texts, optimal_clusters, method_name
    )
    
    # Step 5: Create visualizations and save results
    print(f"\n🔄 STEP 5: Creating visualizations and saving results")
    dynamic_clusterer.create_comparison_visualization(all_results, output_dir)
    
    # Display results
    print(f"\n📊 DYNAMIC CLUSTERING RESULTS:")
    print(f"   Methods tested: {len(all_results)}")
    
    print(f"\n🎯 METHOD COMPARISON:")
    for result in all_results:
        if result:
            print(f"   {result['method']}: {result['optimal_clusters']} clusters (confidence: {result.get('confidence', 0):.2f})")
    
    print(f"\n🏆 SELECTED APPROACH:")
    print(f"   Method: {method_name}")
    print(f"   Optimal clusters: {optimal_clusters}")
    print(f"   Confidence: {best_result.get('confidence', 0):.2f}")
    
    if final_result:
        print(f"\n📈 FINAL CLUSTERING METRICS:")
        metrics = final_result['metrics']
        print(f"   Davies-Bouldin: {metrics['davies_bouldin']:.3f}")
        print(f"   Calinski-Harabasz: {metrics['calinski_harabasz']:.2f}")
        if 'silhouette' in metrics:
            print(f"   Silhouette: {metrics['silhouette']:.3f}")
        
        # Show top clusters
        print(f"\n🎯 TOP CLUSTERS:")
        cluster_items = list(final_result['cluster_analysis'].items())
        cluster_items.sort(key=lambda x: x[1]['size'], reverse=True)
        
        for i, (cluster_name, cluster_info) in enumerate(cluster_items[:5]):
            print(f"   {i+1}. {cluster_name}: {cluster_info['size']} tickets ({cluster_info['percentage']:.1f}%)")
            print(f"      Sample: '{cluster_info['sample_descriptions'][0]}'")
            print(f"      Top categories: {list(cluster_info['category_distribution'].keys())[:3]}")
    
    # Save comprehensive results
    results_summary = {
        'experiment_timestamp': datetime.now().isoformat(),
        'method_results': [r for r in all_results if r is not None],
        'selected_method': method_name,
        'optimal_clusters': optimal_clusters,
        'final_clustering': final_result
    }
    
    results_file = output_dir / 'dynamic_clustering_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\n💾 RESULTS SAVED:")
    print(f"   📄 Comprehensive results: {results_file}")
    print(f"   📊 Method comparison plot: {output_dir}/dynamic_clustering_comparison.png")
    
    print(f"\n🔄 NEXT STEPS:")
    print(f"   1. Review the selected method and cluster results")
    print(f"   2. Validate sample tickets from each cluster")
    print(f"   3. If satisfied, integrate into semantic_analysis.py")
    print(f"   4. Consider creating hierarchical cluster explorer for stakeholders")
    
    logger.info("Dynamic clustering analysis completed successfully!")

if __name__ == "__main__":
    main()