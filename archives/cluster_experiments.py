#!/usr/bin/env python3
"""
CLUSTER EXPERIMENTS - Systematic Clustering Method Comparison

This script imports the proven embedding pipeline from semantic_analysis.py
and runs comprehensive experiments to find the optimal clustering approach.

Strategy:
1. Fork, don't overwrite - keeps existing pipeline intact
2. HDBSCAN first - handles variable densities automatically  
3. Narrow DBSCAN grid search - data-driven parameter selection
4. Quantitative & SME validation - metrics + human review
5. Pick winner & integrate - adopt best performer
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

# Import clustering algorithms
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import cosine_distances
import hdbscan

# Import our proven embedding pipeline
from semantic_analysis import SemanticAnalyzer

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

class ClusteringExperiments:
    """Comprehensive clustering experiments using multiple algorithms."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results = []
        
    def load_embeddings_and_data(self, results_file: Path):
        """Load embeddings using proven pipeline from semantic_analysis.py."""
        self.logger.info("Loading embeddings using proven semantic_analysis pipeline...")
        
        # Use existing SemanticAnalyzer class
        analyzer = SemanticAnalyzer()
        
        # Load unclassified tickets
        df_unclassified = analyzer.load_unclassified_tickets(results_file)
        
        # Generate embeddings using existing method
        embeddings, texts = analyzer.encode_tickets(df_unclassified)
        
        self.logger.info(f"Loaded {len(embeddings)} embeddings with shape: {embeddings.shape}")
        
        return embeddings, df_unclassified, texts
    
    def run_hdbscan_experiment(self, embeddings: np.ndarray, df: pd.DataFrame, texts: list) -> dict:
        """Run HDBSCAN clustering experiment."""
        self.logger.info("🔍 Running HDBSCAN experiment...")
        
        # Test multiple min_cluster_size values
        min_cluster_sizes = [5, 10, 15, 20, 25, 30]
        hdbscan_results = []
        
        for min_cluster_size in min_cluster_sizes:
            try:
                # Run HDBSCAN
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    metric='euclidean',  # Works well with normalized embeddings
                    cluster_selection_method='eom'  # Excess of Mass
                )
                
                labels = clusterer.fit_predict(embeddings)
                
                # Calculate metrics
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                noise_ratio = n_noise / len(labels)
                
                # Clustering metrics (excluding noise points)
                non_noise_mask = labels != -1
                metrics = {}
                
                if n_clusters > 1 and np.sum(non_noise_mask) > 0:
                    try:
                        metrics['davies_bouldin'] = davies_bouldin_score(
                            embeddings[non_noise_mask], 
                            labels[non_noise_mask]
                        )
                    except:
                        metrics['davies_bouldin'] = np.inf
                    
                    # Skip silhouette for large datasets (too slow)
                    if len(embeddings) < 2000:
                        try:
                            metrics['silhouette'] = silhouette_score(
                                embeddings[non_noise_mask], 
                                labels[non_noise_mask]
                            )
                        except:
                            metrics['silhouette'] = -1
                    else:
                        metrics['silhouette'] = None
                else:
                    metrics['davies_bouldin'] = np.inf
                    metrics['silhouette'] = -1
                
                # Analyze clusters
                cluster_analysis = self._analyze_clusters(labels, df, texts)
                
                result = {
                    'algorithm': 'HDBSCAN',
                    'parameters': {'min_cluster_size': min_cluster_size},
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'noise_ratio': noise_ratio,
                    'clustered_ratio': (len(labels) - n_noise) / len(labels),
                    'metrics': metrics,
                    'cluster_analysis': cluster_analysis,
                    'labels': labels.copy()
                }
                
                hdbscan_results.append(result)
                
                self.logger.info(f"   min_cluster_size={min_cluster_size}: {n_clusters} clusters, {noise_ratio:.1%} noise")
                
            except Exception as e:
                self.logger.error(f"HDBSCAN failed with min_cluster_size={min_cluster_size}: {str(e)}")
                continue
        
        return hdbscan_results
    
    def run_dbscan_grid_search(self, embeddings: np.ndarray, df: pd.DataFrame, texts: list) -> dict:
        """Run narrow DBSCAN grid search with data-driven parameters."""
        self.logger.info("🔍 Running DBSCAN grid search...")
        
        # Use optimal parameters from diagnostics
        eps_values = [0.508, 0.544, 0.548]  # Conservative range
        min_samples_values = [4, 5]  # From diagnostics recommendations
        
        # Calculate distance matrix once
        distance_matrix = cosine_distances(embeddings)
        
        dbscan_results = []
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                try:
                    # Run DBSCAN
                    clusterer = DBSCAN(
                        eps=eps,
                        min_samples=min_samples,
                        metric='precomputed'
                    )
                    
                    labels = clusterer.fit_predict(distance_matrix)
                    
                    # Calculate metrics
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    noise_ratio = n_noise / len(labels)
                    
                    # Clustering metrics (excluding noise points)
                    non_noise_mask = labels != -1
                    metrics = {}
                    
                    if n_clusters > 1 and np.sum(non_noise_mask) > 0:
                        try:
                            metrics['davies_bouldin'] = davies_bouldin_score(
                                embeddings[non_noise_mask], 
                                labels[non_noise_mask]
                            )
                        except:
                            metrics['davies_bouldin'] = np.inf
                        
                        # Skip silhouette for large datasets
                        if len(embeddings) < 2000:
                            try:
                                metrics['silhouette'] = silhouette_score(
                                    embeddings[non_noise_mask], 
                                    labels[non_noise_mask]
                                )
                            except:
                                metrics['silhouette'] = -1
                        else:
                            metrics['silhouette'] = None
                    else:
                        metrics['davies_bouldin'] = np.inf
                        metrics['silhouette'] = -1
                    
                    # Analyze clusters
                    cluster_analysis = self._analyze_clusters(labels, df, texts)
                    
                    result = {
                        'algorithm': 'DBSCAN',
                        'parameters': {'eps': eps, 'min_samples': min_samples},
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'noise_ratio': noise_ratio,
                        'clustered_ratio': (len(labels) - n_noise) / len(labels),
                        'metrics': metrics,
                        'cluster_analysis': cluster_analysis,
                        'labels': labels.copy()
                    }
                    
                    dbscan_results.append(result)
                    
                    self.logger.info(f"   eps={eps}, min_samples={min_samples}: {n_clusters} clusters, {noise_ratio:.1%} noise")
                    
                except Exception as e:
                    self.logger.error(f"DBSCAN failed with eps={eps}, min_samples={min_samples}: {str(e)}")
                    continue
        
        return dbscan_results
    
    def run_agglomerative_experiment(self, embeddings: np.ndarray, df: pd.DataFrame, texts: list) -> dict:
        """Run Agglomerative clustering experiment."""
        self.logger.info("🔍 Running Agglomerative clustering experiment...")
        
        # Test different numbers of clusters
        n_clusters_values = [5, 10, 15, 20, 25, 30, 40, 50]
        agglomerative_results = []
        
        for n_clusters in n_clusters_values:
            try:
                # Run Agglomerative clustering
                clusterer = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage='ward'  # Works well with Euclidean distance
                )
                
                labels = clusterer.fit_predict(embeddings)
                
                # Calculate metrics
                n_noise = 0  # Agglomerative doesn't produce noise points
                noise_ratio = 0.0
                
                # Clustering metrics
                metrics = {}
                try:
                    metrics['davies_bouldin'] = davies_bouldin_score(embeddings, labels)
                except:
                    metrics['davies_bouldin'] = np.inf
                
                # Skip silhouette for large datasets
                if len(embeddings) < 2000:
                    try:
                        metrics['silhouette'] = silhouette_score(embeddings, labels)
                    except:
                        metrics['silhouette'] = -1
                else:
                    metrics['silhouette'] = None
                
                # Analyze clusters
                cluster_analysis = self._analyze_clusters(labels, df, texts)
                
                result = {
                    'algorithm': 'Agglomerative',
                    'parameters': {'n_clusters': n_clusters},
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'noise_ratio': noise_ratio,
                    'clustered_ratio': 1.0,  # All points are clustered
                    'metrics': metrics,
                    'cluster_analysis': cluster_analysis,
                    'labels': labels.copy()
                }
                
                agglomerative_results.append(result)
                
                self.logger.info(f"   n_clusters={n_clusters}: DB_score={metrics['davies_bouldin']:.3f}")
                
            except Exception as e:
                self.logger.error(f"Agglomerative failed with n_clusters={n_clusters}: {str(e)}")
                continue
        
        return agglomerative_results
    
    def _analyze_clusters(self, labels: np.ndarray, df: pd.DataFrame, texts: list) -> dict:
        """Analyze cluster composition and extract sample tickets."""
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Handle noise separately
        
        clusters = {}
        
        for cluster_id in sorted(unique_labels):
            cluster_mask = labels == cluster_id
            cluster_tickets = df[cluster_mask]
            cluster_texts = [texts[i] for i in range(len(texts)) if cluster_mask[i]]
            
            if len(cluster_tickets) == 0:
                continue
            
            # Sample tickets: first, median, last by index
            sample_indices = [0, len(cluster_tickets)//2, len(cluster_tickets)-1]
            sample_indices = list(set(sample_indices))  # Remove duplicates for small clusters
            
            cluster_info = {
                'size': len(cluster_tickets),
                'sample_descriptions': [
                    cluster_tickets.iloc[i]['Short description'] 
                    for i in sample_indices[:3]
                ],
                'sample_categories': [
                    cluster_tickets.iloc[i]['category'] 
                    for i in sample_indices[:3]
                ],
                'category_distribution': cluster_tickets['category'].value_counts().head(5).to_dict(),
                'sample_full_texts': cluster_texts[:3]  # First 3 for brevity
            }
            
            clusters[f"cluster_{cluster_id}"] = cluster_info
        
        # Handle noise points
        noise_mask = labels == -1
        if np.sum(noise_mask) > 0:
            noise_tickets = df[noise_mask]
            clusters['noise'] = {
                'size': int(np.sum(noise_mask)),
                'sample_descriptions': noise_tickets['Short description'].head(5).tolist(),
                'category_distribution': noise_tickets['category'].value_counts().head(5).to_dict()
            }
        
        return clusters
    
    def evaluate_and_rank_results(self, all_results: list) -> list:
        """Evaluate and rank all clustering results."""
        self.logger.info("📊 Evaluating and ranking clustering results...")
        
        ranked_results = []
        
        for result in all_results:
            # Calculate composite score
            score_components = {}
            
            # Cluster count score (prefer 5-30 clusters)
            n_clusters = result['n_clusters']
            if 5 <= n_clusters <= 30:
                score_components['cluster_count'] = 1.0
            elif n_clusters < 5:
                score_components['cluster_count'] = n_clusters / 5.0
            else:
                score_components['cluster_count'] = max(0.1, 30.0 / n_clusters)
            
            # Noise ratio score (prefer 20-40% noise for diverse data)
            noise_ratio = result['noise_ratio']
            if 0.2 <= noise_ratio <= 0.4:
                score_components['noise_balance'] = 1.0
            elif noise_ratio < 0.2:
                score_components['noise_balance'] = 0.7  # Too few noise points
            else:
                score_components['noise_balance'] = max(0.1, 1.0 - (noise_ratio - 0.4))
            
            # Davies-Bouldin score (lower is better, normalize)
            db_score = result['metrics'].get('davies_bouldin', np.inf)
            if db_score == np.inf or db_score > 10:
                score_components['davies_bouldin'] = 0.0
            else:
                score_components['davies_bouldin'] = max(0.0, 1.0 - (db_score / 5.0))
            
            # Silhouette score (higher is better)
            sil_score = result['metrics'].get('silhouette', None)
            if sil_score is None or sil_score < 0:
                score_components['silhouette'] = 0.5  # Neutral if not available
            else:
                score_components['silhouette'] = (sil_score + 1) / 2  # Normalize from [-1,1] to [0,1]
            
            # Cluster size balance (prefer clusters with reasonable sizes)
            cluster_sizes = []
            for cluster_info in result['cluster_analysis'].values():
                if 'size' in cluster_info and cluster_info != 'noise':
                    cluster_sizes.append(cluster_info['size'])
            
            if cluster_sizes:
                size_std = np.std(cluster_sizes)
                size_mean = np.mean(cluster_sizes)
                cv = size_std / size_mean if size_mean > 0 else 0
                score_components['size_balance'] = max(0.0, 1.0 - cv)
            else:
                score_components['size_balance'] = 0.0
            
            # Composite score (weighted average)
            weights = {
                'cluster_count': 0.25,
                'noise_balance': 0.20,
                'davies_bouldin': 0.25,
                'silhouette': 0.15,
                'size_balance': 0.15
            }
            
            composite_score = sum(
                weights[component] * score 
                for component, score in score_components.items()
            )
            
            result['evaluation'] = {
                'score_components': score_components,
                'composite_score': composite_score,
                'ranking_factors': {
                    'n_clusters': n_clusters,
                    'noise_ratio': noise_ratio,
                    'davies_bouldin': db_score,
                    'silhouette': sil_score,
                    'cluster_size_cv': cv if cluster_sizes else 0
                }
            }
            
            ranked_results.append(result)
        
        # Sort by composite score (descending)
        ranked_results.sort(key=lambda x: x['evaluation']['composite_score'], reverse=True)
        
        return ranked_results
    
    def generate_comparison_report(self, ranked_results: list, output_dir: Path):
        """Generate comprehensive comparison report."""
        self.logger.info("📋 Generating comparison report...")
        
        # Create summary table
        summary_data = []
        for i, result in enumerate(ranked_results[:10]):  # Top 10
            summary_data.append({
                'Rank': i + 1,
                'Algorithm': result['algorithm'],
                'Parameters': str(result['parameters']),
                'Clusters': result['n_clusters'],
                'Noise %': f"{result['noise_ratio']:.1%}",
                'Clustered %': f"{result['clustered_ratio']:.1%}",
                'Davies-Bouldin': f"{result['metrics'].get('davies_bouldin', np.inf):.3f}",
                'Silhouette': f"{result['metrics'].get('silhouette', 'N/A')}",
                'Composite Score': f"{result['evaluation']['composite_score']:.3f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_file = output_dir / 'clustering_comparison_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        
        # Generate detailed report
        report = {
            'experiment_timestamp': datetime.now().isoformat(),
            'summary_table': summary_data,
            'top_3_detailed': []
        }
        
        # Add detailed analysis for top 3
        for i, result in enumerate(ranked_results[:3]):
            detailed = {
                'rank': i + 1,
                'algorithm': result['algorithm'],
                'parameters': result['parameters'],
                'metrics': result['metrics'],
                'evaluation': result['evaluation'],
                'cluster_summary': {}
            }
            
            # Add cluster examples
            for cluster_name, cluster_info in result['cluster_analysis'].items():
                if cluster_name != 'noise':
                    detailed['cluster_summary'][cluster_name] = {
                        'size': cluster_info['size'],
                        'sample_descriptions': cluster_info['sample_descriptions'],
                        'top_categories': list(cluster_info['category_distribution'].keys())[:3]
                    }
            
            report['top_3_detailed'].append(detailed)
        
        # Save detailed report
        report_file = output_dir / 'clustering_experiments_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return summary_df, report

def main():
    """Main experimental pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # File paths
    results_file = Path('outputs/improved_classification_results.json')
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    if not results_file.exists():
        print(f"❌ Error: Classification results file not found at {results_file}")
        print("   Please run semantic_analysis.py first.")
        return
    
    print("="*80)
    print("CLUSTER EXPERIMENTS - SYSTEMATIC CLUSTERING METHOD COMPARISON")
    print("="*80)
    
    # Initialize experiments
    experiments = ClusteringExperiments()
    
    # Step 1: Load embeddings using proven pipeline
    print(f"\n🔄 STEP 1: Loading embeddings using proven pipeline")
    embeddings, df, texts = experiments.load_embeddings_and_data(results_file)
    
    # Step 2: Run HDBSCAN experiments (first priority)
    print(f"\n🔄 STEP 2: Running HDBSCAN experiments")
    hdbscan_results = experiments.run_hdbscan_experiment(embeddings, df, texts)
    
    # Step 3: Run DBSCAN grid search (narrow range)
    print(f"\n🔄 STEP 3: Running DBSCAN grid search")
    dbscan_results = experiments.run_dbscan_grid_search(embeddings, df, texts)
    
    # Step 4: Run Agglomerative clustering
    print(f"\n🔄 STEP 4: Running Agglomerative clustering experiments")
    agglomerative_results = experiments.run_agglomerative_experiment(embeddings, df, texts)
    
    # Step 5: Combine and evaluate all results
    print(f"\n🔄 STEP 5: Evaluating and ranking results")
    all_results = hdbscan_results + dbscan_results + agglomerative_results
    ranked_results = experiments.evaluate_and_rank_results(all_results)
    
    # Step 6: Generate comparison report
    print(f"\n🔄 STEP 6: Generating comparison report")
    summary_df, report = experiments.generate_comparison_report(ranked_results, output_dir)
    
    # Display results
    print(f"\n📊 CLUSTERING EXPERIMENTS RESULTS:")
    print(f"   Total experiments run: {len(all_results)}")
    print(f"   HDBSCAN experiments: {len(hdbscan_results)}")
    print(f"   DBSCAN experiments: {len(dbscan_results)}")
    print(f"   Agglomerative experiments: {len(agglomerative_results)}")
    
    print(f"\n🏆 TOP 5 CLUSTERING APPROACHES:")
    print(summary_df.head().to_string(index=False))
    
    # Show winner details
    if ranked_results:
        winner = ranked_results[0]
        print(f"\n🥇 WINNER: {winner['algorithm']}")
        print(f"   Parameters: {winner['parameters']}")
        print(f"   Clusters: {winner['n_clusters']}")
        print(f"   Noise ratio: {winner['noise_ratio']:.1%}")
        print(f"   Composite score: {winner['evaluation']['composite_score']:.3f}")
        
        # Show top clusters
        print(f"\n🎯 TOP CLUSTERS (Winner):")
        cluster_items = list(winner['cluster_analysis'].items())
        cluster_items = [(k, v) for k, v in cluster_items if k != 'noise']
        cluster_items.sort(key=lambda x: x[1]['size'], reverse=True)
        
        for i, (cluster_name, cluster_info) in enumerate(cluster_items[:5]):
            print(f"   {i+1}. {cluster_name}: {cluster_info['size']} tickets")
            print(f"      Sample: '{cluster_info['sample_descriptions'][0]}'")
            print(f"      Categories: {list(cluster_info['category_distribution'].keys())[:3]}")
    
    print(f"\n💾 RESULTS SAVED:")
    print(f"   📄 Summary table: {output_dir}/clustering_comparison_summary.csv")
    print(f"   📄 Detailed report: {output_dir}/clustering_experiments_report.json")
    
    print(f"\n🔄 NEXT STEPS:")
    print(f"   1. Review the top 3 clustering approaches")
    print(f"   2. Manually validate sample tickets from winner's clusters")
    print(f"   3. If satisfied, integrate winner into semantic_analysis.py")
    print(f"   4. If not satisfied, consider ensemble methods or parameter refinement")
    
    logger.info("Clustering experiments completed successfully!")

if __name__ == "__main__":
    main()