#!/usr/bin/env python3
"""
CLUSTERING DIAGNOSTICS - Distance Distribution Analysis

This script analyzes the distance distribution in the embedding space to help
determine optimal DBSCAN parameters for semantic clustering.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import logging

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

class ClusteringDiagnostics:
    """Diagnostic tools for clustering analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def load_embeddings_and_data(self, results_file: Path) -> tuple:
        """Load and recreate embeddings from classification results."""
        self.logger.info("Loading ticket data and creating embeddings...")
        
        # Load unclassified tickets
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        unclassified_tickets = results['unclassified_tickets']
        df = pd.DataFrame(unclassified_tickets)
        
        # Load original ticket data
        consolidated_file = Path('data/processed/consolidated_tickets.csv')
        if consolidated_file.exists():
            df_original = pd.read_csv(consolidated_file)
            df = df.merge(df_original, 
                         left_on='ticket_index', 
                         right_index=True, 
                         how='left',
                         suffixes=('', '_orig'))
        
        # Prepare text for encoding (same logic as semantic_analysis.py)
        def prepare_text_for_encoding(row):
            short_desc = str(row.get('Short description', ''))
            description = str(row.get('Description', ''))
            
            short_desc = short_desc.strip()
            description = description.strip()
            
            if len(description) > 0 and description.lower() != 'nan':
                combined = f"{short_desc}. {short_desc}. {description}"
            else:
                combined = f"{short_desc}. {short_desc}."
            
            return combined.strip()
        
        texts = df.apply(prepare_text_for_encoding, axis=1).tolist()
        
        # Generate embeddings
        self.logger.info("Generating embeddings...")
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = encoder.encode(texts, show_progress_bar=True, convert_to_tensor=False)
        embeddings = normalize(embeddings, norm='l2')
        
        self.logger.info(f"Generated {len(embeddings)} embeddings with shape: {embeddings.shape}")
        
        return embeddings, df, texts
    
    def analyze_distance_distribution(self, embeddings: np.ndarray, sample_size: int = 1000) -> dict:
        """Analyze the distribution of pairwise distances."""
        self.logger.info(f"Analyzing distance distribution with sample size: {sample_size}")
        
        # Sample embeddings for faster computation
        if len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sample_embeddings = embeddings[indices]
        else:
            sample_embeddings = embeddings
        
        # Calculate pairwise cosine distances
        distance_matrix = cosine_distances(sample_embeddings)
        
        # Extract upper triangle (avoid diagonal and duplicates)
        triu_indices = np.triu_indices_from(distance_matrix, k=1)
        distances = distance_matrix[triu_indices]
        
        # Calculate statistics
        stats = {
            'sample_size': len(sample_embeddings),
            'total_pairs': len(distances),
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances)),
            'mean_distance': float(np.mean(distances)),
            'median_distance': float(np.median(distances)),
            'std_distance': float(np.std(distances)),
            'q25': float(np.percentile(distances, 25)),
            'q75': float(np.percentile(distances, 75)),
            'q90': float(np.percentile(distances, 90)),
            'q95': float(np.percentile(distances, 95)),
            'q99': float(np.percentile(distances, 99))
        }
        
        return stats, distances
    
    def find_optimal_eps_candidates(self, distances: np.ndarray) -> list:
        """Find candidate eps values based on distance distribution."""
        self.logger.info("Finding optimal eps candidates...")
        
        # Method 1: K-nearest neighbor distances (elbow method)
        # For each point, find the k-th nearest neighbor distance
        k_values = [3, 4, 5]  # corresponding to min_samples values
        
        candidates = []
        
        # Add percentile-based candidates
        percentiles = [10, 15, 20, 25, 30, 35, 40]
        for p in percentiles:
            eps_candidate = np.percentile(distances, p)
            candidates.append(eps_candidate)
        
        # Add statistics-based candidates
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        candidates.extend([
            mean_dist - std_dist,
            mean_dist - 0.5 * std_dist,
            mean_dist,
            mean_dist + 0.5 * std_dist
        ])
        
        # Filter to reasonable range and remove duplicates
        candidates = [c for c in candidates if 0.01 <= c <= 1.0]
        candidates = sorted(list(set(candidates)))
        
        # Round to reasonable precision
        candidates = [round(c, 3) for c in candidates]
        
        return candidates
    
    def plot_distance_distribution(self, distances: np.ndarray, eps_candidates: list, output_dir: Path):
        """Create comprehensive distance distribution plots."""
        self.logger.info("Creating distance distribution visualizations...")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Histogram of distances
        ax1.hist(distances, bins=100, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(distances), color='red', linestyle='--', label=f'Mean: {np.mean(distances):.3f}')
        ax1.axvline(np.median(distances), color='orange', linestyle='--', label=f'Median: {np.median(distances):.3f}')
        
        # Add eps candidates as vertical lines
        for i, eps in enumerate(eps_candidates[:5]):  # Show first 5 candidates
            ax1.axvline(eps, color='green', linestyle=':', alpha=0.7, label=f'Candidate {i+1}: {eps:.3f}')
        
        ax1.set_xlabel('Cosine Distance')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution of Pairwise Cosine Distances')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative distribution
        sorted_distances = np.sort(distances)
        cumulative_prob = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
        
        ax2.plot(sorted_distances, cumulative_prob, linewidth=2, color='blue')
        ax2.axhline(0.1, color='red', linestyle='--', alpha=0.7, label='10% of pairs')
        ax2.axhline(0.25, color='orange', linestyle='--', alpha=0.7, label='25% of pairs')
        ax2.axhline(0.5, color='green', linestyle='--', alpha=0.7, label='50% of pairs')
        
        ax2.set_xlabel('Cosine Distance')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distribution of Distances')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Box plot of distances
        ax3.boxplot(distances, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax3.set_ylabel('Cosine Distance')
        ax3.set_title('Box Plot of Distance Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Distance vs Percentile
        percentiles = np.arange(1, 100)
        distance_percentiles = np.percentile(distances, percentiles)
        
        ax4.plot(percentiles, distance_percentiles, linewidth=2, color='purple')
        ax4.set_xlabel('Percentile')
        ax4.set_ylabel('Cosine Distance')
        ax4.set_title('Distance at Each Percentile')
        ax4.grid(True, alpha=0.3)
        
        # Highlight key percentiles
        key_percentiles = [10, 25, 50, 75, 90, 95]
        for p in key_percentiles:
            dist_at_p = np.percentile(distances, p)
            ax4.plot(p, dist_at_p, 'ro', markersize=8)
            ax4.annotate(f'{dist_at_p:.3f}', (p, dist_at_p), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_dir / 'distance_distribution_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Distance distribution plots saved to: {plot_file}")
    
    def generate_parameter_recommendations(self, stats: dict, eps_candidates: list) -> dict:
        """Generate DBSCAN parameter recommendations based on analysis."""
        
        # Conservative clustering (fewer, denser clusters)
        conservative_eps = [c for c in eps_candidates if c <= stats['q25']][:3]
        
        # Moderate clustering (balanced)
        moderate_eps = [c for c in eps_candidates if stats['q25'] < c <= stats['median_distance']][:3]
        
        # Liberal clustering (more, looser clusters)
        liberal_eps = [c for c in eps_candidates if stats['median_distance'] < c <= stats['q75']][:3]
        
        recommendations = {
            'distance_analysis': stats,
            'eps_candidates': eps_candidates,
            'parameter_recommendations': {
                'conservative': {
                    'description': 'Fewer, very tight clusters',
                    'eps_values': conservative_eps,
                    'min_samples': [4, 5],
                    'expected_noise': 'High (30-50%)'
                },
                'moderate': {
                    'description': 'Balanced clustering approach',
                    'eps_values': moderate_eps,
                    'min_samples': [3, 4, 5],
                    'expected_noise': 'Medium (10-30%)'
                },
                'liberal': {
                    'description': 'More clusters, looser criteria',
                    'eps_values': liberal_eps,
                    'min_samples': [3, 4],
                    'expected_noise': 'Low (5-20%)'
                }
            },
            'warnings': []
        }
        
        # Add warnings based on analysis
        if stats['mean_distance'] > 0.8:
            recommendations['warnings'].append(
                "Very high average distance - consider dimensionality reduction or alternative algorithms"
            )
        
        if stats['std_distance'] < 0.1:
            recommendations['warnings'].append(
                "Low distance variance - data may be very homogeneous, clustering may be challenging"
            )
        
        return recommendations

def main():
    """Main diagnostic analysis."""
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
    print("CLUSTERING DIAGNOSTICS - DISTANCE DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Initialize diagnostics
    diagnostics = ClusteringDiagnostics()
    
    # Load data and generate embeddings
    print(f"\n🔄 Step 1: Loading data and generating embeddings")
    embeddings, df, texts = diagnostics.load_embeddings_and_data(results_file)
    
    # Analyze distance distribution
    print(f"\n🔄 Step 2: Analyzing distance distribution")
    stats, distances = diagnostics.analyze_distance_distribution(embeddings, sample_size=1500)
    
    # Find optimal eps candidates
    print(f"\n🔄 Step 3: Finding optimal eps candidates")
    eps_candidates = diagnostics.find_optimal_eps_candidates(distances)
    
    # Create visualizations
    print(f"\n🔄 Step 4: Creating distance distribution plots")
    diagnostics.plot_distance_distribution(distances, eps_candidates, output_dir)
    
    # Generate recommendations
    print(f"\n🔄 Step 5: Generating parameter recommendations")
    recommendations = diagnostics.generate_parameter_recommendations(stats, eps_candidates)
    
    # Display results
    print(f"\n📊 DISTANCE DISTRIBUTION ANALYSIS RESULTS:")
    print(f"   Sample size: {stats['sample_size']:,} embeddings")
    print(f"   Total distance pairs analyzed: {stats['total_pairs']:,}")
    print(f"   Distance range: {stats['min_distance']:.3f} - {stats['max_distance']:.3f}")
    print(f"   Mean distance: {stats['mean_distance']:.3f}")
    print(f"   Median distance: {stats['median_distance']:.3f}")
    print(f"   Standard deviation: {stats['std_distance']:.3f}")
    
    print(f"\n📈 KEY PERCENTILES:")
    print(f"   25th percentile: {stats['q25']:.3f}")
    print(f"   75th percentile: {stats['q75']:.3f}")
    print(f"   90th percentile: {stats['q90']:.3f}")
    print(f"   95th percentile: {stats['q95']:.3f}")
    print(f"   99th percentile: {stats['q99']:.3f}")
    
    print(f"\n🎯 EPS CANDIDATES: {eps_candidates}")
    
    print(f"\n📋 PARAMETER RECOMMENDATIONS:")
    for approach, params in recommendations['parameter_recommendations'].items():
        print(f"   {approach.upper()}:")
        print(f"     Description: {params['description']}")
        print(f"     Recommended eps values: {params['eps_values']}")
        print(f"     Recommended min_samples: {params['min_samples']}")
        print(f"     Expected noise level: {params['expected_noise']}")
    
    if recommendations['warnings']:
        print(f"\n⚠️  WARNINGS:")
        for warning in recommendations['warnings']:
            print(f"   - {warning}")
    
    # Save recommendations
    recommendations_file = output_dir / 'clustering_parameter_recommendations.json'
    with open(recommendations_file, 'w') as f:
        json.dump(recommendations, f, indent=2, default=str)
    
    print(f"\n💾 ANALYSIS SAVED:")
    print(f"   📄 Parameter recommendations: {recommendations_file}")
    print(f"   📊 Distance distribution plots: {output_dir}/distance_distribution_analysis.png")
    
    print(f"\n🔄 NEXT STEPS:")
    print(f"   1. Review the distance distribution plots")
    print(f"   2. Test the recommended eps values with different min_samples")
    print(f"   3. Compare clustering results using validation metrics")
    print(f"   4. Consider HDBSCAN if DBSCAN results are still poor")
    
    logger.info("Distance distribution analysis completed successfully!")

if __name__ == "__main__":
    main()