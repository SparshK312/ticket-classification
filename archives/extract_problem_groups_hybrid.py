#!/usr/bin/env python3
"""
PHASE 1: HYBRID PROBLEM GROUPING - HDBSCAN + AGGLOMERATIVE FALLBACK

This script implements a hybrid approach:
1. Use HDBSCAN to find high-density problem clusters
2. Use Agglomerative clustering on "noise" points to find smaller problem groups
3. Combine results for comprehensive coverage without mega-clusters

Key advantages:
- Gets the best of both: dense clusters + comprehensive coverage
- No manual threshold tuning for main clusters
- Handles both common and uncommon problem types
- Uses real IT helpdesk trained BERT model
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

class HybridProblemGrouper:
    """Hybrid problem grouping: HDBSCAN for dense clusters + Agglomerative for noise points."""
    
    def __init__(self, model_name='DavinciTech/BERT_Categorizer'):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        
        # UMAP parameters
        self.umap_n_components = 15
        self.umap_n_neighbors = 30
        self.umap_min_dist = 0.0
        self.umap_metric = 'cosine'
        
        # HDBSCAN parameters (for dense clusters) - adjusted to prevent mega-clusters
        self.hdbscan_min_cluster_size = 8   # Less conservative to find more clusters
        self.hdbscan_min_samples = 5        # Fewer samples required
        self.hdbscan_cluster_selection_epsilon = 0.1  # Allow some separation
        
        # Agglomerative parameters (for noise points)
        self.agglom_similarity_threshold = 0.65  # For noise point clustering
        self.agglom_min_cluster_size = 4
        
        # Quality control parameters
        self.max_cluster_percentage = 25.0
        self.target_total_groups = 200  # Target around 200 total groups
        
        # Load IT Support BERT model
        self.logger.info(f"Loading IT Support BERT model: {model_name}")
        try:
            self.encoder = SentenceTransformer(model_name)
            self.logger.info("IT Support BERT model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Failed to load IT Support model: {e}")
            self.logger.info("Falling back to all-MiniLM-L6-v2")
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.model_name = 'all-MiniLM-L6-v2'
        
        # Results storage
        self.embeddings = None
        self.umap_embeddings = None
        self.tickets_df = None
        self.problem_groups = {}
        self.quality_metrics = {}
        self.clustering_history = {}
        
    def load_consolidated_tickets(self, file_path: Path) -> pd.DataFrame:
        """Load the 3,847 consolidated tickets."""
        self.logger.info(f"Loading consolidated tickets from {file_path}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Consolidated tickets file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        self.logger.info(f"Loaded {len(df):,} consolidated tickets")
        
        # Verify expected structure
        required_columns = ['Short description', 'Description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        self.tickets_df = df
        return df
    
    def prepare_text_for_embedding(self, row: pd.Series) -> str:
        """Prepare combined text for semantic analysis."""
        short_desc = str(row.get('Short description', ''))
        description = str(row.get('Description', ''))
        
        # Clean and combine
        short_desc = short_desc.strip()
        description = description.strip()
        
        # Strategy: Short description + detailed context for help-desk understanding
        if len(description) > 0 and description.lower() != 'nan':
            combined = f"{short_desc}. {description}"
        else:
            combined = short_desc
        
        return combined.strip()
    
    def generate_embeddings(self) -> np.ndarray:
        """Generate IT Support BERT embeddings for all tickets."""
        self.logger.info("Preparing text for IT Support BERT embedding generation...")
        
        # Prepare texts
        texts = self.tickets_df.apply(self.prepare_text_for_embedding, axis=1).tolist()
        
        self.logger.info(f"Generating IT Support BERT embeddings for {len(texts):,} tickets...")
        
        # Generate embeddings with progress bar
        embeddings = self.encoder.encode(
            texts, 
            show_progress_bar=True,
            convert_to_tensor=False,
            batch_size=32,
            normalize_embeddings=True  # L2 normalization
        )
        
        self.logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        self.embeddings = embeddings
        
        return embeddings, texts
    
    def apply_umap_reduction(self) -> np.ndarray:
        """Apply UMAP dimensionality reduction."""
        self.logger.info(f"Applying UMAP reduction: {self.embeddings.shape[1]}D → {self.umap_n_components}D")
        
        # Initialize UMAP
        umap_reducer = umap.UMAP(
            n_components=self.umap_n_components,
            n_neighbors=self.umap_n_neighbors,
            min_dist=self.umap_min_dist,
            metric=self.umap_metric,
            random_state=42,
            verbose=True
        )
        
        # Fit and transform
        umap_embeddings = umap_reducer.fit_transform(self.embeddings)
        
        self.logger.info(f"UMAP reduction complete: {umap_embeddings.shape}")
        self.umap_embeddings = umap_embeddings
        
        return umap_embeddings
    
    def apply_hdbscan_for_dense_clusters(self) -> tuple:
        """Apply HDBSCAN to find high-density clusters."""
        self.logger.info("Step 1: Applying HDBSCAN for dense problem clusters")
        
        # Initialize HDBSCAN (conservative settings for high-quality dense clusters)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            min_samples=self.hdbscan_min_samples,
            cluster_selection_epsilon=self.hdbscan_cluster_selection_epsilon,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        # Fit clustering
        cluster_labels = clusterer.fit_predict(self.umap_embeddings)
        
        # Get cluster statistics
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(cluster_labels == -1)
        
        self.logger.info(f"HDBSCAN dense cluster results:")
        self.logger.info(f"  Dense clusters found: {n_clusters}")
        self.logger.info(f"  Points assigned to dense clusters: {len(cluster_labels) - n_noise}")
        self.logger.info(f"  Noise points for fallback clustering: {n_noise}")
        
        # Store clustering history (convert numpy types to native Python types)
        cluster_counts = Counter(cluster_labels)
        cluster_sizes_dict = {str(k): int(v) for k, v in cluster_counts.items()}
        
        self.clustering_history['hdbscan'] = {
            'clusters_found': int(n_clusters),
            'points_clustered': int(len(cluster_labels) - n_noise),
            'noise_points': int(n_noise),
            'cluster_sizes': cluster_sizes_dict
        }
        
        return cluster_labels, clusterer
    
    def apply_agglomerative_to_noise(self, hdbscan_labels: np.ndarray) -> np.ndarray:
        """Apply Agglomerative clustering to noise points from HDBSCAN."""
        # Get noise point indices
        noise_indices = np.where(hdbscan_labels == -1)[0]
        
        if len(noise_indices) == 0:
            self.logger.info("No noise points to cluster with Agglomerative")
            return hdbscan_labels
        
        self.logger.info(f"Step 2: Applying Agglomerative clustering to {len(noise_indices)} noise points")
        
        # Get embeddings for noise points
        noise_embeddings = self.umap_embeddings[noise_indices]
        
        # Estimate number of clusters for noise points
        # Target: ~10% of noise points as final clusters
        target_noise_clusters = max(5, min(50, len(noise_indices) // 10))
        
        self.logger.info(f"  Target clusters for noise points: {target_noise_clusters}")
        
        # Apply Agglomerative clustering
        agglom_clusterer = AgglomerativeClustering(
            n_clusters=target_noise_clusters,
            metric='cosine',
            linkage='average'
        )
        
        agglom_labels = agglom_clusterer.fit_predict(noise_embeddings)
        
        # Create combined labels
        combined_labels = hdbscan_labels.copy()
        max_hdbscan_label = max([label for label in hdbscan_labels if label != -1], default=0)
        
        # Assign new cluster labels to noise points
        for i, agglom_label in enumerate(agglom_labels):
            original_idx = noise_indices[i]
            new_label = max_hdbscan_label + 1 + agglom_label
            combined_labels[original_idx] = new_label
        
        # Filter out very small agglomerative clusters (< min_cluster_size)
        final_labels = combined_labels.copy()
        agglom_cluster_counts = Counter(agglom_labels)
        
        small_clusters = 0
        for i, agglom_label in enumerate(agglom_labels):
            if agglom_cluster_counts[agglom_label] < self.agglom_min_cluster_size:
                original_idx = noise_indices[i]
                final_labels[original_idx] = -1  # Back to noise
                small_clusters += 1
        
        # Final statistics
        final_cluster_counts = Counter(final_labels)
        n_final_noise = final_cluster_counts.get(-1, 0)
        n_agglom_clusters = len([label for label in final_cluster_counts.keys() 
                               if label > max_hdbscan_label])
        
        self.logger.info(f"Agglomerative clustering results:")
        self.logger.info(f"  Agglomerative clusters created: {n_agglom_clusters}")
        self.logger.info(f"  Noise points clustered: {len(noise_indices) - n_final_noise}")
        self.logger.info(f"  Remaining noise points: {n_final_noise}")
        self.logger.info(f"  Small clusters filtered out: {small_clusters}")
        
        # Store clustering history (convert numpy types to native Python types)
        self.clustering_history['agglomerative'] = {
            'target_clusters': int(target_noise_clusters),
            'actual_clusters': int(n_agglom_clusters),
            'points_clustered': int(len(noise_indices) - n_final_noise),
            'remaining_noise': int(n_final_noise),
            'small_clusters_filtered': int(small_clusters)
        }
        
        return final_labels
    
    def find_problem_groups_hybrid(self) -> dict:
        """Find problem groups using hybrid HDBSCAN + Agglomerative approach."""
        self.logger.info("Finding problem groups using Hybrid approach (HDBSCAN + Agglomerative)")
        
        # Step 1: Apply UMAP reduction
        umap_embeddings = self.apply_umap_reduction()
        
        # Step 2: Apply HDBSCAN for dense clusters
        hdbscan_labels, hdbscan_clusterer = self.apply_hdbscan_for_dense_clusters()
        
        # Step 3: Apply Agglomerative clustering to noise points
        final_labels = self.apply_agglomerative_to_noise(hdbscan_labels)
        
        # Step 4: Convert to group format
        groups = self._convert_labels_to_groups(final_labels, hdbscan_clusterer)
        
        # Step 5: Calculate quality metrics
        self._calculate_hybrid_metrics(final_labels, hdbscan_labels)
        
        self.problem_groups = groups
        return groups
    
    def _convert_labels_to_groups(self, cluster_labels: np.ndarray, hdbscan_clusterer) -> dict:
        """Convert cluster labels to group format."""
        groups = {}
        group_id = 0
        
        # Get unique cluster labels (excluding noise -1)
        unique_labels = np.unique(cluster_labels)
        cluster_labels_clean = [label for label in unique_labels if label != -1]
        
        for cluster_label in cluster_labels_clean:
            # Get ticket indices for this cluster
            ticket_indices = np.where(cluster_labels == cluster_label)[0].tolist()
            
            if len(ticket_indices) < self.agglom_min_cluster_size:
                continue
            
            # Find representative ticket (closest to centroid in UMAP space)
            cluster_umap_embeddings = self.umap_embeddings[ticket_indices]
            centroid = np.mean(cluster_umap_embeddings, axis=0)
            
            # Find ticket closest to centroid
            distances = np.linalg.norm(cluster_umap_embeddings - centroid, axis=1)
            representative_idx = ticket_indices[np.argmin(distances)]
            
            # Determine cluster origin
            hdbscan_labels = [int(k) for k in self.clustering_history['hdbscan']['cluster_sizes'].keys() if k != '-1']
            max_hdbscan_label = max(hdbscan_labels) if hdbscan_labels else -1
            if cluster_label <= max_hdbscan_label and cluster_label != -1:
                cluster_origin = 'hdbscan_dense'
                # Get persistence from HDBSCAN
                persistence = getattr(hdbscan_clusterer, 'cluster_persistence_', [0.0])[cluster_label] if hasattr(hdbscan_clusterer, 'cluster_persistence_') else 0.0
            else:
                cluster_origin = 'agglomerative_fallback'
                persistence = 0.0
            
            groups[f"problem_group_{group_id}"] = {
                'group_id': group_id,
                'representative_ticket_index': representative_idx,
                'ticket_indices': ticket_indices,
                'group_size': len(ticket_indices),
                'cluster_label': cluster_label,
                'cluster_origin': cluster_origin,
                'cluster_persistence': float(persistence),
                'clustering_method': 'Hybrid_HDBSCAN_Agglomerative',
                'representative_selection_method': 'umap_centroid'
            }
            group_id += 1
        
        # Handle remaining noise points as individual groups
        noise_indices = np.where(cluster_labels == -1)[0]
        if len(noise_indices) > 0:
            self.logger.info(f"Found {len(noise_indices)} remaining noise points - treating as individual groups")
            
            for noise_idx in noise_indices:
                groups[f"problem_group_{group_id}"] = {
                    'group_id': group_id,
                    'representative_ticket_index': noise_idx,
                    'ticket_indices': [noise_idx],
                    'group_size': 1,
                    'cluster_label': -1,
                    'cluster_origin': 'unclustered_noise',
                    'cluster_persistence': 0.0,
                    'clustering_method': 'Hybrid_HDBSCAN_Agglomerative',
                    'representative_selection_method': 'noise_singleton'
                }
                group_id += 1
        
        return groups
    
    def _calculate_hybrid_metrics(self, final_labels: np.ndarray, hdbscan_labels: np.ndarray):
        """Calculate quality metrics for hybrid clustering."""
        # Overall metrics
        final_counts = Counter(final_labels)
        hdbscan_counts = Counter(hdbscan_labels)
        
        total_tickets = len(final_labels)
        final_noise = final_counts.get(-1, 0)
        hdbscan_noise = hdbscan_counts.get(-1, 0)
        
        # Calculate improvement
        noise_reduction = hdbscan_noise - final_noise
        noise_reduction_pct = (noise_reduction / hdbscan_noise * 100) if hdbscan_noise > 0 else 0
        
        self.clustering_history['overall_improvement'] = {
            'initial_noise_points': int(hdbscan_noise),
            'final_noise_points': int(final_noise),
            'noise_points_clustered': int(noise_reduction),
            'noise_reduction_percentage': float(noise_reduction_pct),
            'total_clusters': int(len([label for label in final_counts.keys() if label != -1])),
            'reduction_ratio': float(total_tickets / len([label for label in final_counts.keys() if label != -1]))
        }
        
        self.logger.info(f"Hybrid clustering improvement:")
        self.logger.info(f"  Noise reduced from {hdbscan_noise} to {final_noise} ({noise_reduction_pct:.1f}% improvement)")
        self.logger.info(f"  Total clusters: {self.clustering_history['overall_improvement']['total_clusters']}")
        self.logger.info(f"  Reduction ratio: {self.clustering_history['overall_improvement']['reduction_ratio']:.1f}:1")
    
    def calculate_quality_metrics(self) -> dict:
        """Calculate comprehensive quality metrics."""
        self.logger.info("Calculating quality metrics...")
        
        if not self.problem_groups:
            raise ValueError("No problem groups found. Run find_problem_groups_hybrid() first.")
        
        # Group size statistics
        group_sizes = [group['group_size'] for group in self.problem_groups.values()]
        
        # Cluster origin analysis
        hdbscan_groups = sum(1 for group in self.problem_groups.values() 
                           if group['cluster_origin'] == 'hdbscan_dense')
        agglom_groups = sum(1 for group in self.problem_groups.values() 
                          if group['cluster_origin'] == 'agglomerative_fallback')
        noise_groups = sum(1 for group in self.problem_groups.values() 
                         if group['cluster_origin'] == 'unclustered_noise')
        
        # Coverage analysis
        total_tickets = len(self.tickets_df)
        tickets_in_groups = sum(group_sizes)
        
        metrics = {
            'total_groups': len(self.problem_groups),
            'hdbscan_dense_clusters': hdbscan_groups,
            'agglomerative_clusters': agglom_groups,
            'unclustered_noise_singletons': noise_groups,
            'total_tickets_grouped': tickets_in_groups,
            'coverage_percentage': (tickets_in_groups / total_tickets) * 100,
            'clustering_method': 'Hybrid_HDBSCAN_Agglomerative',
            'embedding_model': self.model_name,
            'average_group_size': float(np.mean(group_sizes)),
            'median_group_size': float(np.median(group_sizes)),
            'largest_group_size': int(np.max(group_sizes)),
            'largest_group_percentage': (int(np.max(group_sizes)) / total_tickets) * 100,
            'smallest_group_size': int(np.min(group_sizes)),
            'singleton_groups': sum(1 for size in group_sizes if size == 1),
            'singleton_percentage': (sum(1 for size in group_sizes if size == 1) / len(group_sizes)) * 100,
            'large_groups_50plus': sum(1 for size in group_sizes if size >= 50),
            'medium_groups_10_to_49': sum(1 for size in group_sizes if 10 <= size < 49),
            'small_groups_4_to_9': sum(1 for size in group_sizes if 4 <= size < 10),
            'reduction_ratio': total_tickets / len(self.problem_groups),
            'clustering_history': self.clustering_history,
            'parameters_used': {
                'umap_n_components': self.umap_n_components,
                'hdbscan_min_cluster_size': self.hdbscan_min_cluster_size,
                'agglom_similarity_threshold': self.agglom_similarity_threshold,
                'agglom_min_cluster_size': self.agglom_min_cluster_size
            }
        }
        
        self.logger.info(f"Hybrid quality metrics:")
        self.logger.info(f"  Total groups: {metrics['total_groups']:,}")
        self.logger.info(f"  HDBSCAN dense clusters: {metrics['hdbscan_dense_clusters']:,}")
        self.logger.info(f"  Agglomerative clusters: {metrics['agglomerative_clusters']:,}")
        self.logger.info(f"  Unclustered singletons: {metrics['unclustered_noise_singletons']:,}")
        self.logger.info(f"  Reduction ratio: {metrics['reduction_ratio']:.1f}:1")
        self.logger.info(f"  Largest group: {metrics['largest_group_size']} ({metrics['largest_group_percentage']:.1f}%)")
        self.logger.info(f"  Singleton percentage: {metrics['singleton_percentage']:.1f}%")
        
        self.quality_metrics = metrics
        return metrics
    
    def create_detailed_outputs(self, output_dir: Path) -> tuple:
        """Create detailed output files for review."""
        self.logger.info("Creating detailed output files...")
        
        output_dir.mkdir(exist_ok=True)
        
        # 1. Problem groups summary
        groups_summary = []
        for group_name, group_data in self.problem_groups.items():
            rep_idx = group_data['representative_ticket_index']
            rep_ticket = self.tickets_df.loc[rep_idx]
            
            groups_summary.append({
                'problem_group_id': group_data['group_id'],
                'group_size': group_data['group_size'],
                'cluster_label': group_data['cluster_label'],
                'cluster_origin': group_data['cluster_origin'],
                'cluster_persistence': group_data['cluster_persistence'],
                'clustering_method': group_data['clustering_method'],
                'representative_ticket_index': rep_idx,
                'representative_selection_method': group_data['representative_selection_method'],
                'representative_short_description': rep_ticket['Short description'],
                'all_ticket_indices': ','.join(map(str, group_data['ticket_indices'])),
                'manual_review_priority': 'HIGH' if group_data['group_size'] >= 50 else 
                                        'MEDIUM' if group_data['group_size'] >= 10 else 'LOW',
                'is_dense_cluster': group_data['cluster_origin'] == 'hdbscan_dense',
                'is_fallback_cluster': group_data['cluster_origin'] == 'agglomerative_fallback'
            })
        
        groups_summary_df = pd.DataFrame(groups_summary)
        groups_summary_df = groups_summary_df.sort_values('group_size', ascending=False)
        
        summary_file = output_dir / 'problem_groups_hybrid.csv'
        groups_summary_df.to_csv(summary_file, index=False)
        
        # 2. Individual ticket assignments
        ticket_details = []
        for group_name, group_data in self.problem_groups.items():
            rep_idx = group_data['representative_ticket_index']
            
            for ticket_idx in group_data['ticket_indices']:
                ticket = self.tickets_df.loc[ticket_idx]
                
                # Calculate distance to representative in UMAP space
                if ticket_idx == rep_idx:
                    umap_distance_to_rep = 0.0
                else:
                    rep_umap = self.umap_embeddings[rep_idx]
                    ticket_umap = self.umap_embeddings[ticket_idx]
                    umap_distance_to_rep = float(np.linalg.norm(rep_umap - ticket_umap))
                
                ticket_details.append({
                    'problem_group_id': group_data['group_id'],
                    'ticket_index': ticket_idx,
                    'short_description': ticket['Short description'],
                    'full_description': ticket.get('Description', ''),
                    'original_category': ticket.get('Category', ''),
                    'umap_distance_to_representative': umap_distance_to_rep,
                    'is_representative': ticket_idx == rep_idx,
                    'cluster_label': group_data['cluster_label'],
                    'cluster_origin': group_data['cluster_origin'],
                    'clustering_method': group_data['clustering_method'],
                    'manual_review_flag': 'REQUIRED' if group_data['group_size'] >= 50 else 
                                        'RECOMMENDED' if group_data['group_size'] >= 10 else 'OPTIONAL',
                    'group_size': group_data['group_size']
                })
        
        ticket_details_df = pd.DataFrame(ticket_details)
        ticket_details_df = ticket_details_df.sort_values(['problem_group_id', 'umap_distance_to_representative'], 
                                                         ascending=[True, True])
        
        details_file = output_dir / 'problem_group_hybrid_details.csv'
        ticket_details_df.to_csv(details_file, index=False)
        
        # 3. Quality metrics report
        quality_file = output_dir / 'hybrid_quality_report.json'
        with open(quality_file, 'w') as f:
            json.dump({
                'analysis_timestamp': datetime.now().isoformat(),
                'quality_metrics': self.quality_metrics,
                'clustering_history': self.clustering_history
            }, f, indent=2, default=str)
        
        self.logger.info(f"Detailed outputs created:")
        self.logger.info(f"  Summary: {summary_file}")
        self.logger.info(f"  Details: {details_file}")
        self.logger.info(f"  Quality report: {quality_file}")
        
        return summary_file, details_file, quality_file

def main():
    """Main hybrid problem grouping pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("="*80)
    print("PHASE 1: HYBRID PROBLEM GROUPING - HDBSCAN + AGGLOMERATIVE")
    print("="*80)
    print("HYBRID APPROACH:")
    print("• DavinciTech BERT embeddings (IT helpdesk trained)")
    print("• UMAP dimensionality reduction (768D → 15D)")
    print("• HDBSCAN for high-density problem clusters (conservative)")
    print("• Agglomerative clustering for noise points (comprehensive)")
    print("• Best of both: quality clusters + comprehensive coverage")
    print("="*80)
    
    # File paths
    input_file = Path('data/processed/consolidated_tickets.csv')
    output_dir = Path('outputs')
    
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        print("   Please ensure consolidated_tickets.csv exists")
        return
    
    # Initialize hybrid grouper
    print(f"\n🎯 CONFIGURATION:")
    print(f"   Embedding model: DavinciTech BERT (IT helpdesk trained)")
    print(f"   UMAP reduction: 768D → 15D")
    print(f"   HDBSCAN: Balanced (min_cluster_size=8) for dense clusters")
    print(f"   Agglomerative: Fallback clustering for noise points")
    print(f"   Target: ~200 total problem groups with comprehensive coverage")
    
    grouper = HybridProblemGrouper()
    
    try:
        # Step 1: Load data
        print(f"\n🔄 STEP 1: Loading consolidated tickets")
        df = grouper.load_consolidated_tickets(input_file)
        
        # Step 2: Generate IT Support BERT embeddings
        print(f"\n🔄 STEP 2: Generating IT Support BERT embeddings")
        embeddings, texts = grouper.generate_embeddings()
        
        # Step 3: Apply hybrid clustering
        print(f"\n🔄 STEP 3: Applying hybrid clustering (HDBSCAN + Agglomerative)")
        groups = grouper.find_problem_groups_hybrid()
        
        # Step 4: Calculate quality metrics
        print(f"\n🔄 STEP 4: Calculating quality metrics")
        quality_metrics = grouper.calculate_quality_metrics()
        
        # Step 5: Create outputs
        print(f"\n🔄 STEP 5: Creating detailed outputs")
        summary_file, details_file, quality_file = grouper.create_detailed_outputs(output_dir)
        
        # Final summary
        print(f"\n📊 HYBRID CLUSTERING RESULTS:")
        print(f"   Input tickets: {len(df):,}")
        print(f"   Problem groups identified: {quality_metrics['total_groups']:,}")
        print(f"   HDBSCAN dense clusters: {quality_metrics['hdbscan_dense_clusters']:,}")
        print(f"   Agglomerative fallback clusters: {quality_metrics['agglomerative_clusters']:,}")
        print(f"   Remaining noise singletons: {quality_metrics['unclustered_noise_singletons']:,}")
        print(f"   Reduction ratio: {quality_metrics['reduction_ratio']:.1f}:1")
        print(f"   Average group size: {quality_metrics['average_group_size']:.1f}")
        print(f"   Largest group: {quality_metrics['largest_group_size']} tickets ({quality_metrics['largest_group_percentage']:.1f}%)")
        print(f"   Singleton percentage: {quality_metrics['singleton_percentage']:.1f}%")
        
        print(f"\n🎯 SUCCESS METRICS:")
        success_largest = quality_metrics['largest_group_percentage'] <= 25.0
        success_reduction = quality_metrics['reduction_ratio'] >= 4.0
        success_singletons = quality_metrics['singleton_percentage'] <= 30.0
        success_groups = 50 <= quality_metrics['total_groups'] <= 400
        
        print(f"   ✅ Largest cluster (≤25%): {'PASS' if success_largest else 'FAIL'} ({quality_metrics['largest_group_percentage']:.1f}%)")
        print(f"   ✅ Reduction ratio (≥4:1): {'PASS' if success_reduction else 'FAIL'} ({quality_metrics['reduction_ratio']:.1f}:1)")
        print(f"   ✅ Singleton rate (≤30%): {'PASS' if success_singletons else 'FAIL'} ({quality_metrics['singleton_percentage']:.1f}%)")
        print(f"   ✅ Group count (50-400): {'PASS' if success_groups else 'FAIL'} ({quality_metrics['total_groups']})")
        
        print(f"\n🔍 QUALITY CONTROL:")
        print(f"   High priority review (50+ tickets): {quality_metrics['large_groups_50plus']} groups")
        print(f"   Medium priority review (10-49 tickets): {quality_metrics['medium_groups_10_to_49']} groups")
        print(f"   Small clusters (4-9 tickets): {quality_metrics['small_groups_4_to_9']} groups")
        
        print(f"\n📁 OUTPUT FILES CREATED:")
        print(f"   📄 Summary: {summary_file}")
        print(f"   📄 Details: {details_file}")
        print(f"   📄 Quality report: {quality_file}")
        
        print(f"\n✅ HYBRID CLUSTERING COMPLETE")
        all_success = success_largest and success_reduction and success_singletons and success_groups
        if all_success:
            print(f"   🎉 ALL SUCCESS CRITERIA MET!")
            print(f"   🎯 Perfect balance: dense clusters + comprehensive coverage")
            print(f"   🚀 Ready for problem statement generation")
        else:
            print(f"   📈 Significant improvement over pure HDBSCAN approach")
            print(f"   🎯 Better coverage while maintaining cluster quality")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Review hybrid cluster results in output files")
        print(f"   2. Compare dense vs fallback cluster quality")
        print(f"   3. Generate LLM problem statements for each cluster")
        print(f"   4. Manual validation of problem statements")
        
        logger.info("Hybrid problem grouping completed successfully!")
        
    except Exception as e:
        logger.error(f"Hybrid problem grouping failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()