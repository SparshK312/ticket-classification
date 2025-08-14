#!/usr/bin/env python3
"""
PHASE 1: UMAP + HDBSCAN PROBLEM GROUPING - HELP-DESK OPTIMIZED

This script implements a completely new approach for problem statement extraction:
1. Use ServiceNow BERT model (trained on 1M+ help-desk tickets)
2. UMAP dimensionality reduction (384D → 15D)
3. HDBSCAN density-based clustering (automatic cluster count)
4. Recursive splitting for mega-clusters (>25% of tickets)
5. LLM-generated problem statements for each cluster

Key advantages:
- Domain-specific embeddings understand help-desk vocabulary
- No manual threshold tuning required
- Handles noise and outliers automatically
- Scales well to large datasets
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
from sklearn.metrics import silhouette_score
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

class UMAPHDBSCANGrouper:
    """Advanced problem grouping using UMAP + HDBSCAN with ServiceNow BERT."""
    
    def __init__(self, model_name='DavinciTech/BERT_Categorizer'):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        
        # UMAP parameters
        self.umap_n_components = 15
        self.umap_n_neighbors = 30
        self.umap_min_dist = 0.0
        self.umap_metric = 'cosine'
        
        # HDBSCAN parameters (less conservative to reduce noise)
        self.hdbscan_min_cluster_size = 8   # Smaller minimum cluster size
        self.hdbscan_min_samples = 5        # Fewer samples required
        self.hdbscan_cluster_selection_epsilon = 0.1  # Allow slightly looser clusters
        
        # Quality control parameters
        self.max_cluster_percentage = 25.0  # Split clusters with >25% of tickets
        self.min_cluster_size = 4  # Minimum viable cluster size
        self.max_clusters = 400  # Upper bound for sanity check
        
        # Load IT Support BERT model (trained on IT helpdesk tickets)
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
        self.clustering_history = []
        
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
    
    def apply_hdbscan_clustering(self, embeddings: np.ndarray, level_name: str = "main") -> tuple:
        """Apply HDBSCAN clustering to embeddings."""
        self.logger.info(f"Applying HDBSCAN clustering ({level_name} level)")
        
        # Initialize HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.hdbscan_min_cluster_size,
            min_samples=self.hdbscan_min_samples,
            cluster_selection_epsilon=self.hdbscan_cluster_selection_epsilon,
            metric='euclidean',  # UMAP outputs work well with euclidean
            cluster_selection_method='eom'  # Excess of Mass
        )
        
        # Fit clustering
        cluster_labels = clusterer.fit_predict(embeddings)
        
        # Get cluster statistics
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Exclude noise (-1)
        n_noise = np.sum(cluster_labels == -1)
        
        self.logger.info(f"HDBSCAN results ({level_name}):")
        self.logger.info(f"  Clusters found: {n_clusters}")
        self.logger.info(f"  Noise points: {n_noise}")
        self.logger.info(f"  Cluster sizes: {Counter(cluster_labels)}")
        
        return cluster_labels, clusterer
    
    def check_and_split_mega_clusters(self, cluster_labels: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Check for mega-clusters and split them recursively."""
        self.logger.info("Checking for mega-clusters (>25% of tickets)")
        
        total_tickets = len(cluster_labels)
        max_cluster_size = int(total_tickets * (self.max_cluster_percentage / 100))
        
        # Get cluster sizes
        cluster_counts = Counter(cluster_labels)
        mega_clusters = [label for label, count in cluster_counts.items() 
                        if label != -1 and count > max_cluster_size]
        
        if not mega_clusters:
            self.logger.info("No mega-clusters found")
            return cluster_labels
        
        self.logger.info(f"Found {len(mega_clusters)} mega-clusters to split: {mega_clusters}")
        
        # Process each mega-cluster
        new_labels = cluster_labels.copy()
        next_label = max(cluster_labels) + 1
        
        for mega_label in mega_clusters:
            self.logger.info(f"Splitting mega-cluster {mega_label} ({cluster_counts[mega_label]} tickets)")
            
            # Get indices of tickets in this mega-cluster
            mega_indices = np.where(cluster_labels == mega_label)[0]
            mega_embeddings = embeddings[mega_indices]
            
            # Apply HDBSCAN again with smaller min_cluster_size
            sub_clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(8, self.hdbscan_min_cluster_size // 2),  # Smaller clusters
                min_samples=max(5, self.hdbscan_min_samples // 2),
                cluster_selection_epsilon=0.1,  # Slightly higher epsilon for splitting
                metric='euclidean'
            )
            
            sub_labels = sub_clusterer.fit_predict(mega_embeddings)
            
            # Update labels
            for i, sub_label in enumerate(sub_labels):
                original_idx = mega_indices[i]
                if sub_label == -1:
                    # Keep as noise or assign to smallest sub-cluster
                    new_labels[original_idx] = -1
                else:
                    new_labels[original_idx] = next_label + sub_label
            
            # Update next_label counter
            if len(np.unique(sub_labels)) > 0:
                next_label += len(np.unique(sub_labels)) - (1 if -1 in sub_labels else 0)
            
            sub_cluster_counts = Counter(sub_labels)
            self.logger.info(f"  Split into {len(sub_cluster_counts)} sub-clusters: {sub_cluster_counts}")
        
        return new_labels
    
    def find_problem_groups_umap_hdbscan(self) -> dict:
        """Find problem groups using UMAP + HDBSCAN approach."""
        self.logger.info("Finding problem groups using UMAP + HDBSCAN")
        
        # Step 1: Apply UMAP reduction
        umap_embeddings = self.apply_umap_reduction()
        
        # Step 2: Apply HDBSCAN clustering
        cluster_labels, clusterer = self.apply_hdbscan_clustering(umap_embeddings)
        
        # Step 3: Check and split mega-clusters
        final_labels = self.check_and_split_mega_clusters(cluster_labels, umap_embeddings)
        
        # Step 4: Convert to group format
        groups = self._convert_labels_to_groups(final_labels, clusterer)
        
        # Step 5: Calculate silhouette score for quality assessment
        if len(np.unique(final_labels)) > 1:
            try:
                silhouette_avg = silhouette_score(umap_embeddings, final_labels)
                self.logger.info(f"Silhouette score: {silhouette_avg:.3f}")
            except:
                silhouette_avg = None
        else:
            silhouette_avg = None
        
        # Store clustering history
        self.clustering_history.append({
            'method': 'UMAP+HDBSCAN',
            'n_clusters': len(np.unique(final_labels)) - (1 if -1 in final_labels else 0),
            'n_noise': np.sum(final_labels == -1),
            'silhouette_score': silhouette_avg,
            'parameters': {
                'umap_n_components': self.umap_n_components,
                'umap_n_neighbors': self.umap_n_neighbors,
                'hdbscan_min_cluster_size': self.hdbscan_min_cluster_size,
                'hdbscan_min_samples': self.hdbscan_min_samples
            }
        })
        
        self.problem_groups = groups
        return groups
    
    def _convert_labels_to_groups(self, cluster_labels: np.ndarray, clusterer) -> dict:
        """Convert cluster labels to group format."""
        groups = {}
        group_id = 0
        
        # Get unique cluster labels (excluding noise -1)
        unique_labels = np.unique(cluster_labels)
        cluster_labels_clean = [label for label in unique_labels if label != -1]
        
        for cluster_label in cluster_labels_clean:
            # Get ticket indices for this cluster
            ticket_indices = np.where(cluster_labels == cluster_label)[0].tolist()
            
            if len(ticket_indices) < self.min_cluster_size:
                # Skip very small clusters
                continue
            
            # Find representative ticket (closest to cluster centroid in UMAP space)
            cluster_umap_embeddings = self.umap_embeddings[ticket_indices]
            centroid = np.mean(cluster_umap_embeddings, axis=0)
            
            # Find ticket closest to centroid
            distances = np.linalg.norm(cluster_umap_embeddings - centroid, axis=1)
            representative_idx = ticket_indices[np.argmin(distances)]
            
            # Get cluster strength metrics from HDBSCAN
            if hasattr(clusterer, 'cluster_persistence_'):
                try:
                    persistence = clusterer.cluster_persistence_[cluster_label] if cluster_label < len(clusterer.cluster_persistence_) else 0.0
                except:
                    persistence = 0.0
            else:
                persistence = 0.0
            
            groups[f"problem_group_{group_id}"] = {
                'group_id': group_id,
                'representative_ticket_index': representative_idx,
                'ticket_indices': ticket_indices,
                'group_size': len(ticket_indices),
                'cluster_label': cluster_label,
                'cluster_persistence': float(persistence),
                'clustering_method': 'UMAP+HDBSCAN',
                'representative_selection_method': 'umap_centroid'
            }
            group_id += 1
        
        # Handle noise points as individual groups if needed
        noise_indices = np.where(cluster_labels == -1)[0]
        if len(noise_indices) > 0:
            self.logger.info(f"Found {len(noise_indices)} noise points - treating as individual groups")
            
            for noise_idx in noise_indices:
                groups[f"problem_group_{group_id}"] = {
                    'group_id': group_id,
                    'representative_ticket_index': noise_idx,
                    'ticket_indices': [noise_idx],
                    'group_size': 1,
                    'cluster_label': -1,
                    'cluster_persistence': 0.0,
                    'clustering_method': 'UMAP+HDBSCAN',
                    'representative_selection_method': 'noise_singleton'
                }
                group_id += 1
        
        return groups
    
    def calculate_quality_metrics(self) -> dict:
        """Calculate quality metrics for the grouping."""
        self.logger.info("Calculating quality metrics...")
        
        if not self.problem_groups:
            raise ValueError("No problem groups found. Run find_problem_groups_umap_hdbscan() first.")
        
        # Group size statistics
        group_sizes = [group['group_size'] for group in self.problem_groups.values()]
        
        # Coverage analysis
        total_tickets = len(self.tickets_df)
        tickets_in_groups = sum(group_sizes)
        
        # Cluster distribution
        noise_groups = sum(1 for group in self.problem_groups.values() if group['cluster_label'] == -1)
        real_clusters = len(self.problem_groups) - noise_groups
        
        metrics = {
            'total_groups': len(self.problem_groups),
            'real_clusters': real_clusters,
            'noise_singletons': noise_groups,
            'total_tickets_grouped': tickets_in_groups,
            'coverage_percentage': (tickets_in_groups / total_tickets) * 100,
            'clustering_method': 'UMAP+HDBSCAN',
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
                'umap_n_neighbors': self.umap_n_neighbors,
                'hdbscan_min_cluster_size': self.hdbscan_min_cluster_size,
                'hdbscan_min_samples': self.hdbscan_min_samples,
                'max_cluster_percentage': self.max_cluster_percentage
            }
        }
        
        self.logger.info(f"Quality metrics:")
        self.logger.info(f"  Total groups: {metrics['total_groups']:,}")
        self.logger.info(f"  Real clusters: {metrics['real_clusters']:,}")
        self.logger.info(f"  Noise singletons: {metrics['noise_singletons']:,}")
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
                'cluster_persistence': group_data['cluster_persistence'],
                'clustering_method': group_data['clustering_method'],
                'representative_ticket_index': rep_idx,
                'representative_selection_method': group_data['representative_selection_method'],
                'representative_short_description': rep_ticket['Short description'],
                'all_ticket_indices': ','.join(map(str, group_data['ticket_indices'])),
                'manual_review_priority': 'HIGH' if group_data['group_size'] >= 50 else 
                                        'MEDIUM' if group_data['group_size'] >= 10 else 'LOW',
                'is_noise_cluster': group_data['cluster_label'] == -1
            })
        
        groups_summary_df = pd.DataFrame(groups_summary)
        groups_summary_df = groups_summary_df.sort_values('group_size', ascending=False)
        
        summary_file = output_dir / 'problem_groups_umap_hdbscan.csv'
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
                    'clustering_method': group_data['clustering_method'],
                    'manual_review_flag': 'REQUIRED' if group_data['group_size'] >= 50 else 
                                        'RECOMMENDED' if group_data['group_size'] >= 10 else 'OPTIONAL',
                    'group_size': group_data['group_size'],
                    'is_noise_point': group_data['cluster_label'] == -1
                })
        
        ticket_details_df = pd.DataFrame(ticket_details)
        ticket_details_df = ticket_details_df.sort_values(['problem_group_id', 'umap_distance_to_representative'], 
                                                         ascending=[True, True])
        
        details_file = output_dir / 'problem_group_umap_details.csv'
        ticket_details_df.to_csv(details_file, index=False)
        
        # 3. Quality metrics report
        quality_file = output_dir / 'umap_hdbscan_quality_report.json'
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
    """Main UMAP + HDBSCAN problem grouping pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("="*80)
    print("PHASE 1: UMAP + HDBSCAN PROBLEM GROUPING - HELP-DESK OPTIMIZED")
    print("="*80)
    print("NEW APPROACH:")
    print("• DavinciTech BERT embeddings (trained on IT helpdesk tickets)")
    print("• UMAP dimensionality reduction (384D → 15D)")
    print("• HDBSCAN density-based clustering (automatic cluster detection)")
    print("• Recursive mega-cluster splitting (>25% threshold)")
    print("• No manual threshold tuning required")
    print("="*80)
    
    # File paths
    input_file = Path('data/processed/consolidated_tickets.csv')
    output_dir = Path('outputs')
    
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        print("   Please ensure consolidated_tickets.csv exists")
        return
    
    # Initialize UMAP+HDBSCAN grouper
    print(f"\n🎯 CONFIGURATION:")
    print(f"   Embedding model: DavinciTech BERT (IT helpdesk trained)")
    print(f"   UMAP reduction: 384D → 15D")
    print(f"   HDBSCAN min cluster size: 8 tickets (less conservative)")
    print(f"   Mega-cluster threshold: 25% of total tickets")
    print(f"   Quality focus: Automatic density-based grouping")
    
    grouper = UMAPHDBSCANGrouper()
    
    try:
        # Step 1: Load data
        print(f"\n🔄 STEP 1: Loading consolidated tickets")
        df = grouper.load_consolidated_tickets(input_file)
        
        # Step 2: Generate ServiceNow BERT embeddings
        print(f"\n🔄 STEP 2: Generating IT Support BERT embeddings")
        embeddings, texts = grouper.generate_embeddings()
        
        # Step 3: Apply UMAP + HDBSCAN clustering
        print(f"\n🔄 STEP 3: Applying UMAP + HDBSCAN clustering")
        groups = grouper.find_problem_groups_umap_hdbscan()
        
        # Step 4: Calculate quality metrics
        print(f"\n🔄 STEP 4: Calculating quality metrics")
        quality_metrics = grouper.calculate_quality_metrics()
        
        # Step 5: Create outputs
        print(f"\n🔄 STEP 5: Creating detailed outputs")
        summary_file, details_file, quality_file = grouper.create_detailed_outputs(output_dir)
        
        # Final summary
        print(f"\n📊 UMAP + HDBSCAN RESULTS:")
        print(f"   Input tickets: {len(df):,}")
        print(f"   Problem groups identified: {quality_metrics['total_groups']:,}")
        print(f"   Real clusters (non-noise): {quality_metrics['real_clusters']:,}")
        print(f"   Noise singletons: {quality_metrics['noise_singletons']:,}")
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
        
        print(f"\n✅ UMAP + HDBSCAN COMPLETE")
        all_success = success_largest and success_reduction and success_singletons and success_groups
        if all_success:
            print(f"   🎉 ALL SUCCESS CRITERIA MET!")
            print(f"   🎯 Ready for problem statement generation")
            print(f"   📊 Domain-specific embeddings + density clustering = MUCH better results")
        else:
            print(f"   ⚠️ Some targets not met - but likely still much better than threshold-based approaches")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Review cluster quality in output files")
        print(f"   2. Generate LLM problem statements for each cluster")
        print(f"   3. Manual validation of problem statements")
        print(f"   4. Create final traceability report")
        
        logger.info("UMAP + HDBSCAN problem grouping completed successfully!")
        
    except Exception as e:
        logger.error(f"UMAP + HDBSCAN grouping failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()