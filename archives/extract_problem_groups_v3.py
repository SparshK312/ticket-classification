#!/usr/bin/env python3
"""
PHASE 1: SEMANTIC PROBLEM GROUPING V3 - TICKET-LEVEL CRITERIA

This script implements the refined approach with proper success criteria:

SUCCESS CRITERIA (ticket-level, not group-level):
- Largest cluster share: ≤25% of tickets
- Singleton tickets: ≤30% of tickets  
- Total clusters: 50-400
- Reduction ratio: 4-20×

FIXES:
- Proper threshold search (0.65 → 0.45)
- Full dataset evaluation (not samples)
- Top-k union-find (prevent O(n²) blowup)
- Recursive split for mega-clusters
- HDBSCAN fallback if threshold search fails
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
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.preprocessing import normalize
import hdbscan

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

class UnionFindTopK:
    """Union-Find with top-k neighbors to prevent O(n²) blowup."""
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
    
    def get_components(self):
        components = defaultdict(list)
        for i in range(len(self.parent)):
            components[self.find(i)].append(i)
        return list(components.values())

class ProblemGroupingV3:
    """Refined semantic problem grouping with ticket-level success criteria."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        
        # Success criteria (ticket-level)
        self.max_largest_cluster_pct = 25.0  # ≤25% of tickets in largest cluster
        self.max_singleton_tickets_pct = 30.0  # ≤30% of tickets in singletons
        self.min_clusters = 50
        self.max_clusters = 400
        self.min_reduction_ratio = 4.0
        self.max_reduction_ratio = 20.0
        
        self.top_k_neighbors = 10  # Mutual-k strategy (reduced from 20)
        self.max_combined_pct = 45.0  # largest_cluster_pct + singleton_pct ≤ 45%
        
        # Load sentence transformer model
        self.logger.info(f"Loading sentence transformer model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.logger.info("Model loaded successfully")
        
        # Results storage
        self.embeddings = None
        self.tickets_df = None
        self.problem_groups = {}
        self.quality_metrics = {}
        self.threshold_search_results = []
        self.optimal_threshold = None
        self.final_method = None
        
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
        
        # Strategy: Emphasize short description but include full context
        if len(description) > 0 and description.lower() != 'nan':
            combined = f"{short_desc}. {description}"
        else:
            combined = short_desc
        
        return combined.strip()
    
    def generate_embeddings(self) -> np.ndarray:
        """Generate semantic embeddings for all tickets."""
        self.logger.info("Preparing text for embedding generation...")
        
        # Prepare texts
        texts = self.tickets_df.apply(self.prepare_text_for_embedding, axis=1).tolist()
        
        self.logger.info(f"Generating embeddings for {len(texts):,} tickets...")
        
        # Generate embeddings with progress bar
        embeddings = self.encoder.encode(
            texts, 
            show_progress_bar=True,
            convert_to_tensor=False,
            batch_size=32
        )
        
        # Normalize embeddings for cosine similarity
        embeddings = normalize(embeddings, norm='l2')
        
        self.logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        self.embeddings = embeddings
        
        return embeddings, texts
    
    def calculate_similarity_distribution(self) -> dict:
        """Calculate comprehensive similarity statistics."""
        self.logger.info("Calculating similarity distribution...")
        
        # Use sample for statistics but full dataset for clustering
        n_tickets = len(self.embeddings)
        sample_size = min(2000, n_tickets)
        
        indices = np.random.choice(n_tickets, sample_size, replace=False)
        sample_embeddings = self.embeddings[indices]
        
        # Calculate pairwise cosine similarities
        similarity_matrix = cosine_similarity(sample_embeddings)
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = np.triu(similarity_matrix, k=1)
        similarities = upper_triangle[upper_triangle > 0]
        
        # Calculate percentiles for threshold guidance
        percentiles = [50, 60, 70, 75, 80, 85, 90, 95, 99]
        percentile_values = [np.percentile(similarities, p) for p in percentiles]
        
        stats = {
            'mean_similarity': float(np.mean(similarities)),
            'median_similarity': float(np.median(similarities)),
            'std_similarity': float(np.std(similarities)),
            'percentiles': dict(zip(percentiles, percentile_values)),
            'sample_size': sample_size,
            'suggested_search_range': {
                'start': 0.70,  # Fixed high start to avoid early percolation
                'end': 0.55    # Skip anything below ~90-95th percentile range
            }
        }
        
        self.logger.info(f"Similarity distribution:")
        self.logger.info(f"  Mean: {stats['mean_similarity']:.3f}")
        self.logger.info(f"  80th percentile: {stats['percentiles'][80]:.3f}")
        self.logger.info(f"  85th percentile: {stats['percentiles'][85]:.3f}")
        self.logger.info(f"  Suggested search range: {stats['suggested_search_range']['start']:.2f} → {stats['suggested_search_range']['end']:.2f}")
        
        return stats
    
    def evaluate_clustering_full_dataset(self, threshold: float) -> dict:
        """Evaluate clustering with mutual-k strategy and early abort."""
        n_tickets = len(self.embeddings)
        max_cluster_size = int(0.25 * n_tickets)  # 25% threshold for early abort
        
        # Step 1: Find top-k neighbors for each ticket
        self.logger.info(f"    Finding mutual top-{self.top_k_neighbors} neighbors...")
        top_k_neighbors = {}
        
        chunk_size = 100
        for i in range(0, n_tickets, chunk_size):
            end_i = min(i + chunk_size, n_tickets)
            
            for ticket_idx in range(i, end_i):
                ticket_embedding = self.embeddings[ticket_idx].reshape(1, -1)
                similarities = cosine_similarity(ticket_embedding, self.embeddings)[0]
                
                # Find top-k most similar tickets above threshold (excluding self)
                above_threshold = np.where(similarities >= threshold)[0]
                other_tickets = above_threshold[above_threshold != ticket_idx]
                
                if len(other_tickets) > 0:
                    # Get top-k most similar
                    top_k_indices = other_tickets[np.argsort(similarities[other_tickets])[-self.top_k_neighbors:]]
                    top_k_neighbors[ticket_idx] = set(top_k_indices)
                else:
                    top_k_neighbors[ticket_idx] = set()
        
        # Step 2: Mutual-k union-find with early abort
        self.logger.info(f"    Building mutual-k graph with early abort...")
        uf = UnionFindTopK(n_tickets)
        total_edges = 0
        
        for ticket_i in range(n_tickets):
            neighbors_i = top_k_neighbors.get(ticket_i, set())
            
            for ticket_j in neighbors_i:
                neighbors_j = top_k_neighbors.get(ticket_j, set())
                
                # Mutual-k: only connect if both are in each other's top-k
                if ticket_i in neighbors_j:
                    uf.union(ticket_i, ticket_j)
                    total_edges += 1
            
            # Early abort check: if any component already > 25%, stop
            if ticket_i % 200 == 0:  # Check every 200 tickets
                temp_components = uf.get_components()
                max_component_size = max(len(comp) for comp in temp_components)
                if max_component_size > max_cluster_size:
                    self.logger.info(f"    Early abort: component size {max_component_size} > {max_cluster_size}")
                    return {
                        'threshold': threshold,
                        'early_abort': True,
                        'largest_cluster_pct': (max_component_size / n_tickets) * 100,
                        'meets_all_criteria': False
                    }
        
        # Get final components
        components = uf.get_components()
        
        # Calculate ticket-level metrics
        cluster_sizes = [len(comp) for comp in components]
        largest_cluster_size = max(cluster_sizes)
        singleton_tickets = sum(1 for size in cluster_sizes if size == 1)
        
        # Ticket-level percentages
        largest_cluster_pct = (largest_cluster_size / n_tickets) * 100
        singleton_tickets_pct = (singleton_tickets / n_tickets) * 100
        reduction_ratio = n_tickets / len(components)
        
        # Success criteria check (including tightened combined criteria)
        meets_largest_cluster = largest_cluster_pct <= self.max_largest_cluster_pct
        meets_singleton_tickets = singleton_tickets_pct <= self.max_singleton_tickets_pct
        meets_cluster_count = self.min_clusters <= len(components) <= self.max_clusters
        meets_reduction_ratio = self.min_reduction_ratio <= reduction_ratio <= self.max_reduction_ratio
        meets_combined_pct = (largest_cluster_pct + singleton_tickets_pct) <= self.max_combined_pct
        
        meets_all_criteria = (meets_largest_cluster and meets_singleton_tickets and 
                             meets_cluster_count and meets_reduction_ratio and meets_combined_pct)
        
        result = {
            'threshold': threshold,
            'total_clusters': len(components),
            'largest_cluster_size': largest_cluster_size,
            'largest_cluster_pct': largest_cluster_pct,
            'singleton_tickets': singleton_tickets,
            'singleton_tickets_pct': singleton_tickets_pct,
            'combined_pct': largest_cluster_pct + singleton_tickets_pct,
            'reduction_ratio': reduction_ratio,
            'total_edges': total_edges,
            'meets_largest_cluster': meets_largest_cluster,
            'meets_singleton_tickets': meets_singleton_tickets,
            'meets_cluster_count': meets_cluster_count,
            'meets_reduction_ratio': meets_reduction_ratio,
            'meets_combined_pct': meets_combined_pct,
            'meets_all_criteria': meets_all_criteria,
            'early_abort': False,
            'components': components if meets_all_criteria else None  # Only store if successful
        }
        
        return result
    
    def threshold_search_with_criteria(self) -> float:
        """Search for optimal threshold using ticket-level success criteria."""
        self.logger.info("🔍 THRESHOLD SEARCH WITH TICKET-LEVEL CRITERIA")
        self.logger.info(f"   Success criteria:")
        self.logger.info(f"   • Largest cluster: ≤{self.max_largest_cluster_pct}% of tickets")
        self.logger.info(f"   • Singleton tickets: ≤{self.max_singleton_tickets_pct}% of tickets")
        self.logger.info(f"   • Total clusters: {self.min_clusters}-{self.max_clusters}")
        self.logger.info(f"   • Reduction ratio: {self.min_reduction_ratio}-{self.max_reduction_ratio}×")
        
        # Get search range
        similarity_stats = self.calculate_similarity_distribution()
        start_threshold = similarity_stats['suggested_search_range']['start']
        end_threshold = similarity_stats['suggested_search_range']['end']
        
        # Search from high to low threshold in steps of 0.02
        thresholds_to_test = np.arange(start_threshold, end_threshold - 0.02, -0.02)
        
        self.logger.info(f"Testing thresholds from {start_threshold:.2f} down to {end_threshold:.2f}")
        
        best_result = None
        
        for threshold in thresholds_to_test:
            self.logger.info(f"  Testing threshold: {threshold:.2f}")
            
            # Evaluate on full dataset
            result = self.evaluate_clustering_full_dataset(threshold)
            self.threshold_search_results.append(result)
            
            # Skip if early abort happened
            if result.get('early_abort', False):
                self.logger.info(f"    ❌ Early abort - mega-cluster detected")
                continue
            
            self.logger.info(f"    Clusters: {result['total_clusters']:,}")
            self.logger.info(f"    Largest cluster: {result['largest_cluster_pct']:.1f}% of tickets")
            self.logger.info(f"    Singleton tickets: {result['singleton_tickets_pct']:.1f}%")
            self.logger.info(f"    Combined %: {result['combined_pct']:.1f}% (≤{self.max_combined_pct}%)")
            self.logger.info(f"    Reduction ratio: {result['reduction_ratio']:.1f}×")
            
            # Check success criteria
            if result['meets_all_criteria']:
                self.logger.info(f"    ✅ MEETS ALL CRITERIA! Selecting threshold {threshold:.2f}")
                self.optimal_threshold = threshold
                return threshold, result['components']
            else:
                # Track best partial result
                if best_result is None:
                    best_result = result
                else:
                    # Prefer result that meets more criteria
                    current_score = sum([
                        result['meets_largest_cluster'],
                        result['meets_singleton_tickets'], 
                        result['meets_cluster_count'],
                        result['meets_reduction_ratio']
                    ])
                    best_score = sum([
                        best_result['meets_largest_cluster'],
                        best_result['meets_singleton_tickets'],
                        best_result['meets_cluster_count'],
                        best_result['meets_reduction_ratio']
                    ])
                    if current_score > best_score:
                        best_result = result
        
        # No threshold met all criteria
        if best_result:
            self.logger.info(f"❌ No threshold met all criteria")
            self.logger.info(f"   Best partial result: {best_result['threshold']:.2f}")
            self.logger.info(f"   Will try HDBSCAN fallback...")
            return None, None
        else:
            raise RuntimeError("Threshold search failed completely")
    
    def recursive_split_mega_clusters(self, components: list) -> list:
        """Recursively split any cluster > 25% of tickets."""
        self.logger.info("🔄 Checking for mega-clusters requiring recursive split")
        
        n_tickets = len(self.embeddings)
        max_cluster_size = int(0.25 * n_tickets)  # 25% threshold
        
        final_components = []
        split_count = 0
        
        for component in components:
            if len(component) > max_cluster_size:
                self.logger.info(f"   Splitting mega-cluster of {len(component)} tickets (>{max_cluster_size})")
                
                # Re-run threshold search within this cluster
                cluster_embeddings = self.embeddings[component]
                
                # Try higher thresholds within cluster
                for split_threshold in [0.65, 0.60, 0.55, 0.50, 0.45]:
                    sub_uf = UnionFindTopK(len(component))
                    
                    for i, ticket_i in enumerate(component):
                        for j, ticket_j in enumerate(component):
                            if i != j:
                                similarity = cosine_similarity(
                                    cluster_embeddings[i].reshape(1, -1),
                                    cluster_embeddings[j].reshape(1, -1)
                                )[0][0]
                                
                                if similarity >= split_threshold:
                                    sub_uf.union(i, j)
                    
                    sub_components = sub_uf.get_components()
                    
                    # Check if split was successful
                    max_sub_size = max(len(sub_comp) for sub_comp in sub_components)
                    if max_sub_size <= max_cluster_size and len(sub_components) > 1:
                        self.logger.info(f"   Successfully split into {len(sub_components)} sub-clusters")
                        # Convert back to original indices
                        for sub_comp in sub_components:
                            final_components.append([component[i] for i in sub_comp])
                        split_count += 1
                        break
                else:
                    # Couldn't split effectively, keep as is
                    self.logger.info(f"   Could not split effectively, keeping as single cluster")
                    final_components.append(component)
            else:
                final_components.append(component)
        
        if split_count > 0:
            self.logger.info(f"✅ Split {split_count} mega-clusters")
        else:
            self.logger.info(f"✅ No mega-clusters required splitting")
        
        return final_components
    
    def hdbscan_fallback(self) -> list:
        """HDBSCAN fallback with tuned parameters for bigger cores."""
        self.logger.info("🔄 HDBSCAN FALLBACK - threshold search failed")
        
        # Try tuned parameters for bigger cores (as suggested)
        hdbscan_configs = [
            {'min_cluster_size': 40, 'min_samples': 10},
            {'min_cluster_size': 50, 'min_samples': 15},
            {'min_cluster_size': 30, 'min_samples': 8},
            {'min_cluster_size': 25, 'min_samples': 6}
        ]
        
        for config in hdbscan_configs:
            min_cluster_size = config['min_cluster_size']
            min_samples = config['min_samples']
            
            self.logger.info(f"   Trying HDBSCAN: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean'
            )
            
            labels = clusterer.fit_predict(self.embeddings)
            
            # Convert to components format
            components = defaultdict(list)
            for i, label in enumerate(labels):
                components[label].append(i)
            
            components_list = list(components.values())
            
            # Evaluate with success criteria
            cluster_sizes = [len(comp) for comp in components_list]
            largest_cluster_size = max(cluster_sizes)
            singleton_tickets = sum(1 for size in cluster_sizes if size == 1)
            
            n_tickets = len(self.embeddings)
            largest_cluster_pct = (largest_cluster_size / n_tickets) * 100
            singleton_tickets_pct = (singleton_tickets / n_tickets) * 100
            combined_pct = largest_cluster_pct + singleton_tickets_pct
            reduction_ratio = n_tickets / len(components_list)
            
            self.logger.info(f"     Clusters: {len(components_list)}")
            self.logger.info(f"     Largest cluster: {largest_cluster_pct:.1f}% of tickets")
            self.logger.info(f"     Singleton tickets: {singleton_tickets_pct:.1f}%")
            self.logger.info(f"     Combined %: {combined_pct:.1f}% (≤{self.max_combined_pct}%)")
            self.logger.info(f"     Reduction ratio: {reduction_ratio:.1f}×")
            
            # Check success criteria (including combined percentage)
            meets_criteria = (
                largest_cluster_pct <= self.max_largest_cluster_pct and
                singleton_tickets_pct <= self.max_singleton_tickets_pct and
                combined_pct <= self.max_combined_pct and
                self.min_clusters <= len(components_list) <= self.max_clusters and
                self.min_reduction_ratio <= reduction_ratio <= self.max_reduction_ratio
            )
            
            if meets_criteria:
                self.logger.info(f"     ✅ HDBSCAN SUCCESS! Using config: {config}")
                self.final_method = f'HDBSCAN_min_cluster_size_{min_cluster_size}_min_samples_{min_samples}'
                
                # Apply recursive split if any cluster still > 25%
                return self.recursive_split_hdbscan(components_list)
        
        # HDBSCAN also failed
        self.logger.error("❌ HDBSCAN fallback also failed to meet criteria")
        raise RuntimeError("Both threshold search and HDBSCAN failed to meet success criteria")
    
    def recursive_split_hdbscan(self, components: list) -> list:
        """Recursively split mega-clusters after HDBSCAN."""
        self.logger.info("🔄 Checking HDBSCAN results for mega-clusters")
        
        n_tickets = len(self.embeddings)
        max_cluster_size = int(0.25 * n_tickets)
        
        final_components = []
        split_count = 0
        
        for component in components:
            if len(component) > max_cluster_size:
                self.logger.info(f"   Splitting HDBSCAN mega-cluster of {len(component)} tickets")
                
                # Re-run HDBSCAN within this cluster with halved min_cluster_size
                cluster_embeddings = self.embeddings[component]
                
                # Try progressively smaller min_cluster_size
                for min_cluster_size in [20, 15, 10, 8]:
                    sub_clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=max(3, min_cluster_size // 3),
                        metric='euclidean'
                    )
                    
                    sub_labels = sub_clusterer.fit_predict(cluster_embeddings)
                    
                    # Convert to sub-components
                    sub_components = defaultdict(list)
                    for i, label in enumerate(sub_labels):
                        sub_components[label].append(component[i])  # Map back to original indices
                    
                    sub_components_list = list(sub_components.values())
                    max_sub_size = max(len(sub_comp) for sub_comp in sub_components_list)
                    
                    if max_sub_size <= max_cluster_size and len(sub_components_list) > 1:
                        self.logger.info(f"   Successfully split into {len(sub_components_list)} sub-clusters")
                        final_components.extend(sub_components_list)
                        split_count += 1
                        break
                else:
                    # Couldn't split effectively
                    self.logger.info(f"   Could not split effectively, keeping as single cluster")
                    final_components.append(component)
            else:
                final_components.append(component)
        
        if split_count > 0:
            self.logger.info(f"✅ Split {split_count} HDBSCAN mega-clusters")
        
        return final_components
    
    def find_problem_groups(self) -> dict:
        """Main clustering pipeline with fallbacks."""
        # Try threshold search first
        threshold, components = self.threshold_search_with_criteria()
        
        if threshold is not None and components is not None:
            self.logger.info(f"✅ Threshold search successful at {threshold:.2f}")
            self.optimal_threshold = threshold
            self.final_method = f'threshold_search_{threshold:.2f}'
            
            # Apply recursive split for any remaining mega-clusters
            components = self.recursive_split_mega_clusters(components)
            
        else:
            # Fallback to HDBSCAN
            components = self.hdbscan_fallback()
        
        # Convert to groups format
        groups = {}
        for group_id, ticket_indices in enumerate(components):
            representative_idx = self._find_centroid_representative(ticket_indices)
            
            # Calculate group similarity metrics
            if len(ticket_indices) > 1:
                group_embeddings = self.embeddings[ticket_indices]
                similarity_matrix = cosine_similarity(group_embeddings)
                upper_triangle = np.triu(similarity_matrix, k=1)
                group_similarities = upper_triangle[upper_triangle > 0]
                
                min_sim = float(np.min(group_similarities))
                max_sim = float(np.max(group_similarities))
                mean_sim = float(np.mean(group_similarities))
            else:
                min_sim = max_sim = mean_sim = 1.0
            
            groups[f"problem_group_{group_id}"] = {
                'group_id': group_id,
                'representative_ticket_index': representative_idx,
                'ticket_indices': ticket_indices,
                'group_size': len(ticket_indices),
                'similarity_threshold_used': self.optimal_threshold or 'HDBSCAN',
                'min_similarity_in_group': min_sim,
                'max_similarity_in_group': max_sim,
                'mean_similarity_in_group': mean_sim,
                'clustering_method': self.final_method
            }
        
        self.problem_groups = groups
        return groups
    
    def _find_centroid_representative(self, ticket_indices: list) -> int:
        """Find the centroid ticket (highest average similarity to others in group)."""
        if len(ticket_indices) == 1:
            return ticket_indices[0]
        
        # Get embeddings for tickets in this group  
        group_embeddings = self.embeddings[ticket_indices]
        
        # Calculate pairwise similarities within group
        similarity_matrix = cosine_similarity(group_embeddings)
        
        # Find ticket with highest average similarity to others
        avg_similarities = []
        for i in range(len(ticket_indices)):
            # Get similarities to all other tickets in group (excluding self)
            similarities_to_others = np.concatenate([
                similarity_matrix[i, :i],
                similarity_matrix[i, i+1:]
            ])
            avg_similarity = np.mean(similarities_to_others) if len(similarities_to_others) > 0 else 0
            avg_similarities.append(avg_similarity)
        
        # Return ticket index with highest average similarity
        centroid_idx = np.argmax(avg_similarities)
        return ticket_indices[centroid_idx]
    
    def calculate_final_metrics(self) -> dict:
        """Calculate final success metrics."""
        if not self.problem_groups:
            raise ValueError("No problem groups found.")
        
        n_tickets = len(self.tickets_df)
        group_sizes = [group['group_size'] for group in self.problem_groups.values()]
        
        largest_cluster_size = max(group_sizes)
        singleton_tickets = sum(1 for size in group_sizes if size == 1)
        
        largest_cluster_pct = (largest_cluster_size / n_tickets) * 100
        singleton_tickets_pct = (singleton_tickets / n_tickets) * 100
        reduction_ratio = n_tickets / len(self.problem_groups)
        
        # Check final success criteria
        meets_largest_cluster = largest_cluster_pct <= self.max_largest_cluster_pct
        meets_singleton_tickets = singleton_tickets_pct <= self.max_singleton_tickets_pct
        meets_cluster_count = self.min_clusters <= len(self.problem_groups) <= self.max_clusters
        meets_reduction_ratio = self.min_reduction_ratio <= reduction_ratio <= self.max_reduction_ratio
        
        metrics = {
            'total_clusters': len(self.problem_groups),
            'total_tickets': n_tickets,
            'largest_cluster_size': largest_cluster_size,
            'largest_cluster_pct': largest_cluster_pct,
            'singleton_tickets': singleton_tickets,
            'singleton_tickets_pct': singleton_tickets_pct,
            'reduction_ratio': reduction_ratio,
            'clustering_method': self.final_method,
            'threshold_used': self.optimal_threshold,
            'success_criteria': {
                'largest_cluster_pass': meets_largest_cluster,
                'singleton_tickets_pass': meets_singleton_tickets,
                'cluster_count_pass': meets_cluster_count,
                'reduction_ratio_pass': meets_reduction_ratio,
                'combined_pct_pass': (largest_cluster_pct + singleton_tickets_pct) <= self.max_combined_pct,
                'all_criteria_pass': all([meets_largest_cluster, meets_singleton_tickets, 
                                        meets_cluster_count, meets_reduction_ratio,
                                        (largest_cluster_pct + singleton_tickets_pct) <= self.max_combined_pct])
            },
            'group_size_distribution': {
                'singleton': sum(1 for size in group_sizes if size == 1),
                'small_2_9': sum(1 for size in group_sizes if 2 <= size < 10),
                'medium_10_49': sum(1 for size in group_sizes if 10 <= size < 50),
                'large_50plus': sum(1 for size in group_sizes if size >= 50)
            }
        }
        
        self.quality_metrics = metrics
        return metrics
    
    def create_outputs(self, output_dir: Path) -> tuple:
        """Create output files."""
        output_dir.mkdir(exist_ok=True)
        
        # 1. Groups summary
        groups_summary = []
        for group_name, group_data in self.problem_groups.items():
            rep_idx = group_data['representative_ticket_index']
            rep_ticket = self.tickets_df.loc[rep_idx]
            
            groups_summary.append({
                'problem_group_id': group_data['group_id'],
                'group_size': group_data['group_size'],
                'clustering_method': group_data['clustering_method'],
                'representative_ticket_index': rep_idx,
                'representative_short_description': rep_ticket['Short description'],
                'all_ticket_indices': ','.join(map(str, group_data['ticket_indices'])),
                'mean_similarity_in_group': group_data['mean_similarity_in_group'],
                'manual_review_priority': 'HIGH' if group_data['group_size'] >= 50 else 
                                        'MEDIUM' if group_data['group_size'] >= 10 else 'LOW'
            })
        
        groups_summary_df = pd.DataFrame(groups_summary)
        groups_summary_df = groups_summary_df.sort_values('group_size', ascending=False)
        
        summary_file = output_dir / 'problem_groups_v3_detailed.csv'
        groups_summary_df.to_csv(summary_file, index=False)
        
        # 2. Threshold search results
        if self.threshold_search_results:
            threshold_df = pd.DataFrame(self.threshold_search_results)
            threshold_file = output_dir / 'threshold_search_v3_results.csv'
            threshold_df.to_csv(threshold_file, index=False)
        else:
            threshold_file = None
        
        return summary_file, threshold_file

def main():
    """Main problem grouping pipeline V3."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("="*80)
    print("PHASE 1: SEMANTIC PROBLEM GROUPING V3 - TICKET-LEVEL CRITERIA")
    print("="*80)
    print("SUCCESS CRITERIA (ticket-level):")
    print("• Largest cluster share: ≤25% of tickets")
    print("• Singleton tickets: ≤30% of tickets")
    print("• Total clusters: 50-400")
    print("• Reduction ratio: 4-20×")
    print("="*80)
    
    # File paths
    input_file = Path('data/processed/consolidated_tickets.csv')
    output_dir = Path('outputs')
    
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        return
    
    # Initialize grouper
    grouper = ProblemGroupingV3()
    
    try:
        # Step 1: Load data
        print(f"\n🔄 STEP 1: Loading consolidated tickets")
        df = grouper.load_consolidated_tickets(input_file)
        
        # Step 2: Generate embeddings
        print(f"\n🔄 STEP 2: Generating semantic embeddings")
        embeddings, texts = grouper.generate_embeddings()
        
        # Step 3: Find problem groups (with automatic fallbacks)
        print(f"\n🔄 STEP 3: Finding problem groups (threshold search → HDBSCAN fallback)")
        groups = grouper.find_problem_groups()
        
        # Step 4: Calculate final metrics
        print(f"\n🔄 STEP 4: Calculating final success metrics")
        metrics = grouper.calculate_final_metrics()
        
        # Step 5: Create outputs
        print(f"\n🔄 STEP 5: Creating outputs")
        summary_file, threshold_file = grouper.create_outputs(output_dir)
        
        # Final results
        print(f"\n📊 FINAL RESULTS:")
        print(f"   Input tickets: {metrics['total_tickets']:,}")
        print(f"   Problem groups: {metrics['total_clusters']:,}")
        print(f"   Method used: {metrics['clustering_method']}")
        print(f"   Reduction ratio: {metrics['reduction_ratio']:.1f}:1")
        
        print(f"\n🎯 SUCCESS CRITERIA CHECK:")
        success = metrics['success_criteria']
        print(f"   ✅ Largest cluster ≤25%: {'PASS' if success['largest_cluster_pass'] else 'FAIL'} ({metrics['largest_cluster_pct']:.1f}%)")
        print(f"   ✅ Singleton tickets ≤30%: {'PASS' if success['singleton_tickets_pass'] else 'FAIL'} ({metrics['singleton_tickets_pct']:.1f}%)")
        print(f"   ✅ Combined ≤45%: {'PASS' if success.get('combined_pct_pass', False) else 'FAIL'} ({metrics['largest_cluster_pct'] + metrics['singleton_tickets_pct']:.1f}%)")
        print(f"   ✅ Clusters 50-400: {'PASS' if success['cluster_count_pass'] else 'FAIL'} ({metrics['total_clusters']})")
        print(f"   ✅ Reduction 4-20×: {'PASS' if success['reduction_ratio_pass'] else 'FAIL'} ({metrics['reduction_ratio']:.1f}×)")
        print(f"   🎉 OVERALL: {'SUCCESS' if success['all_criteria_pass'] else 'PARTIAL SUCCESS'}")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"   📄 Summary: {summary_file}")
        if threshold_file:
            print(f"   📄 Threshold search: {threshold_file}")
        
        if success['all_criteria_pass']:
            print(f"\n✅ PHASE 1 V3 COMPLETE - ALL CRITERIA MET!")
            print(f"   🎯 Ready for SME manual validation")
            print(f"   📊 Proper ticket-level success achieved")
        else:
            print(f"\n⚠️ PHASE 1 V3 PARTIAL SUCCESS")
            print(f"   🔍 Review results and consider parameter adjustment")
        
        logger.info("Phase 1 V3: Ticket-level criteria grouping completed!")
        
    except Exception as e:
        logger.error(f"Problem grouping V3 failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()