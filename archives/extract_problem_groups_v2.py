#!/usr/bin/env python3
"""
PHASE 1: SEMANTIC PROBLEM GROUPING V2 - FIXED APPROACH

This script fixes the critical issues with the initial approach:
1. Automatic threshold search to find optimal similarity level
2. Union-find clustering for transitive similarity connections
3. Target ≤40% singletons (realistic for problem types, not duplicates)
4. Proper similarity range for problem grouping vs duplicate detection

Key Changes:
- Sample-guided threshold search (0.45-0.75 range)
- Transitive closure clustering (union-find)
- Problem-type focused (not duplicate-focused)
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
import networkx as nx

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

class UnionFind:
    """Union-Find data structure for transitive clustering."""
    
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

class ProblemGroupingV2:
    """Fixed semantic problem grouping with automatic threshold search and union-find clustering."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', target_singleton_percent=40, max_groups=500):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.target_singleton_percent = target_singleton_percent
        self.max_groups = max_groups
        
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
        
        # Use larger sample for better statistics
        n_tickets = len(self.embeddings)
        sample_size = min(2000, n_tickets)
        
        if n_tickets > sample_size:
            indices = np.random.choice(n_tickets, sample_size, replace=False)
            sample_embeddings = self.embeddings[indices]
            self.logger.info(f"Using sample of {sample_size} tickets for similarity analysis")
        else:
            sample_embeddings = self.embeddings
            indices = np.arange(n_tickets)
        
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
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'percentiles': dict(zip(percentiles, percentile_values)),
            'sample_size': len(indices),
            'total_comparisons': len(similarities),
            'suggested_threshold_range': {
                'problem_grouping_min': float(percentile_values[percentiles.index(60)]),  # 60th percentile
                'problem_grouping_max': float(percentile_values[percentiles.index(80)]),  # 80th percentile
                'duplicate_detection': float(percentile_values[percentiles.index(95)])     # 95th percentile
            }
        }
        
        self.logger.info(f"Similarity distribution:")
        self.logger.info(f"  Mean: {stats['mean_similarity']:.3f}")
        self.logger.info(f"  Median: {stats['median_similarity']:.3f}")
        self.logger.info(f"  60th percentile: {stats['percentiles'][60]:.3f}")
        self.logger.info(f"  80th percentile: {stats['percentiles'][80]:.3f}")
        self.logger.info(f"  95th percentile: {stats['percentiles'][95]:.3f}")
        self.logger.info(f"  Suggested problem grouping range: {stats['suggested_threshold_range']['problem_grouping_min']:.3f} - {stats['suggested_threshold_range']['problem_grouping_max']:.3f}")
        
        return stats
    
    def find_optimal_threshold_automatic(self) -> float:
        """Automatically find optimal threshold using sample-guided search."""
        self.logger.info(f"Searching for optimal threshold (target: ≤{self.target_singleton_percent}% singletons, ≤{self.max_groups} groups)")
        
        # Define search range based on similarity distribution
        similarity_stats = self.calculate_similarity_distribution()
        min_threshold = max(0.45, similarity_stats['suggested_threshold_range']['problem_grouping_min'])
        max_threshold = min(0.75, similarity_stats['suggested_threshold_range']['problem_grouping_max'])
        
        # Search from high to low threshold in steps of 0.02
        thresholds_to_test = np.arange(max_threshold, min_threshold - 0.02, -0.02)
        
        self.logger.info(f"Testing thresholds from {max_threshold:.2f} down to {min_threshold:.2f}")
        
        best_threshold = None
        best_result = None
        
        for threshold in thresholds_to_test:
            self.logger.info(f"  Testing threshold: {threshold:.2f}")
            
            # Test this threshold
            result = self._test_threshold(threshold)
            self.threshold_search_results.append(result)
            
            self.logger.info(f"    Groups: {result['total_groups']:,}, Singletons: {result['singleton_percentage']:.1f}%")
            
            # Check if this meets our criteria
            meets_singleton_target = result['singleton_percentage'] <= self.target_singleton_percent
            meets_group_target = result['total_groups'] <= self.max_groups
            
            if meets_singleton_target and meets_group_target:
                self.logger.info(f"    ✅ Meets criteria! Selecting threshold {threshold:.2f}")
                best_threshold = threshold
                best_result = result
                break
            elif result['singleton_percentage'] <= self.target_singleton_percent:
                # Meets singleton target but too many groups - keep as backup
                if best_result is None or result['total_groups'] < best_result['total_groups']:
                    best_threshold = threshold
                    best_result = result
        
        # If no threshold met criteria, use the best one found
        if best_threshold is None:
            # Find threshold with lowest singleton percentage
            best_result = min(self.threshold_search_results, key=lambda x: x['singleton_percentage'])
            best_threshold = best_result['threshold']
            self.logger.info(f"  No threshold met all criteria, using best: {best_threshold:.2f} ({best_result['singleton_percentage']:.1f}% singletons)")
        
        self.optimal_threshold = best_threshold
        return best_threshold
    
    def _test_threshold(self, threshold: float) -> dict:
        """Test a specific threshold and return metrics."""
        # Quick test using union-find clustering
        n_tickets = len(self.embeddings)
        uf = UnionFind(n_tickets)
        
        # Add edges for similar tickets (sample-based for speed)
        sample_indices = np.random.choice(n_tickets, min(1500, n_tickets), replace=False)
        
        for i, idx_i in enumerate(sample_indices):
            embedding_i = self.embeddings[idx_i].reshape(1, -1)
            
            # Calculate similarities to all tickets
            similarities = cosine_similarity(embedding_i, self.embeddings)[0]
            
            # Find tickets above threshold
            similar_indices = np.where(similarities >= threshold)[0]
            
            # Union with similar tickets
            for idx_j in similar_indices:
                if idx_j != idx_i:
                    uf.union(idx_i, idx_j)
        
        # Get components
        components = uf.get_components()
        
        # Calculate metrics
        group_sizes = [len(comp) for comp in components]
        singleton_count = sum(1 for size in group_sizes if size == 1)
        singleton_percentage = (singleton_count / len(components)) * 100
        
        return {
            'threshold': threshold,
            'total_groups': len(components),
            'singleton_count': singleton_count,
            'singleton_percentage': singleton_percentage,
            'average_group_size': np.mean(group_sizes),
            'largest_group': max(group_sizes),
            'components': components  # Store for actual use if selected
        }
    
    def find_problem_groups_union_find(self, threshold: float) -> dict:
        """Find problem groups using union-find clustering."""
        self.logger.info(f"Finding problem groups using union-find at threshold {threshold:.2f}")
        
        n_tickets = len(self.embeddings)
        uf = UnionFind(n_tickets)
        
        # Process in chunks to manage memory
        chunk_size = 200
        edges_added = 0
        
        for i in range(0, n_tickets, chunk_size):
            end_i = min(i + chunk_size, n_tickets)
            
            for ticket_idx in range(i, end_i):
                # Calculate similarities to all tickets
                ticket_embedding = self.embeddings[ticket_idx].reshape(1, -1)
                similarities = cosine_similarity(ticket_embedding, self.embeddings)[0]
                
                # Find tickets above threshold
                similar_indices = np.where(similarities >= threshold)[0]
                
                # Union with similar tickets
                for similar_idx in similar_indices:
                    if similar_idx != ticket_idx:
                        uf.union(ticket_idx, similar_idx)
                        edges_added += 1
            
            # Progress update
            self.logger.info(f"  Processed {end_i:,}/{n_tickets:,} tickets, {edges_added:,} edges added")
        
        # Get connected components
        components = uf.get_components()
        
        self.logger.info(f"Found {len(components)} components from {edges_added:,} similarity edges")
        
        # Convert to group format
        groups = {}
        for group_id, ticket_indices in enumerate(components):
            # Find centroid representative
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
                'similarity_threshold_used': threshold,
                'min_similarity_in_group': min_sim,
                'max_similarity_in_group': max_sim,
                'mean_similarity_in_group': mean_sim,
                'clustering_method': 'union_find'
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
    
    def calculate_quality_metrics(self) -> dict:
        """Calculate quality metrics for the grouping."""
        self.logger.info("Calculating quality metrics...")
        
        if not self.problem_groups:
            raise ValueError("No problem groups found. Run clustering first.")
        
        # Group size statistics
        group_sizes = [group['group_size'] for group in self.problem_groups.values()]
        
        # Similarity statistics
        min_similarities = [group['min_similarity_in_group'] for group in self.problem_groups.values()]
        mean_similarities = [group['mean_similarity_in_group'] for group in self.problem_groups.values()]
        
        # Coverage analysis
        total_tickets = len(self.tickets_df)
        tickets_in_groups = sum(group_sizes)
        
        metrics = {
            'total_groups': len(self.problem_groups),
            'total_tickets_grouped': tickets_in_groups,
            'coverage_percentage': (tickets_in_groups / total_tickets) * 100,
            'threshold_used': self.optimal_threshold,
            'threshold_search_iterations': len(self.threshold_search_results),
            'average_group_size': float(np.mean(group_sizes)),
            'median_group_size': float(np.median(group_sizes)),
            'largest_group_size': int(np.max(group_sizes)),
            'smallest_group_size': int(np.min(group_sizes)),
            'singleton_groups': sum(1 for size in group_sizes if size == 1),
            'singleton_percentage': (sum(1 for size in group_sizes if size == 1) / len(group_sizes)) * 100,
            'large_groups_50plus': sum(1 for size in group_sizes if size >= 50),
            'medium_groups_10_to_49': sum(1 for size in group_sizes if 10 <= size < 50),
            'small_groups_2_to_9': sum(1 for size in group_sizes if 2 <= size < 10),
            'group_size_distribution': {
                'singleton': sum(1 for size in group_sizes if size == 1),
                'small_2_9': sum(1 for size in group_sizes if 2 <= size < 10),
                'medium_10_49': sum(1 for size in group_sizes if 10 <= size < 50),
                'large_50plus': sum(1 for size in group_sizes if size >= 50)
            },
            'similarity_quality': {
                'min_similarity_mean': float(np.mean(min_similarities)),
                'mean_similarity_mean': float(np.mean(mean_similarities)),
                'groups_below_threshold': sum(1 for sim in min_similarities if sim < self.optimal_threshold)
            },
            'clustering_method': 'union_find_transitive'
        }
        
        self.logger.info(f"Quality metrics:")
        self.logger.info(f"  Total groups: {metrics['total_groups']:,}")
        self.logger.info(f"  Threshold used: {metrics['threshold_used']:.2f}")
        self.logger.info(f"  Average group size: {metrics['average_group_size']:.1f}")
        self.logger.info(f"  Singleton percentage: {metrics['singleton_percentage']:.1f}%")
        self.logger.info(f"  Large groups (50+): {metrics['large_groups_50plus']}")
        self.logger.info(f"  Medium groups (10-49): {metrics['medium_groups_10_to_49']}")
        self.logger.info(f"  Reduction ratio: {total_tickets / metrics['total_groups']:.1f}:1")
        
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
                'similarity_threshold_used': group_data['similarity_threshold_used'],
                'clustering_method': group_data['clustering_method'],
                'representative_ticket_index': rep_idx,
                'representative_short_description': rep_ticket['Short description'],
                'all_ticket_indices': ','.join(map(str, group_data['ticket_indices'])),
                'min_similarity_in_group': group_data['min_similarity_in_group'],
                'max_similarity_in_group': group_data['max_similarity_in_group'],
                'mean_similarity_in_group': group_data['mean_similarity_in_group'],
                'manual_review_priority': 'HIGH' if group_data['group_size'] >= 50 else 
                                        'MEDIUM' if group_data['group_size'] >= 10 else 'LOW'
            })
        
        groups_summary_df = pd.DataFrame(groups_summary)
        groups_summary_df = groups_summary_df.sort_values('group_size', ascending=False)
        
        summary_file = output_dir / 'problem_groups_v2_detailed.csv'
        groups_summary_df.to_csv(summary_file, index=False)
        
        # 2. Individual ticket assignments
        ticket_details = []
        for group_name, group_data in self.problem_groups.items():
            rep_idx = group_data['representative_ticket_index']
            rep_embedding = self.embeddings[rep_idx].reshape(1, -1)
            
            for ticket_idx in group_data['ticket_indices']:
                ticket = self.tickets_df.loc[ticket_idx]
                
                # Calculate similarity to representative
                if ticket_idx == rep_idx:
                    similarity_to_rep = 1.0
                else:
                    ticket_embedding = self.embeddings[ticket_idx].reshape(1, -1)
                    similarity_to_rep = cosine_similarity(rep_embedding, ticket_embedding)[0][0]
                
                ticket_details.append({
                    'problem_group_id': group_data['group_id'],
                    'ticket_index': ticket_idx,
                    'short_description': ticket['Short description'],
                    'full_description': ticket.get('Description', ''),
                    'original_category': ticket.get('Category', ''),
                    'similarity_to_representative': float(similarity_to_rep),
                    'is_representative': ticket_idx == rep_idx,
                    'clustering_method': group_data['clustering_method'],
                    'manual_review_flag': 'REQUIRED' if group_data['group_size'] >= 50 else 
                                        'RECOMMENDED' if group_data['group_size'] >= 10 else 'OPTIONAL',
                    'group_size': group_data['group_size']
                })
        
        ticket_details_df = pd.DataFrame(ticket_details)
        ticket_details_df = ticket_details_df.sort_values(['problem_group_id', 'similarity_to_representative'], 
                                                         ascending=[True, False])
        
        details_file = output_dir / 'problem_group_v2_details.csv'
        ticket_details_df.to_csv(details_file, index=False)
        
        # 3. Threshold search results
        threshold_search_df = pd.DataFrame(self.threshold_search_results)
        threshold_search_file = output_dir / 'threshold_search_results.csv'
        threshold_search_df.to_csv(threshold_search_file, index=False)
        
        self.logger.info(f"Detailed outputs created:")
        self.logger.info(f"  Summary: {summary_file}")
        self.logger.info(f"  Details: {details_file}")
        self.logger.info(f"  Threshold search: {threshold_search_file}")
        
        return summary_file, details_file, threshold_search_file

def main():
    """Main problem grouping pipeline V2."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("="*80)
    print("PHASE 1: SEMANTIC PROBLEM GROUPING V2 - FIXED APPROACH")
    print("="*80)
    print("FIXES:")
    print("• Automatic threshold search (problem-type focused)")
    print("• Union-find clustering (transitive similarity)")
    print("• Target ≤40% singletons (realistic for problem grouping)")
    print("• Proper similarity range (0.45-0.75, not 0.85+)")
    print("="*80)
    
    # File paths
    input_file = Path('data/processed/consolidated_tickets.csv')
    output_dir = Path('outputs')
    
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        print("   Please ensure consolidated_tickets.csv exists")
        return
    
    # Initialize improved grouper
    print(f"\n🎯 CONFIGURATION:")
    print(f"   Target singleton percentage: ≤40%")
    print(f"   Maximum groups: ≤500")
    print(f"   Clustering method: Union-find (transitive)")
    print(f"   Representative selection: Centroid")
    print(f"   Threshold search: Automatic")
    
    grouper = ProblemGroupingV2(
        target_singleton_percent=40,
        max_groups=500
    )
    
    try:
        # Step 1: Load data
        print(f"\n🔄 STEP 1: Loading consolidated tickets")
        df = grouper.load_consolidated_tickets(input_file)
        
        # Step 2: Generate embeddings
        print(f"\n🔄 STEP 2: Generating semantic embeddings")
        embeddings, texts = grouper.generate_embeddings()
        
        # Step 3: Analyze similarity distribution
        print(f"\n🔄 STEP 3: Analyzing similarity distribution")
        similarity_stats = grouper.calculate_similarity_distribution()
        
        # Step 4: Find optimal threshold automatically
        print(f"\n🔄 STEP 4: Finding optimal threshold (automatic search)")
        optimal_threshold = grouper.find_optimal_threshold_automatic()
        
        # Step 5: Find problem groups using union-find
        print(f"\n🔄 STEP 5: Clustering with union-find at threshold {optimal_threshold:.2f}")
        groups = grouper.find_problem_groups_union_find(optimal_threshold)
        
        # Step 6: Calculate quality metrics
        print(f"\n🔄 STEP 6: Calculating quality metrics")
        quality_metrics = grouper.calculate_quality_metrics()
        
        # Step 7: Create outputs
        print(f"\n🔄 STEP 7: Creating detailed outputs")
        summary_file, details_file, threshold_file = grouper.create_detailed_outputs(output_dir)
        
        # Final summary
        print(f"\n📊 PROBLEM GROUPING RESULTS V2:")
        print(f"   Input tickets: {len(df):,}")
        print(f"   Problem groups identified: {quality_metrics['total_groups']:,}")
        print(f"   Optimal threshold found: {optimal_threshold:.2f}")
        print(f"   Reduction ratio: {len(df) / quality_metrics['total_groups']:.1f}:1")
        print(f"   Average group size: {quality_metrics['average_group_size']:.1f}")
        print(f"   Largest group: {quality_metrics['largest_group_size']} tickets")
        print(f"   Singleton groups: {quality_metrics['singleton_groups']:,} ({quality_metrics['singleton_percentage']:.1f}%)")
        
        print(f"\n🎯 SUCCESS METRICS:")
        success_singleton = quality_metrics['singleton_percentage'] <= 40
        success_groups = quality_metrics['total_groups'] <= 500
        success_reduction = (len(df) / quality_metrics['total_groups']) >= 3
        
        print(f"   ✅ Singleton target (≤40%): {'PASS' if success_singleton else 'FAIL'} ({quality_metrics['singleton_percentage']:.1f}%)")
        print(f"   ✅ Group count target (≤500): {'PASS' if success_groups else 'FAIL'} ({quality_metrics['total_groups']})")
        print(f"   ✅ Reduction target (≥3:1): {'PASS' if success_reduction else 'FAIL'} ({len(df) / quality_metrics['total_groups']:.1f}:1)")
        
        print(f"\n🔍 QUALITY CONTROL REQUIREMENTS:")
        print(f"   High priority review (50+ tickets): {quality_metrics['large_groups_50plus']} groups")
        print(f"   Medium priority review (10-49 tickets): {quality_metrics['medium_groups_10_to_49']} groups")
        print(f"   Total groups requiring review: {quality_metrics['large_groups_50plus'] + quality_metrics['medium_groups_10_to_49']}")
        
        print(f"\n📁 OUTPUT FILES CREATED:")
        print(f"   📄 Summary: {summary_file}")
        print(f"   📄 Details: {details_file}")
        print(f"   📄 Threshold search: {threshold_file}")
        
        print(f"\n✅ PHASE 1 V2 COMPLETE")
        if success_singleton and success_groups and success_reduction:
            print(f"   🎉 ALL SUCCESS CRITERIA MET!")
            print(f"   🎯 Ready for manual quality validation")
            print(f"   📊 Significant improvement over V1 results")
        else:
            print(f"   ⚠️ Some targets not met - review threshold search results")
        
        print(f"   🔍 Next: Review problem groups for quality assurance")
        
        logger.info("Phase 1 V2: Fixed semantic problem grouping completed successfully!")
        
    except Exception as e:
        logger.error(f"Problem grouping V2 failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()