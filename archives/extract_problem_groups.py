#!/usr/bin/env python3
"""
PHASE 1: SEMANTIC PROBLEM GROUPING WITH QUALITY CONTROL

This script implements the first phase of problem statement extraction:
1. Load 3,847 consolidated tickets
2. Generate semantic embeddings using proven pipeline
3. Apply adaptive similarity grouping (0.85 → 0.75 floor)
4. Use centroid-based representative selection
5. Create detailed output with full traceability
6. Include statistical validation metrics

Focus: Quality and transparency over quantity - prevent misclassifications
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
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

class SemanticProblemGrouper:
    """Conservative semantic grouping with adaptive thresholds and centroid representatives."""
    
    def __init__(self, initial_threshold=0.85, floor_threshold=0.75, model_name='all-MiniLM-L6-v2'):
        self.logger = logging.getLogger(__name__)
        self.initial_threshold = initial_threshold
        self.floor_threshold = floor_threshold
        self.current_threshold = initial_threshold
        self.model_name = model_name
        
        # Load sentence transformer model
        self.logger.info(f"Loading sentence transformer model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.logger.info("Model loaded successfully")
        
        # Results storage
        self.embeddings = None
        self.tickets_df = None
        self.problem_groups = {}
        self.quality_metrics = {}
        self.threshold_iterations = []
        
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
    
    def calculate_similarity_statistics(self) -> dict:
        """Calculate similarity statistics for quality assessment."""
        self.logger.info("Calculating similarity statistics...")
        
        # Calculate pairwise similarities (sample for large datasets)
        n_tickets = len(self.embeddings)
        
        if n_tickets > 1000:
            # Sample for computational efficiency
            sample_size = 1000
            indices = np.random.choice(n_tickets, sample_size, replace=False)
            sample_embeddings = self.embeddings[indices]
            self.logger.info(f"Using sample of {sample_size} tickets for similarity statistics")
        else:
            sample_embeddings = self.embeddings
            indices = np.arange(n_tickets)
        
        # Calculate pairwise cosine similarities
        similarity_matrix = cosine_similarity(sample_embeddings)
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = np.triu(similarity_matrix, k=1)
        similarities = upper_triangle[upper_triangle > 0]
        
        stats = {
            'mean_similarity': float(np.mean(similarities)),
            'median_similarity': float(np.median(similarities)),
            'std_similarity': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'percentile_95': float(np.percentile(similarities, 95)),
            'percentile_99': float(np.percentile(similarities, 99)),
            'above_initial_threshold': float(np.mean(similarities >= self.initial_threshold) * 100),
            'above_floor_threshold': float(np.mean(similarities >= self.floor_threshold) * 100),
            'sample_size': len(indices),
            'total_comparisons': len(similarities)
        }
        
        self.logger.info(f"Similarity statistics:")
        self.logger.info(f"  Mean: {stats['mean_similarity']:.3f}")
        self.logger.info(f"  Median: {stats['median_similarity']:.3f}")
        self.logger.info(f"  95th percentile: {stats['percentile_95']:.3f}")
        self.logger.info(f"  Above initial threshold ({self.initial_threshold}): {stats['above_initial_threshold']:.1f}%")
        self.logger.info(f"  Above floor threshold ({self.floor_threshold}): {stats['above_floor_threshold']:.1f}%")
        
        return stats
    
    def find_centroid_representative(self, ticket_indices: list) -> int:
        """Find the centroid ticket (highest average similarity to others in group)."""
        if len(ticket_indices) == 1:
            return ticket_indices[0]
        
        # Get embeddings for tickets in this group
        group_embeddings = self.embeddings[ticket_indices]
        
        # Calculate pairwise similarities within group
        similarity_matrix = cosine_similarity(group_embeddings)
        
        # Find ticket with highest average similarity to others
        # (excluding self-similarity of 1.0)
        avg_similarities = []
        for i in range(len(ticket_indices)):
            # Get similarities to all other tickets in group (excluding self)
            similarities_to_others = np.concatenate([
                similarity_matrix[i, :i],  # Before current ticket
                similarity_matrix[i, i+1:]  # After current ticket
            ])
            avg_similarity = np.mean(similarities_to_others) if len(similarities_to_others) > 0 else 0
            avg_similarities.append(avg_similarity)
        
        # Return ticket index with highest average similarity
        centroid_idx = np.argmax(avg_similarities)
        return ticket_indices[centroid_idx]
    
    def find_problem_groups_adaptive(self) -> dict:
        """Find problem groups using adaptive similarity threshold."""
        self.logger.info(f"Finding problem groups with adaptive thresholds: {self.initial_threshold} → {self.floor_threshold}")
        
        # Try initial threshold first
        groups = self._find_groups_at_threshold(self.initial_threshold)
        singleton_count = sum(1 for group in groups.values() if group['group_size'] == 1)
        singleton_percentage = (singleton_count / len(groups)) * 100
        
        self.threshold_iterations.append({
            'threshold': self.initial_threshold,
            'total_groups': len(groups),
            'singleton_groups': singleton_count,
            'singleton_percentage': singleton_percentage
        })
        
        self.logger.info(f"Initial threshold {self.initial_threshold}: {len(groups)} groups, {singleton_count} singletons ({singleton_percentage:.1f}%)")
        
        # If too many singletons (>60%), try lower threshold
        if singleton_percentage > 60 and self.initial_threshold > self.floor_threshold:
            self.logger.info(f"Too many singleton groups ({singleton_percentage:.1f}%), trying floor threshold {self.floor_threshold}")
            
            # Try floor threshold
            floor_groups = self._find_groups_at_threshold(self.floor_threshold)
            floor_singleton_count = sum(1 for group in floor_groups.values() if group['group_size'] == 1)
            floor_singleton_percentage = (floor_singleton_count / len(floor_groups)) * 100
            
            self.threshold_iterations.append({
                'threshold': self.floor_threshold,
                'total_groups': len(floor_groups),
                'singleton_groups': floor_singleton_count,
                'singleton_percentage': floor_singleton_percentage
            })
            
            self.logger.info(f"Floor threshold {self.floor_threshold}: {len(floor_groups)} groups, {floor_singleton_count} singletons ({floor_singleton_percentage:.1f}%)")
            
            # Choose better result (fewer singletons, but not if it creates too few groups)
            if floor_singleton_percentage < singleton_percentage * 0.8:  # Significant improvement
                self.logger.info(f"Using floor threshold results (better singleton reduction)")
                groups = floor_groups
                self.current_threshold = self.floor_threshold
            else:
                self.logger.info(f"Keeping initial threshold results (floor didn't improve significantly)")
                self.current_threshold = self.initial_threshold
        else:
            self.current_threshold = self.initial_threshold
        
        self.problem_groups = groups
        return groups
    
    def _find_groups_at_threshold(self, threshold: float) -> dict:
        """Find groups at specific threshold with centroid representatives."""
        n_tickets = len(self.embeddings)
        assigned = set()
        groups = {}
        group_id = 0
        
        # Calculate similarity matrix in chunks to manage memory
        chunk_size = 500
        
        for i in range(0, n_tickets, chunk_size):
            end_i = min(i + chunk_size, n_tickets)
            
            for ticket_idx in range(i, end_i):
                if ticket_idx in assigned:
                    continue
                
                # Find all tickets similar to this one
                ticket_embedding = self.embeddings[ticket_idx].reshape(1, -1)
                similarities = cosine_similarity(ticket_embedding, self.embeddings)[0]
                
                # Find tickets above threshold (excluding self)
                similar_indices = np.where(similarities >= threshold)[0]
                similar_indices = [idx for idx in similar_indices if idx not in assigned]
                
                if len(similar_indices) >= 1:  # At least the ticket itself
                    # Find centroid representative
                    representative_idx = self.find_centroid_representative(similar_indices)
                    
                    # Calculate group similarity metrics
                    group_embeddings = self.embeddings[similar_indices]
                    group_similarity_matrix = cosine_similarity(group_embeddings)
                    
                    # Get similarities (excluding diagonal)
                    upper_triangle = np.triu(group_similarity_matrix, k=1)
                    group_similarities = upper_triangle[upper_triangle > 0]
                    
                    groups[f"problem_group_{group_id}"] = {
                        'group_id': group_id,
                        'representative_ticket_index': representative_idx,
                        'ticket_indices': similar_indices,
                        'group_size': len(similar_indices),
                        'similarity_threshold_used': threshold,
                        'min_similarity_in_group': float(np.min(group_similarities)) if len(group_similarities) > 0 else 1.0,
                        'max_similarity_in_group': float(np.max(group_similarities)) if len(group_similarities) > 0 else 1.0,
                        'mean_similarity_in_group': float(np.mean(group_similarities)) if len(group_similarities) > 0 else 1.0,
                        'representative_selection_method': 'centroid'
                    }
                    
                    # Mark all tickets in this group as assigned
                    assigned.update(similar_indices)
                    group_id += 1
        
        # Handle any unassigned tickets as singleton groups
        unassigned = [i for i in range(n_tickets) if i not in assigned]
        for ticket_idx in unassigned:
            groups[f"problem_group_{group_id}"] = {
                'group_id': group_id,
                'representative_ticket_index': ticket_idx,
                'ticket_indices': [ticket_idx],
                'group_size': 1,
                'similarity_threshold_used': threshold,
                'min_similarity_in_group': 1.0,
                'max_similarity_in_group': 1.0,
                'mean_similarity_in_group': 1.0,
                'representative_selection_method': 'singleton'
            }
            group_id += 1
        
        return groups
    
    def calculate_quality_metrics(self) -> dict:
        """Calculate quality metrics for the grouping."""
        self.logger.info("Calculating quality metrics...")
        
        if not self.problem_groups:
            raise ValueError("No problem groups found. Run find_problem_groups_adaptive() first.")
        
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
            'threshold_used': self.current_threshold,
            'threshold_iterations': self.threshold_iterations,
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
                'groups_below_threshold': sum(1 for sim in min_similarities if sim < self.current_threshold)
            },
            'representative_methods': {
                'centroid': sum(1 for group in self.problem_groups.values() 
                              if group['representative_selection_method'] == 'centroid'),
                'singleton': sum(1 for group in self.problem_groups.values() 
                               if group['representative_selection_method'] == 'singleton')
            }
        }
        
        self.logger.info(f"Quality metrics:")
        self.logger.info(f"  Total groups: {metrics['total_groups']:,}")
        self.logger.info(f"  Threshold used: {metrics['threshold_used']}")
        self.logger.info(f"  Average group size: {metrics['average_group_size']:.1f}")
        self.logger.info(f"  Singleton percentage: {metrics['singleton_percentage']:.1f}%")
        self.logger.info(f"  Large groups (50+): {metrics['large_groups_50plus']}")
        self.logger.info(f"  Coverage: {metrics['coverage_percentage']:.1f}%")
        
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
                'representative_ticket_index': rep_idx,
                'representative_selection_method': group_data['representative_selection_method'],
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
        
        summary_file = output_dir / 'problem_groups_detailed.csv'
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
                    'representative_selection_method': group_data['representative_selection_method'],
                    'manual_review_flag': 'REQUIRED' if group_data['group_size'] >= 50 else 
                                        'RECOMMENDED' if group_data['group_size'] >= 10 else 'OPTIONAL',
                    'group_size': group_data['group_size']
                })
        
        ticket_details_df = pd.DataFrame(ticket_details)
        ticket_details_df = ticket_details_df.sort_values(['problem_group_id', 'similarity_to_representative'], 
                                                         ascending=[True, False])
        
        details_file = output_dir / 'problem_group_details.csv'
        ticket_details_df.to_csv(details_file, index=False)
        
        self.logger.info(f"Detailed outputs created:")
        self.logger.info(f"  Summary: {summary_file}")
        self.logger.info(f"  Details: {details_file}")
        
        return summary_file, details_file
    
    def create_quality_report(self, output_dir: Path) -> Path:
        """Create comprehensive quality assessment report."""
        self.logger.info("Creating quality assessment report...")
        
        # Combine all metrics
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'input_data': {
                'total_tickets': len(self.tickets_df),
                'source_file': 'consolidated_tickets.csv'
            },
            'parameters': {
                'initial_threshold': self.initial_threshold,
                'floor_threshold': self.floor_threshold,
                'threshold_used': self.current_threshold,
                'embedding_model': self.model_name,
                'representative_method': 'centroid'
            },
            'similarity_statistics': self.calculate_similarity_statistics(),
            'grouping_results': self.quality_metrics,
            'quality_assessment': {
                'threshold_adaptation': {
                    'iterations_tested': len(self.threshold_iterations),
                    'final_threshold': self.current_threshold,
                    'singleton_reduction_achieved': len(self.threshold_iterations) > 1,
                    'threshold_history': self.threshold_iterations
                },
                'recommended_manual_review': {
                    'high_priority_groups': self.quality_metrics['large_groups_50plus'],
                    'medium_priority_groups': self.quality_metrics['medium_groups_10_to_49'],
                    'total_groups_for_review': (self.quality_metrics['large_groups_50plus'] + 
                                              self.quality_metrics['medium_groups_10_to_49'])
                },
                'consolidation_achievement': {
                    'original_tickets': len(self.tickets_df),
                    'problem_groups': self.quality_metrics['total_groups'],
                    'reduction_ratio': len(self.tickets_df) / self.quality_metrics['total_groups'],
                    'reduction_percentage': ((len(self.tickets_df) - self.quality_metrics['total_groups']) 
                                           / len(self.tickets_df)) * 100,
                    'singleton_percentage': self.quality_metrics['singleton_percentage']
                }
            }
        }
        
        report_file = output_dir / 'problem_grouping_quality_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Quality report saved: {report_file}")
        return report_file

def main():
    """Main problem grouping pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("="*80)
    print("PHASE 1: SEMANTIC PROBLEM GROUPING WITH QUALITY CONTROL")
    print("="*80)
    print("Extracting distinct problem groups from consolidated tickets")
    print("Focus: Quality and transparency over quantity")
    print("="*80)
    
    # File paths
    input_file = Path('data/processed/consolidated_tickets.csv')
    output_dir = Path('outputs')
    
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        print("   Please ensure consolidated_tickets.csv exists")
        return
    
    # Initialize grouper with adaptive thresholds and centroid representatives
    initial_threshold = 0.85
    floor_threshold = 0.75
    print(f"\n🎯 CONFIGURATION:")
    print(f"   Initial similarity threshold: {initial_threshold}")
    print(f"   Floor similarity threshold: {floor_threshold}")
    print(f"   Representative selection: Centroid (highest avg similarity)")
    print(f"   Embedding model: all-MiniLM-L6-v2")
    print(f"   Quality focus: Zero tolerance for misclassifications")
    
    grouper = SemanticProblemGrouper(
        initial_threshold=initial_threshold,
        floor_threshold=floor_threshold
    )
    
    try:
        # Step 1: Load data
        print(f"\n🔄 STEP 1: Loading consolidated tickets")
        df = grouper.load_consolidated_tickets(input_file)
        
        # Step 2: Generate embeddings
        print(f"\n🔄 STEP 2: Generating semantic embeddings")
        embeddings, texts = grouper.generate_embeddings()
        
        # Step 3: Calculate similarity statistics
        print(f"\n🔄 STEP 3: Analyzing similarity distribution")
        similarity_stats = grouper.calculate_similarity_statistics()
        
        # Step 4: Find problem groups with adaptive thresholds
        print(f"\n🔄 STEP 4: Finding problem groups (adaptive thresholds)")
        groups = grouper.find_problem_groups_adaptive()
        
        # Step 5: Calculate quality metrics
        print(f"\n🔄 STEP 5: Calculating quality metrics")
        quality_metrics = grouper.calculate_quality_metrics()
        
        # Step 6: Create outputs
        print(f"\n🔄 STEP 6: Creating detailed outputs")
        summary_file, details_file = grouper.create_detailed_outputs(output_dir)
        quality_report = grouper.create_quality_report(output_dir)
        
        # Final summary
        print(f"\n📊 PROBLEM GROUPING RESULTS:")
        print(f"   Input tickets: {len(df):,}")
        print(f"   Problem groups identified: {quality_metrics['total_groups']:,}")
        print(f"   Similarity threshold used: {quality_metrics['threshold_used']}")
        print(f"   Reduction ratio: {len(df) / quality_metrics['total_groups']:.1f}:1")
        print(f"   Average group size: {quality_metrics['average_group_size']:.1f}")
        print(f"   Largest group: {quality_metrics['largest_group_size']} tickets")
        print(f"   Singleton groups: {quality_metrics['singleton_groups']:,} ({quality_metrics['singleton_percentage']:.1f}%)")
        
        print(f"\n🎯 REPRESENTATIVE SELECTION:")
        print(f"   Centroid representatives: {quality_metrics['representative_methods']['centroid']:,}")
        print(f"   Singleton representatives: {quality_metrics['representative_methods']['singleton']:,}")
        
        print(f"\n🔍 QUALITY CONTROL REQUIREMENTS:")
        print(f"   High priority review (50+ tickets): {quality_metrics['large_groups_50plus']} groups")
        print(f"   Medium priority review (10-49 tickets): {quality_metrics['medium_groups_10_to_49']} groups")
        print(f"   Total groups requiring review: {quality_metrics['large_groups_50plus'] + quality_metrics['medium_groups_10_to_49']}")
        
        print(f"\n📁 OUTPUT FILES CREATED:")
        print(f"   📄 Summary: {summary_file}")
        print(f"   📄 Details: {details_file}")
        print(f"   📄 Quality report: {quality_report}")
        
        print(f"\n✅ PHASE 1 COMPLETE")
        print(f"   🎯 Ready for manual quality validation")
        print(f"   📊 {quality_metrics['coverage_percentage']:.1f}% ticket coverage achieved")
        print(f"   🔧 Adaptive thresholding prevented excessive singleton groups")
        print(f"   🎯 Centroid representatives improve group interpretability")
        print(f"   🔍 Next: Review problem groups for quality assurance")
        
        logger.info("Phase 1: Semantic problem grouping completed successfully!")
        
    except Exception as e:
        logger.error(f"Problem grouping failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()