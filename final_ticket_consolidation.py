#!/usr/bin/env python3
"""
FINAL TICKET CONSOLIDATION - Complete Classification System

This script creates the definitive final classification by consolidating ALL processing stages:

STAGE 1: Load 6,964 original tickets
STAGE 2: Map 411 identical groups (3,528 tickets) to their representative classifications  
STAGE 3: Apply hardcoded rule classifications (639 tickets)
STAGE 4: Apply hierarchical clustering business categories (3,208 tickets)
STAGE 5: Create final unified CSV with 100% classification coverage

Result: ONE final CSV with all 6,964 tickets properly classified and labeled
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

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

class FinalTicketConsolidator:
    """Complete ticket classification consolidation system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Business category mapping from hierarchical clustering
        self.business_categories = {
            'L1_0_L2_0': {
                'name': 'Software & Application Issues',
                'subcategory': 'Vision/Appstream Issues',
                'queue': 'SOFTWARE_AND_APPLICATION_ISSUES',
                'routing_team': 'Application Support',
                'priority': 'High',
                'sla_hours': 4
            },
            'L1_1_L2_0': {
                'name': 'Printing Services',
                'subcategory': 'Vision Printing',
                'queue': 'PRINT_VISION',
                'routing_team': 'Infrastructure',
                'priority': 'Medium',
                'sla_hours': 8
            },
            'L1_1_L2_1': {
                'name': 'Printing Services',
                'subcategory': 'Infrastructure Printing',
                'queue': 'PRINT_INFRA',
                'routing_team': 'Infrastructure',
                'priority': 'Medium',
                'sla_hours': 8
            },
            'L1_1_edge-case': {
                'name': 'Printing Services',
                'subcategory': 'Complex Printing Issues',
                'queue': 'PRINT_ESCALATION',
                'routing_team': 'Infrastructure',
                'priority': 'Medium',
                'sla_hours': 8
            },
            'L1_2_L2_0': {
                'name': 'Back Office & Financial',
                'subcategory': 'Financial Operations',
                'queue': 'BACK_OFFICE_AND_FINANCIAL',
                'routing_team': 'Business Operations',
                'priority': 'High',
                'sla_hours': 2
            },
            'L1_3': {
                'name': 'Till Operations',
                'subcategory': 'Till Management',
                'queue': 'TILL_OPERATIONS',
                'routing_team': 'Store Systems',
                'priority': 'Critical',
                'sla_hours': 1
            },
            'L1_4': {
                'name': 'Payment Processing',
                'subcategory': 'Card Payment Issues',
                'queue': 'PAYMENT_PROCESSING',
                'routing_team': 'Payment Systems',
                'priority': 'Critical',
                'sla_hours': 1
            },
            'L1_5': {
                'name': 'Mobile Devices',
                'subcategory': 'Zebra & Mobile Hardware',
                'queue': 'MOBILE_DEVICES',
                'routing_team': 'Hardware Support',
                'priority': 'Medium',
                'sla_hours': 6
            },
            'L1_6_L2_0': {
                'name': 'Inventory & SKU Management',
                'subcategory': 'SKU Operations',
                'queue': 'INVENTORY_AND_SKU_MANAGEMENT',
                'routing_team': 'Business Operations',
                'priority': 'Medium',
                'sla_hours': 8
            },
            'L1_7': {
                'name': 'Performance & Software Issues',
                'subcategory': 'System Performance',
                'queue': 'PERFORMANCE_AND_SOFTWARE_ISSUES',
                'routing_team': 'Application Support',
                'priority': 'Medium',
                'sla_hours': 6
            },
            'L1_8_L2_0': {
                'name': 'User Account Management',
                'subcategory': 'Active Directory & Authentication',
                'queue': 'USER_ACCOUNT_MANAGEMENT',
                'routing_team': 'Identity Management',
                'priority': 'Medium',
                'sla_hours': 4
            },
            'L1_9': {
                'name': 'Email & Communications',
                'subcategory': 'Email & Communication Tools',
                'queue': 'EMAIL_AND_COMMUNICATIONS',
                'routing_team': 'End User Support',
                'priority': 'Low',
                'sla_hours': 24
            }
        }
        
        # Stage tracking for validation
        self.stage_results = {}
        
    def stage_1_load_original_tickets(self):
        """STAGE 1: Load all 6,964 original tickets."""
        self.logger.info("🔄 STAGE 1: Loading original ticket dataset")
        
        original_file = Path('data/processed/clean_tickets.csv')
        if not original_file.exists():
            raise FileNotFoundError(f"Original tickets file not found: {original_file}")
        
        df_original = pd.read_csv(original_file)
        
        # Initialize final classification columns
        df_original['classification_method'] = 'UNCLASSIFIED'
        df_original['final_business_category'] = 'UNCLASSIFIED' 
        df_original['final_subcategory'] = 'UNCLASSIFIED'
        df_original['final_queue'] = 'UNCLASSIFIED'
        df_original['routing_team'] = 'UNCLASSIFIED'
        df_original['priority_level'] = 'UNCLASSIFIED'
        df_original['sla_hours'] = np.nan
        df_original['confidence_score'] = 0.0
        df_original['is_edge_case'] = False
        df_original['fuzzy_group_id'] = -1
        df_original['fuzzy_group_size'] = 1
        df_original['representative_ticket_index'] = -1
        df_original['processing_notes'] = ''
        
        self.stage_results['stage_1'] = {
            'description': 'Original ticket dataset loaded',
            'count': len(df_original),
            'file': str(original_file)
        }
        
        self.logger.info(f"   ✅ Loaded {len(df_original):,} original tickets")
        self.logger.info(f"   📊 Columns: {list(df_original.columns)}")
        
        return df_original
    
    def stage_2_map_identical_groups(self, df_original):
        """STAGE 2: Map 411 identical groups back to original tickets."""
        self.logger.info("🔄 STAGE 2: Mapping identical groups to original tickets")
        
        # Load identical groups analysis
        identical_file = Path('outputs/identical_tickets_analysis.json')
        if not identical_file.exists():
            raise FileNotFoundError(f"Identical groups file not found: {identical_file}")
        
        with open(identical_file, 'r') as f:
            identical_data = json.load(f)
        
        identical_groups = identical_data['results']['identical_groups']
        
        # Create group mapping
        group_mapping = {}
        grouped_tickets = set()
        
        for group in identical_groups:
            group_id = group['group_id']
            ticket_indices = group['ticket_indices']
            representative_idx = ticket_indices[0]  # First ticket is representative
            
            for ticket_idx in ticket_indices:
                group_mapping[ticket_idx] = {
                    'group_id': group_id,
                    'group_size': len(ticket_indices),
                    'representative_index': representative_idx,
                    'is_representative': ticket_idx == representative_idx
                }
                grouped_tickets.add(ticket_idx)
        
        # Apply group mappings to original dataset
        tickets_mapped = 0
        for idx, row in df_original.iterrows():
            if idx in group_mapping:
                mapping = group_mapping[idx]
                df_original.at[idx, 'fuzzy_group_id'] = mapping['group_id']
                df_original.at[idx, 'fuzzy_group_size'] = mapping['group_size']
                df_original.at[idx, 'representative_ticket_index'] = mapping['representative_index']
                
                if mapping['is_representative']:
                    df_original.at[idx, 'processing_notes'] = f"Representative of {mapping['group_size']} similar tickets"
                else:
                    df_original.at[idx, 'processing_notes'] = f"Similar to ticket {mapping['representative_index']}"
                
                tickets_mapped += 1
        
        self.stage_results['stage_2'] = {
            'description': 'Identical group mappings applied',
            'total_groups': len(identical_groups),
            'tickets_in_groups': len(grouped_tickets),
            'tickets_mapped': tickets_mapped,
            'file': str(identical_file)
        }
        
        self.logger.info(f"   ✅ Mapped {len(identical_groups):,} groups containing {len(grouped_tickets):,} tickets")
        self.logger.info(f"   📊 {tickets_mapped:,} tickets updated with group information")
        
        return df_original
    
    def stage_3_apply_hardcoded_classifications(self, df_original):
        """STAGE 3: Apply hardcoded rule classifications."""
        self.logger.info("🔄 STAGE 3: Applying hardcoded rule classifications")
        
        # Load hardcoded classification results
        hardcoded_file = Path('outputs/improved_classification_results.json')
        if not hardcoded_file.exists():
            raise FileNotFoundError(f"Hardcoded classification file not found: {hardcoded_file}")
        
        with open(hardcoded_file, 'r') as f:
            hardcoded_data = json.load(f)
        
        classified_tickets = hardcoded_data['classified_tickets']
        
        # Apply hardcoded classifications
        tickets_classified = 0
        for ticket in classified_tickets:
            ticket_idx = ticket['ticket_index']
            
            if ticket_idx in df_original.index:
                df_original.at[ticket_idx, 'classification_method'] = 'HARDCODED_RULES'
                df_original.at[ticket_idx, 'final_business_category'] = ticket['category']
                df_original.at[ticket_idx, 'final_subcategory'] = f"Rule-based {ticket['category']}"
                df_original.at[ticket_idx, 'final_queue'] = ticket['category'].upper().replace(' ', '_')
                df_original.at[ticket_idx, 'confidence_score'] = ticket.get('confidence_score', 0.9)
                
                # Set priority and routing based on category
                if ticket['category'] in ['Till', 'Chip & Pin']:
                    df_original.at[ticket_idx, 'priority_level'] = 'Critical'
                    df_original.at[ticket_idx, 'sla_hours'] = 1
                    df_original.at[ticket_idx, 'routing_team'] = 'Store Systems'
                elif ticket['category'] in ['Vision', 'Back Office']:
                    df_original.at[ticket_idx, 'priority_level'] = 'High'
                    df_original.at[ticket_idx, 'sla_hours'] = 2
                    df_original.at[ticket_idx, 'routing_team'] = 'Business Operations'
                else:
                    df_original.at[ticket_idx, 'priority_level'] = 'Medium'
                    df_original.at[ticket_idx, 'sla_hours'] = 6
                    df_original.at[ticket_idx, 'routing_team'] = 'Application Support'
                
                df_original.at[ticket_idx, 'processing_notes'] = f"Classified by hardcoded rules: {ticket.get('pattern_matched', 'Multiple patterns')}"
                tickets_classified += 1
        
        self.stage_results['stage_3'] = {
            'description': 'Hardcoded rule classifications applied',
            'classified_tickets': len(classified_tickets),
            'tickets_updated': tickets_classified,
            'file': str(hardcoded_file)
        }
        
        self.logger.info(f"   ✅ Applied hardcoded classifications to {tickets_classified:,} tickets")
        
        return df_original
    
    def stage_4_apply_hierarchical_clustering(self, df_original):
        """STAGE 4: Apply hierarchical clustering business categories."""
        self.logger.info("🔄 STAGE 4: Applying hierarchical clustering classifications")
        
        # Load hierarchical clustering assignments
        hierarchical_file = Path('outputs/hierarchical_cluster_assignments.csv')
        if not hierarchical_file.exists():
            raise FileNotFoundError(f"Hierarchical clustering file not found: {hierarchical_file}")
        
        df_hierarchical = pd.read_csv(hierarchical_file)
        
        # Apply hierarchical classifications
        tickets_classified = 0
        edge_cases_found = 0
        
        for _, row in df_hierarchical.iterrows():
            ticket_idx = row['ticket_index']
            hierarchical_label = row['hierarchical_label']
            
            if ticket_idx in df_original.index:
                # Skip if already classified by hardcoded rules
                if df_original.at[ticket_idx, 'classification_method'] != 'UNCLASSIFIED':
                    continue
                
                # Apply hierarchical classification
                df_original.at[ticket_idx, 'classification_method'] = 'HIERARCHICAL_CLUSTERING'
                
                # Handle edge cases
                if 'edge-case' in hierarchical_label:
                    df_original.at[ticket_idx, 'is_edge_case'] = True
                    df_original.at[ticket_idx, 'confidence_score'] = 0.3
                    edge_cases_found += 1
                    
                    # Extract parent cluster for edge case
                    parent_cluster = hierarchical_label.split('_')[1]
                    parent_label = f"L1_{parent_cluster}"
                    
                    # Find a matching business category for the parent
                    matching_category = None
                    for label, category in self.business_categories.items():
                        if parent_label in label:
                            matching_category = category
                            break
                    
                    if matching_category:
                        df_original.at[ticket_idx, 'final_business_category'] = matching_category['name']
                        df_original.at[ticket_idx, 'final_subcategory'] = 'Edge Case - Manual Review Required'
                        df_original.at[ticket_idx, 'final_queue'] = matching_category['queue'] + '_ESCALATION'
                        df_original.at[ticket_idx, 'routing_team'] = matching_category['routing_team']
                        df_original.at[ticket_idx, 'priority_level'] = 'High'  # Edge cases get higher priority
                        df_original.at[ticket_idx, 'sla_hours'] = 4
                    else:
                        df_original.at[ticket_idx, 'final_business_category'] = 'EDGE_CASE_REVIEW'
                        df_original.at[ticket_idx, 'final_subcategory'] = 'Manual Review Required'
                        df_original.at[ticket_idx, 'final_queue'] = 'EDGE_CASE_ESCALATION'
                        df_original.at[ticket_idx, 'routing_team'] = 'Senior Analysts'
                        df_original.at[ticket_idx, 'priority_level'] = 'High'
                        df_original.at[ticket_idx, 'sla_hours'] = 4
                    
                    df_original.at[ticket_idx, 'processing_notes'] = f"Edge case in hierarchical clustering: {hierarchical_label}"
                
                else:
                    # Normal hierarchical classification
                    if hierarchical_label in self.business_categories:
                        category = self.business_categories[hierarchical_label]
                        df_original.at[ticket_idx, 'final_business_category'] = category['name']
                        df_original.at[ticket_idx, 'final_subcategory'] = category['subcategory']
                        df_original.at[ticket_idx, 'final_queue'] = category['queue']
                        df_original.at[ticket_idx, 'routing_team'] = category['routing_team']
                        df_original.at[ticket_idx, 'priority_level'] = category['priority']
                        df_original.at[ticket_idx, 'sla_hours'] = category['sla_hours']
                        df_original.at[ticket_idx, 'confidence_score'] = 0.7
                        df_original.at[ticket_idx, 'processing_notes'] = f"Hierarchical clustering: {hierarchical_label}"
                    else:
                        # Unmapped hierarchical label
                        df_original.at[ticket_idx, 'final_business_category'] = 'UNMAPPED_CLUSTER'
                        df_original.at[ticket_idx, 'final_subcategory'] = hierarchical_label
                        df_original.at[ticket_idx, 'final_queue'] = 'UNMAPPED_REVIEW'
                        df_original.at[ticket_idx, 'routing_team'] = 'Data Team'
                        df_original.at[ticket_idx, 'priority_level'] = 'Medium'
                        df_original.at[ticket_idx, 'sla_hours'] = 8
                        df_original.at[ticket_idx, 'confidence_score'] = 0.5
                        df_original.at[ticket_idx, 'processing_notes'] = f"Unmapped hierarchical label: {hierarchical_label}"
                
                tickets_classified += 1
        
        self.stage_results['stage_4'] = {
            'description': 'Hierarchical clustering classifications applied',
            'total_hierarchical_tickets': len(df_hierarchical),
            'tickets_classified': tickets_classified,
            'edge_cases_found': edge_cases_found,
            'file': str(hierarchical_file)
        }
        
        self.logger.info(f"   ✅ Applied hierarchical classifications to {tickets_classified:,} tickets")
        self.logger.info(f"   ⚠️ Found {edge_cases_found:,} edge cases requiring manual review")
        
        return df_original
    
    def stage_5_handle_remaining_tickets(self, df_original):
        """STAGE 5: Handle any remaining unclassified tickets."""
        self.logger.info("🔄 STAGE 5: Handling remaining unclassified tickets")
        
        # Find tickets that are still unclassified
        unclassified_mask = df_original['classification_method'] == 'UNCLASSIFIED'
        remaining_tickets = df_original[unclassified_mask]
        
        if len(remaining_tickets) > 0:
            self.logger.warning(f"   ⚠️ Found {len(remaining_tickets):,} tickets still unclassified")
            
            # Apply default classification to remaining tickets
            for idx in remaining_tickets.index:
                original_category = df_original.at[idx, 'Category']
                
                df_original.at[idx, 'classification_method'] = 'DEFAULT_ASSIGNMENT'
                df_original.at[idx, 'final_business_category'] = f"Legacy {original_category}"
                df_original.at[idx, 'final_subcategory'] = 'Legacy Category Assignment'
                df_original.at[idx, 'final_queue'] = f"LEGACY_{original_category.upper().replace(' ', '_')}"
                df_original.at[idx, 'routing_team'] = 'General Support'
                df_original.at[idx, 'priority_level'] = 'Low'
                df_original.at[idx, 'sla_hours'] = 24
                df_original.at[idx, 'confidence_score'] = 0.2
                df_original.at[idx, 'processing_notes'] = f"Default assignment based on original category: {original_category}"
        
        self.stage_results['stage_5'] = {
            'description': 'Remaining tickets handled with default assignments',
            'remaining_tickets': len(remaining_tickets),
            'default_assignments': len(remaining_tickets)
        }
        
        if len(remaining_tickets) > 0:
            self.logger.info(f"   ✅ Applied default classifications to {len(remaining_tickets):,} remaining tickets")
        else:
            self.logger.info(f"   ✅ No remaining tickets - 100% coverage achieved!")
        
        return df_original
    
    def stage_6_inherit_group_classifications(self, df_original):
        """STAGE 6: Inherit classifications from representatives to group members."""
        self.logger.info("🔄 STAGE 6: Inheriting classifications from representatives to group members")
        
        tickets_inherited = 0
        
        # Find all tickets that are in groups but not representatives
        group_members = df_original[
            (df_original['fuzzy_group_id'] != -1) & 
            (df_original.index != df_original['representative_ticket_index'])
        ]
        
        for idx, row in group_members.iterrows():
            representative_idx = int(row['representative_ticket_index'])
            
            if representative_idx in df_original.index:
                # Copy all classification fields from representative
                classification_fields = [
                    'classification_method', 'final_business_category', 'final_subcategory',
                    'final_queue', 'routing_team', 'priority_level', 'sla_hours',
                    'confidence_score', 'is_edge_case'
                ]
                
                for field in classification_fields:
                    df_original.at[idx, field] = df_original.at[representative_idx, field]
                
                # Update processing notes to indicate inheritance
                original_notes = df_original.at[representative_idx, 'processing_notes']
                df_original.at[idx, 'processing_notes'] = f"Inherited from representative #{representative_idx}: {original_notes}"
                
                # Slightly lower confidence for inherited classifications
                current_confidence = df_original.at[idx, 'confidence_score']
                df_original.at[idx, 'confidence_score'] = max(0.1, current_confidence * 0.9)
                
                tickets_inherited += 1
        
        self.stage_results['stage_6'] = {
            'description': 'Classifications inherited from representatives to group members',
            'group_members_found': len(group_members),
            'tickets_inherited': tickets_inherited
        }
        
        self.logger.info(f"   ✅ Inherited classifications for {tickets_inherited:,} group member tickets")
        
        return df_original
    
    def validate_final_dataset(self, df_final):
        """Validate the final consolidated dataset."""
        self.logger.info("🔍 VALIDATING FINAL DATASET")
        
        # Basic validation checks
        total_tickets = len(df_final)
        
        # Check classification coverage
        unclassified_count = len(df_final[df_final['classification_method'] == 'UNCLASSIFIED'])
        classification_coverage = ((total_tickets - unclassified_count) / total_tickets) * 100
        
        # Check method distribution
        method_distribution = df_final['classification_method'].value_counts()
        
        # Check business category distribution
        category_distribution = df_final['final_business_category'].value_counts()
        
        # Check group inheritance
        group_tickets = len(df_final[df_final['fuzzy_group_id'] != -1])
        representatives = len(df_final[df_final.index == df_final['representative_ticket_index']])
        
        # Check edge cases
        edge_cases = len(df_final[df_final['is_edge_case'] == True])
        
        # Check priority distribution
        priority_distribution = df_final['priority_level'].value_counts()
        
        validation_results = {
            'total_tickets': total_tickets,
            'classification_coverage_percent': classification_coverage,
            'unclassified_count': unclassified_count,
            'method_distribution': method_distribution.to_dict(),
            'category_distribution': category_distribution.to_dict(),
            'group_tickets': group_tickets,
            'representative_tickets': representatives,
            'edge_cases': edge_cases,
            'priority_distribution': priority_distribution.to_dict()
        }
        
        # Display validation results
        self.logger.info(f"   📊 Total tickets: {total_tickets:,}")
        self.logger.info(f"   📊 Classification coverage: {classification_coverage:.1f}%")
        self.logger.info(f"   📊 Unclassified: {unclassified_count:,}")
        
        self.logger.info(f"   📊 Classification methods:")
        for method, count in method_distribution.items():
            percentage = (count / total_tickets) * 100
            self.logger.info(f"      {method}: {count:,} ({percentage:.1f}%)")
        
        self.logger.info(f"   📊 Group structure:")
        self.logger.info(f"      Tickets in groups: {group_tickets:,}")
        self.logger.info(f"      Representative tickets: {representatives:,}")
        self.logger.info(f"      Edge cases: {edge_cases:,}")
        
        self.logger.info(f"   📊 Priority distribution:")
        for priority, count in priority_distribution.items():
            percentage = (count / total_tickets) * 100
            self.logger.info(f"      {priority}: {count:,} ({percentage:.1f}%)")
        
        return validation_results
    
    def create_final_report(self, df_final, validation_results):
        """Create comprehensive final consolidation report."""
        self.logger.info("📝 CREATING FINAL CONSOLIDATION REPORT")
        
        # Generate comprehensive report
        report = {
            'consolidation_timestamp': datetime.now().isoformat(),
            'methodology': {
                'description': 'Complete ticket classification system combining multiple approaches',
                'stages': [
                    'Original dataset loading (6,964 tickets)',
                    'Identical group mapping (411 groups, 3,528 tickets)',
                    'Hardcoded rule classification (639 tickets)',
                    'Hierarchical clustering (3,208 tickets)',
                    'Default assignment for remaining tickets',
                    'Classification inheritance for group members'
                ]
            },
            'stage_results': self.stage_results,
            'validation_results': validation_results,
            'business_impact': {
                'total_tickets_processed': len(df_final),
                'automation_achieved': f"{validation_results['classification_coverage_percent']:.1f}%",
                'manual_review_required': validation_results['edge_cases'],
                'ready_for_production': validation_results['unclassified_count'] == 0
            },
            'deployment_structure': {
                'business_categories': len(set(df_final['final_business_category'])),
                'routing_queues': len(set(df_final['final_queue'])),
                'routing_teams': len(set(df_final['routing_team'])),
                'priority_levels': len(set(df_final['priority_level']))
            }
        }
        
        return report

def main():
    """Main final consolidation pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("="*80)
    print("FINAL TICKET CONSOLIDATION - COMPLETE CLASSIFICATION SYSTEM")
    print("="*80)
    print("Consolidating ALL processing stages into final unified classification")
    print("="*80)
    
    # Initialize consolidator
    consolidator = FinalTicketConsolidator()
    
    try:
        # Execute all consolidation stages
        print(f"\n{'='*80}")
        df_final = consolidator.stage_1_load_original_tickets()
        
        print(f"\n{'='*80}")
        df_final = consolidator.stage_2_map_identical_groups(df_final)
        
        print(f"\n{'='*80}")
        df_final = consolidator.stage_3_apply_hardcoded_classifications(df_final)
        
        print(f"\n{'='*80}")
        df_final = consolidator.stage_4_apply_hierarchical_clustering(df_final)
        
        print(f"\n{'='*80}")
        df_final = consolidator.stage_5_handle_remaining_tickets(df_final)
        
        print(f"\n{'='*80}")
        df_final = consolidator.stage_6_inherit_group_classifications(df_final)
        
        print(f"\n{'='*80}")
        validation_results = consolidator.validate_final_dataset(df_final)
        
        print(f"\n{'='*80}")
        final_report = consolidator.create_final_report(df_final, validation_results)
        
        # Save final results
        output_dir = Path('outputs')
        
        # Save final classified dataset
        final_csv_file = output_dir / 'final_complete_ticket_classification.csv'
        df_final.to_csv(final_csv_file, index=True)  # Keep index as original ticket ID
        
        # Save final report
        final_report_file = output_dir / 'final_consolidation_report.json'
        with open(final_report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        # Display final summary
        print(f"\n🎯 FINAL CONSOLIDATION COMPLETE!")
        print(f"   📄 Final dataset: {final_csv_file}")
        print(f"   📄 Consolidation report: {final_report_file}")
        print(f"   📊 Total tickets: {len(df_final):,}")
        print(f"   📊 Classification coverage: {validation_results['classification_coverage_percent']:.1f}%")
        print(f"   📊 Ready for production: {'✅ YES' if final_report['business_impact']['ready_for_production'] else '❌ NO'}")
        
        print(f"\n🚀 PRODUCTION DEPLOYMENT SUMMARY:")
        deployment = final_report['deployment_structure']
        print(f"   🏢 Business categories: {deployment['business_categories']}")
        print(f"   📮 Routing queues: {deployment['routing_queues']}")
        print(f"   👥 Routing teams: {deployment['routing_teams']}")
        print(f"   ⚡ Priority levels: {deployment['priority_levels']}")
        print(f"   ⚠️ Edge cases for review: {validation_results['edge_cases']:,}")
        
        logger.info("Final ticket consolidation completed successfully!")
        
    except Exception as e:
        logger.error(f"Consolidation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()