#!/usr/bin/env python3
"""
STEP 1: Identical Ticket Detection Script

Implements fuzzy string matching to find tickets that are essentially identical
except for minor details like store numbers, user names, error codes.

This follows the manager's specific requirements for ticket grouping.
"""

import json
import logging
import pandas as pd
from pathlib import Path
from src.grouping import TicketGrouper

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('identical_detection.log')
        ]
    )

def main():
    """Main identical ticket detection pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # File paths
    input_file = Path('data/processed/clean_tickets.csv')
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    if not input_file.exists():
        print(f"❌ Error: Clean data file not found at {input_file}")
        print("   Please run process_tickets.py first to generate clean data.")
        return
    
    logger.info("Starting STEP 1: Identical Ticket Detection")
    
    # Load clean data
    print(f"📂 Loading clean data from {input_file}")
    df = pd.read_csv(input_file)
    print(f"   Loaded {len(df):,} tickets")
    
    # Initialize grouper
    grouper = TicketGrouper()
    print(f"   Using {grouper.similarity_threshold}% similarity threshold")
    
    print("\n" + "="*80)
    print("STEP 1: FUZZY STRING MATCHING FOR IDENTICAL TICKETS")
    print("="*80)
    
    # Step 1: Find identical tickets
    print("\n🔍 PHASE 1: IDENTIFYING IDENTICAL TICKETS")
    results = grouper.find_identical_tickets(df)
    
    # Step 2: Analyze high-frequency patterns
    print("\n🔍 PHASE 2: ANALYZING HIGH-FREQUENCY PATTERNS")
    pattern_analysis = grouper.analyze_high_frequency_patterns(df)
    
    print(f"\n📋 HIGH-FREQUENCY PATTERN ANALYSIS:")
    for pattern, analysis in pattern_analysis.items():
        print(f"\n   '{pattern.upper()}':")
        print(f"      Exact matches: {analysis['exact_matches']:,}")
        print(f"      Unique variations: {analysis['variations_found']}")
        print(f"      Consolidation potential: {analysis['consolidation_potential']:,}")
        
        if analysis['sample_descriptions']:
            print(f"      Sample variations:")
            for desc in analysis['sample_descriptions'][:3]:
                print(f"         - '{desc}'")
    
    # Step 3: Generate comprehensive report
    print("\n🔍 PHASE 3: GENERATING CONSOLIDATION REPORT")
    report = grouper.generate_consolidation_report(results)
    print(report)
    
    # Step 4: Save detailed results
    print("\n" + "="*80)
    print("SAVING DETAILED RESULTS")
    print("="*80)
    
    # Save identical groups analysis
    identical_results = {
        'analysis_timestamp': pd.Timestamp.now().isoformat(),
        'methodology': {
            'step': 'STEP 1 - Identical Ticket Detection',
            'technique': 'Fuzzy String Matching',
            'similarity_threshold': grouper.similarity_threshold,
            'focus_field': 'Short description',
            'normalization_patterns': [
                'Store numbers → [STORE]',
                'Generic numbers → [NUMBER]', 
                'Error codes → [CODE]',
                'Item numbers → [ITEM_NUM]',
                'Email addresses → [EMAIL]',
                'Names → [NAME]'
            ]
        },
        'results': results,
        'high_frequency_patterns': pattern_analysis
    }
    
    # Save JSON results
    json_file = output_dir / 'identical_tickets_analysis.json'
    with open(json_file, 'w') as f:
        json.dump(identical_results, f, indent=2, default=str)
    print(f"   📄 Detailed analysis saved to: {json_file}")
    
    # Save consolidation report
    report_file = output_dir / 'consolidation_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"   📄 Consolidation report saved to: {report_file}")
    
    # Save group details for manual review
    if results['identical_groups']:
        groups_df = []
        for group in results['identical_groups']:
            for ticket_idx in group['ticket_indices']:
                groups_df.append({
                    'ticket_index': ticket_idx,
                    'group_id': group['group_id'],
                    'group_size': group['ticket_count'],
                    'similarity_type': group['similarity_type'],
                    'representative_desc': group['representative_description'],
                    'original_desc': df.loc[ticket_idx, 'Short description'],
                    'normalized_desc': group['normalized_description']
                })
        
        groups_df = pd.DataFrame(groups_df)
        groups_file = output_dir / 'identical_ticket_groups.csv'
        groups_df.to_csv(groups_file, index=False)
        print(f"   📄 Group assignments saved to: {groups_file}")
    
    # Final summary
    summary = results['summary']
    print(f"\n✅ STEP 1 COMPLETE: IDENTICAL TICKET DETECTION")
    print(f"   📊 Found {summary['total_identical_groups']:,} identical groups")
    print(f"   🎯 Covering {summary['tickets_in_groups']:,} tickets")
    print(f"   💡 Consolidation potential: {summary['consolidation_potential']:,} tickets ({summary['consolidation_percentage']}%)")
    print(f"   📁 Results saved to: {output_dir}")
    
    # Next steps recommendation
    print(f"\n🔄 NEXT STEPS:")
    print(f"   1. Review the consolidation report for business validation")
    print(f"   2. Examine the largest identical groups for accuracy")
    print(f"   3. Proceed to STEP 2: Semantic grouping of remaining unique tickets")
    
    logger.info("STEP 1: Identical ticket detection completed successfully!")

if __name__ == "__main__":
    main()