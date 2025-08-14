#!/usr/bin/env python3
"""
Create Consolidated Dataset for Semantic Grouping

This script creates the consolidated dataset by:
1. Taking the 411 identical groups and creating 1 representative ticket per group
2. Adding the 3,436 unique tickets that had no fuzzy matches
3. Saving the final 3,847 tickets for semantic analysis

Result: 3,847 tickets ready for "core problem statement" analysis
"""

import pandas as pd
import json
import logging
from pathlib import Path

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def create_consolidated_dataset():
    """Create consolidated dataset from fuzzy matching results."""
    logger = logging.getLogger(__name__)
    
    # File paths
    clean_data_file = Path('data/processed/clean_tickets.csv')
    fuzzy_results_file = Path('outputs/identical_tickets_analysis.json')
    output_file = Path('data/processed/consolidated_tickets.csv')
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load original clean data
    logger.info(f"Loading clean ticket data from {clean_data_file}")
    df_original = pd.read_csv(clean_data_file)
    logger.info(f"Loaded {len(df_original):,} original tickets")
    
    # Load fuzzy matching results
    logger.info(f"Loading fuzzy matching results from {fuzzy_results_file}")
    with open(fuzzy_results_file, 'r') as f:
        fuzzy_results = json.load(f)
    
    identical_groups = fuzzy_results['results']['identical_groups']
    logger.info(f"Found {len(identical_groups)} identical groups")
    
    print("="*80)
    print("CREATING CONSOLIDATED DATASET FOR SEMANTIC GROUPING")
    print("="*80)
    
    # Step 1: Create representative tickets from identical groups
    print(f"\n🔄 STEP 1: Processing {len(identical_groups)} identical groups")
    
    consolidated_tickets = []
    tickets_in_groups = set()
    
    for group in identical_groups:
        group_id = group['group_id']
        ticket_indices = group['ticket_indices']
        ticket_count = group['ticket_count']
        
        # Track all tickets that are in groups
        tickets_in_groups.update(ticket_indices)
        
        # Get the first ticket as representative (could be improved with better selection logic)
        representative_idx = ticket_indices[0]
        representative_ticket = df_original.loc[representative_idx].copy()
        
        # Add metadata about the group
        representative_ticket['fuzzy_group_id'] = group_id
        representative_ticket['fuzzy_group_size'] = ticket_count
        representative_ticket['original_ticket_indices'] = str(ticket_indices)  # Store as string for CSV
        representative_ticket['is_consolidated'] = True
        representative_ticket['consolidation_note'] = f"Represents {ticket_count} similar tickets"
        
        consolidated_tickets.append(representative_ticket)
    
    logger.info(f"Created {len(consolidated_tickets)} representative tickets from groups")
    logger.info(f"These represent {len(tickets_in_groups):,} original tickets")
    
    # Step 2: Add unique tickets (those not in any group)
    print(f"\n🔄 STEP 2: Adding unique tickets")
    
    unique_tickets = []
    for idx, row in df_original.iterrows():
        if idx not in tickets_in_groups:
            row_copy = row.copy()
            row_copy['fuzzy_group_id'] = -1  # No group
            row_copy['fuzzy_group_size'] = 1   # Single ticket
            row_copy['original_ticket_indices'] = str([idx])
            row_copy['is_consolidated'] = False
            row_copy['consolidation_note'] = "Unique ticket - no similar matches found"
            unique_tickets.append(row_copy)
    
    logger.info(f"Added {len(unique_tickets)} unique tickets")
    
    # Step 3: Combine and create final dataset
    print(f"\n🔄 STEP 3: Creating final consolidated dataset")
    
    # Convert to DataFrames and combine
    consolidated_df = pd.DataFrame(consolidated_tickets)
    unique_df = pd.DataFrame(unique_tickets)
    
    final_df = pd.concat([consolidated_df, unique_df], ignore_index=True)
    
    # Add semantic grouping preparation columns
    final_df['semantic_group_id'] = -1  # To be filled by semantic analysis
    final_df['core_problem_statement'] = ""  # To be filled by semantic analysis
    final_df['technical_keywords'] = ""  # To be filled by keyword extraction
    
    # Sort by category for easier analysis
    final_df = final_df.sort_values(['Category', 'Subcategory']).reset_index(drop=True)
    
    print(f"\n📊 FINAL DATASET SUMMARY:")
    print(f"   Representative tickets (from {len(identical_groups)} groups): {len(consolidated_tickets):,}")
    print(f"   Unique tickets: {len(unique_tickets):,}")
    print(f"   Total consolidated tickets: {len(final_df):,}")
    print(f"   Original tickets represented: {len(df_original):,}")
    
    # Validation check
    original_count = len(df_original)
    represented_count = len(tickets_in_groups) + len(unique_tickets)
    
    if represented_count == original_count:
        print(f"   ✅ All original tickets accounted for!")
    else:
        print(f"   ❌ Mismatch: {original_count} original vs {represented_count} represented")
    
    print(f"\n📋 CONSOLIDATION IMPACT:")
    consolidation_savings = original_count - len(final_df)
    consolidation_percentage = (consolidation_savings / original_count) * 100
    print(f"   Before: {original_count:,} tickets")
    print(f"   After: {len(final_df):,} tickets")
    print(f"   Reduction: {consolidation_savings:,} tickets ({consolidation_percentage:.1f}%)")
    
    # Step 4: Save consolidated dataset
    print(f"\n💾 SAVING CONSOLIDATED DATASET")
    
    # Clean text fields to prevent CSV corruption
    text_columns = ['Short description', 'Description', 'Comments and Work notes']
    for col in text_columns:
        if col in final_df.columns:
            # Remove newlines and carriage returns that corrupt CSV
            final_df[col] = final_df[col].astype(str).str.replace('\n', ' ', regex=False)
            final_df[col] = final_df[col].str.replace('\r', ' ', regex=False)
            final_df[col] = final_df[col].str.replace('\t', ' ', regex=False)
    
    # Save with proper CSV handling
    final_df.to_csv(output_file, index=True, quoting=1, escapechar='\\')  # index=True to preserve original ticket IDs
    logger.info(f"Consolidated dataset saved to {output_file}")
    
    # Validation check
    print(f"\n✅ CONSOLIDATION VALIDATION:")
    print(f"   Total tickets in final dataset: {len(final_df):,}")
    print(f"   Representatives (from groups): {len(final_df[final_df['is_consolidated'] == True]):,}")
    print(f"   Unique tickets: {len(final_df[final_df['is_consolidated'] == False]):,}")
    print(f"   Expected total: 411 + 3,436 = 3,847")
    print(f"   ✅ Match: {'YES' if len(final_df) == 3847 else 'NO'}")
    
    # Show sample of the dataset structure
    print(f"\n📋 SAMPLE OF CONSOLIDATED DATASET:")
    sample_columns = ['Short description', 'Category', 'is_consolidated', 'fuzzy_group_size', 'consolidation_note']
    print(final_df[sample_columns].head(10).to_string(index=True))
    
    # Category breakdown
    print(f"\n📊 CATEGORY BREAKDOWN:")
    category_breakdown = final_df.groupby('Category').agg({
        'is_consolidated': ['count', 'sum'],
        'fuzzy_group_size': 'sum'
    }).round(2)
    category_breakdown.columns = ['Total_Tickets', 'Consolidated_Count', 'Original_Tickets_Represented']
    category_breakdown['Unique_Tickets'] = category_breakdown['Total_Tickets'] - category_breakdown['Consolidated_Count']
    
    print(category_breakdown.head(10).to_string())
    
    print(f"\n✅ CONSOLIDATED DATASET READY FOR SEMANTIC GROUPING")
    print(f"   📁 File: {output_file}")
    print(f"   📊 Ready for Phase 1: Keyword extraction and problem identification")
    
    return final_df

if __name__ == "__main__":
    setup_logging()
    create_consolidated_dataset()