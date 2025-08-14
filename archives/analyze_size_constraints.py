#!/usr/bin/env python3
"""
ANALYZE SIZE CONSTRAINTS - ARE WE OVER-SPLITTING?

This script examines whether the 40-ticket hard cap is appropriate
or if we're artificially splitting semantically coherent clusters.

Analysis approach:
1. Examine the largest clusters (>40 tickets)
2. Check their quality scores vs smaller clusters
3. Analyze semantic coherence within large clusters
4. Determine if adaptive sizing would be better
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter

def analyze_size_quality_relationship():
    """Analyze the relationship between cluster size and quality."""
    
    print("="*80)
    print("SIZE CONSTRAINT ANALYSIS - IS 40-TICKET CAP TOO RESTRICTIVE?")
    print("="*80)
    
    # Load the improved groups
    groups_df = pd.read_csv('outputs/improved_problem_groups.csv')
    
    # Group by size ranges
    size_ranges = [
        ("Very Large (100+)", groups_df[groups_df['group_size'] >= 100]),
        ("Large (50-99)", groups_df[(groups_df['group_size'] >= 50) & (groups_df['group_size'] < 100)]),
        ("Medium (20-49)", groups_df[(groups_df['group_size'] >= 20) & (groups_df['group_size'] < 50)]),
        ("Small (10-19)", groups_df[(groups_df['group_size'] >= 10) & (groups_df['group_size'] < 20)]),
        ("Very Small (4-9)", groups_df[(groups_df['group_size'] >= 4) & (groups_df['group_size'] < 10)]),
        ("Tiny (1-3)", groups_df[groups_df['group_size'] < 4])
    ]
    
    print(f"\n📊 SIZE vs QUALITY ANALYSIS:")
    print(f"{'Size Range':<20} {'Count':<8} {'Avg Quality':<12} {'Min Quality':<12} {'Max Quality':<12}")
    print("-" * 75)
    
    for range_name, subset in size_ranges:
        if len(subset) > 0:
            count = len(subset)
            avg_quality = subset['quality_score'].mean()
            min_quality = subset['quality_score'].min()
            max_quality = subset['quality_score'].max()
            
            print(f"{range_name:<20} {count:<8} {avg_quality:<12.3f} {min_quality:<12.3f} {max_quality:<12.3f}")
    
    return groups_df

def examine_large_clusters(groups_df):
    """Examine the largest clusters in detail."""
    
    print(f"\n🔍 DETAILED ANALYSIS OF LARGE CLUSTERS (>40 tickets):")
    
    large_clusters = groups_df[groups_df['group_size'] > 40].sort_values('group_size', ascending=False)
    
    if len(large_clusters) == 0:
        print("   No clusters exceed 40 tickets - size constraint is working as intended")
        return
    
    print(f"\n{'ID':<5} {'Size':<6} {'Quality':<8} {'Origin':<20} {'Representative Problem':<50}")
    print("-" * 100)
    
    for _, cluster in large_clusters.head(10).iterrows():
        cluster_id = cluster['problem_group_id']
        size = cluster['group_size']
        quality = cluster['quality_score']
        origin = cluster['cluster_origin']
        problem = cluster['representative_short_description'][:47] + "..." if len(cluster['representative_short_description']) > 50 else cluster['representative_short_description']
        
        print(f"{cluster_id:<5} {size:<6} {quality:<8.3f} {origin:<20} {problem:<50}")
    
    # Detailed analysis of top 3 largest
    print(f"\n📋 SEMANTIC COHERENCE ANALYSIS:")
    
    for i, (_, cluster) in enumerate(large_clusters.head(3).iterrows()):
        cluster_id = cluster['problem_group_id']
        size = cluster['group_size']
        quality = cluster['quality_score']
        problem = cluster['representative_short_description']
        
        print(f"\n{'='*60}")
        print(f"CLUSTER #{cluster_id} - Size: {size}, Quality: {quality:.3f}")
        print(f"Problem: {problem}")
        print(f"{'='*60}")
        
        # Analyze semantic coherence
        analyze_cluster_coherence(cluster_id, problem, size, quality)

def analyze_cluster_coherence(cluster_id, problem, size, quality):
    """Analyze semantic coherence of a specific cluster."""
    
    # Load details for this cluster
    details_df = pd.read_csv('outputs/improved_problem_group_details.csv')
    cluster_details = details_df[details_df['problem_group_id'] == cluster_id]
    
    print(f"\n🔍 SEMANTIC ANALYSIS:")
    
    # Basic coherence indicators
    if "hardware" in problem.lower() and "monitor" in problem.lower():
        print(f"   📱 Hardware/Monitor cluster - likely includes diverse hardware issues")
        print(f"   ⚠️  May be a catch-all cluster for different hardware problems")
        coherence_assessment = "POTENTIALLY_INCOHERENT"
        
    elif "till" in problem.lower() and "card payments" in problem.lower():
        print(f"   💳 Till payment cluster - specific POS payment issues")
        print(f"   ✅ Likely coherent - all related to card payment failures")
        coherence_assessment = "COHERENT"
        
    elif "chip" in problem.lower() and "pin" in problem.lower():
        print(f"   💳 Chip & PIN cluster - specific payment method issues")
        print(f"   ✅ Likely coherent - all related to chip & pin failures")
        coherence_assessment = "COHERENT"
        
    elif "proj" in problem.lower() or "ped" in problem.lower():
        print(f"   🔌 PED/Project cluster - payment device connectivity")
        print(f"   ✅ Likely coherent - all related to PED device issues")
        coherence_assessment = "COHERENT"
        
    elif "vision" in problem.lower():
        print(f"   📦 Vision system cluster - ordering system issues")
        print(f"   ✅ Likely coherent - all related to Vision ordering system")
        coherence_assessment = "COHERENT"
        
    else:
        print(f"   🤔 Mixed/unclear cluster - needs manual review")
        coherence_assessment = "UNCLEAR"
    
    # Quality-based assessment
    if quality >= 0.6:
        quality_assessment = "HIGH_QUALITY"
        print(f"   📈 High quality score ({quality:.3f}) suggests good semantic coherence")
    elif quality >= 0.4:
        quality_assessment = "MEDIUM_QUALITY"
        print(f"   📊 Medium quality score ({quality:.3f}) suggests moderate coherence")
    else:
        quality_assessment = "LOW_QUALITY"
        print(f"   📉 Low quality score ({quality:.3f}) suggests poor coherence")
    
    # Size assessment
    if size > 100:
        print(f"   🚨 Very large size ({size} tickets) - likely needs splitting")
        size_assessment = "TOO_LARGE"
    elif size > 60:
        print(f"   ⚠️  Large size ({size} tickets) - may need review")
        size_assessment = "LARGE"
    else:
        print(f"   ✅ Reasonable size ({size} tickets) for a coherent problem")
        size_assessment = "REASONABLE"
    
    # Overall recommendation
    print(f"\n💡 RECOMMENDATION:")
    
    if coherence_assessment == "COHERENT" and quality_assessment == "HIGH_QUALITY":
        print(f"   ✅ Keep as-is: High quality and semantically coherent")
        print(f"   📈 Consider raising size limit for this type of problem")
        
    elif coherence_assessment == "COHERENT" and quality_assessment == "MEDIUM_QUALITY":
        print(f"   🔄 Monitor: Coherent but quality could be improved")
        print(f"   📊 Acceptable for production use")
        
    elif coherence_assessment == "POTENTIALLY_INCOHERENT" or quality_assessment == "LOW_QUALITY":
        print(f"   🔧 Needs splitting: Likely contains multiple problem types")
        print(f"   📉 Size constraint working correctly")
        
    else:
        print(f"   👀 Manual review needed: Unclear coherence")

def calculate_adaptive_sizing_impact():
    """Calculate what would happen with adaptive sizing."""
    
    print(f"\n{'='*80}")
    print(f"ADAPTIVE SIZING ANALYSIS - QUALITY-BASED SIZE LIMITS")
    print(f"{'='*80}")
    
    groups_df = pd.read_csv('outputs/improved_problem_groups.csv')
    
    # Current approach: Hard 40-ticket limit
    current_oversized = len(groups_df[groups_df['group_size'] > 40])
    current_avg_quality = groups_df['quality_score'].mean()
    
    print(f"\n📊 CURRENT APPROACH (Hard 40-ticket limit):")
    print(f"   Clusters exceeding limit: {current_oversized}")
    print(f"   Average quality score: {current_avg_quality:.3f}")
    
    # Proposed adaptive approach: Size limit based on quality
    print(f"\n🔄 PROPOSED ADAPTIVE APPROACH:")
    print(f"   High quality (≥0.7): Allow up to 80 tickets")
    print(f"   Medium quality (0.5-0.7): Allow up to 50 tickets") 
    print(f"   Low quality (<0.5): Limit to 30 tickets")
    
    # Calculate impact
    high_quality = groups_df[groups_df['quality_score'] >= 0.7]
    medium_quality = groups_df[(groups_df['quality_score'] >= 0.5) & (groups_df['quality_score'] < 0.7)]
    low_quality = groups_df[groups_df['quality_score'] < 0.5]
    
    adaptive_violations = 0
    adaptive_violations += len(high_quality[high_quality['group_size'] > 80])
    adaptive_violations += len(medium_quality[medium_quality['group_size'] > 50])
    adaptive_violations += len(low_quality[low_quality['group_size'] > 30])
    
    print(f"\n📈 ADAPTIVE SIZING IMPACT:")
    print(f"   High quality clusters: {len(high_quality)} (max size: {high_quality['group_size'].max() if len(high_quality) > 0 else 0})")
    print(f"   Medium quality clusters: {len(medium_quality)} (max size: {medium_quality['group_size'].max() if len(medium_quality) > 0 else 0})")
    print(f"   Low quality clusters: {len(low_quality)} (max size: {low_quality['group_size'].max() if len(low_quality) > 0 else 0})")
    print(f"   Violations under adaptive approach: {adaptive_violations}")
    
    # Recommendation
    print(f"\n🎯 RECOMMENDATION:")
    
    if adaptive_violations < current_oversized:
        print(f"   ✅ IMPLEMENT ADAPTIVE SIZING")
        print(f"   📈 Reduces violations from {current_oversized} to {adaptive_violations}")
        print(f"   🎯 Allows high-quality coherent clusters to grow naturally")
        
    else:
        print(f"   ❌ KEEP CURRENT HARD LIMIT")
        print(f"   📊 Adaptive approach doesn't improve the situation")
        print(f"   🔒 Current 40-ticket limit is appropriate")

def main():
    """Run size constraint analysis."""
    
    # Load and analyze
    groups_df = analyze_size_quality_relationship()
    
    # Examine large clusters
    examine_large_clusters(groups_df)
    
    # Calculate adaptive sizing impact
    calculate_adaptive_sizing_impact()
    
    print(f"\n{'='*80}")
    print(f"CONCLUSION")
    print(f"{'='*80}")
    
    print(f"\nThe analysis reveals whether our 40-ticket hard cap is:")
    print(f"1. 🎯 Appropriately preventing garbage clusters")
    print(f"2. ⚠️  Over-splitting semantically coherent problems")
    print(f"3. 🔄 Could benefit from adaptive quality-based sizing")
    
    print(f"\nBased on the quality scores and semantic analysis above,")
    print(f"we can determine if the pipeline needs refinement or is")
    print(f"ready for production use as-is.")

if __name__ == "__main__":
    main() 