#!/usr/bin/env python3
"""
Test 5 Unique Groups - Improved Classification Approach

Tests the improved 4-step classification approach with 5 carefully selected,
diverse problem groups to validate the conservative framework.
"""

import pandas as pd
from llm_automation_classifier import LLMAutomationClassifier
import json
from pathlib import Path

def select_10_unique_test_groups():
    """Select 10 carefully chosen diverse problem groups representing full automation spectrum"""
    
    groups_df = pd.read_csv("outputs/improved_problem_groups.csv")
    
    # Manually selected diverse test cases based on analysis
    # Format: (Expected Category, Problem Group ID, Description)
    selected_ids = [
        # FULLY_AUTOMATABLE candidates (3 groups)
        (149, "Account Lock - Fully Auto", "till - cashier locked on till - till 4"),
        (94, "Order Unlock - Fully Auto", "unlock order"), 
        (18, "Password Reset - Fully Auto", "google: password reset for yuliia yunak"),
        
        # PARTIALLY_AUTOMATABLE candidates (4 groups)
        (0, "Hardware Diagnostic - Partial", "till: scanner issue on till 02"),
        (175, "Software Installation - Partial", "printer - installation"),
        (1, "System Error - Partial", "vision - orange bar error"),
        (57, "System Init Failure - Partial", "fusion issue - failed to initialize fusion for launch"),
        
        # NOT_AUTOMATABLE candidates (3 groups)
        (14, "Manual Config - Not Auto", "store manager requests screen adjustment"),
        (73, "Hardware Failure - Not Auto", "project 4 pc - screen flickering constantly"),
        (137, "Physical Hardware - Not Auto", "helpdesk 2 not turning on")
    ]
    
    test_groups = []
    
    for group_id, expected_category, description in selected_ids:
        matching_group = groups_df[groups_df['problem_group_id'] == group_id]
        if not matching_group.empty:
            test_groups.append((expected_category, matching_group.iloc[0]))
        else:
            print(f"Warning: Could not find group {group_id}: {description}")
    
    return test_groups

def run_10_group_test():
    """Run improved classification test on 10 carefully selected diverse groups"""
    
    print("🧪 COMPREHENSIVE CLASSIFICATION TEST - 10 Diverse Groups")
    print("=" * 70)
    print("Testing conservative 4-step framework across full automation spectrum...")
    
    # Select test groups
    test_groups = select_10_unique_test_groups()
    
    print(f"\n📋 Selected 10 diverse test groups:")
    for i, (category, group) in enumerate(test_groups, 1):
        print(f"{i:2d}. {category}")
        print(f"     PG_{group['problem_group_id']}: '{group['representative_short_description']}'")
        print(f"     Size: {group['group_size']} tickets, Quality: {group['quality_score']:.3f}")
        print()
    
    # Initialize improved classifier
    classifier = LLMAutomationClassifier()
    
    # Load problem group data
    problem_groups = classifier.load_problem_groups()
    
    # Filter to test groups
    test_problem_groups = []
    for category, group_row in test_groups:
        matching_pg = next((pg for pg in problem_groups if pg.problem_group_id == group_row['problem_group_id']), None)
        if matching_pg:
            test_problem_groups.append((category, matching_pg))
    
    print(f"🤖 Running COMPREHENSIVE 4-step classification approach...")
    
    # Run classifications
    results = []
    for i, (category, problem_group) in enumerate(test_problem_groups, 1):
        print(f"\n--- Test {i}/10: {category} ---")
        print(f"PG_{problem_group.problem_group_id}: {problem_group.representative_short_description}")
        
        try:
            classification = classifier._classify_single_group(problem_group)
            results.append({
                'expected_type': category,
                'problem_group': problem_group,
                'classification': classification,
                'success': True
            })
            
            print(f"✅ {classification.automation_category} (confidence: {classification.confidence_score:.3f})")
            print(f"   Priority: {classification.business_priority}, Complexity: {classification.implementation_complexity}")
            print(f"   Analysis Steps:")
            print(f"     Account Check: {classification.account_check}")
            print(f"     Hardware: {classification.step_1_hardware_involved}")
            print(f"     Diagnosis: {classification.step_2_requires_diagnosis}")
            print(f"     Command: {classification.step_3_maps_to_command}")
            print(f"     Human Input: {classification.step_4_needs_human_input}")
            
        except Exception as e:
            print(f"❌ Classification failed: {e}")
            results.append({
                'expected_type': category,
                'problem_group': problem_group,
                'classification': None,
                'success': False,
                'error': str(e)
            })
    
    # Save results
    save_test_results(results)
    
    # Analyze results
    analyze_test_results(results)
    
    return results

def save_test_results(results):
    """Save test results for review"""
    
    output_dir = Path("outputs/test_10")
    output_dir.mkdir(exist_ok=True)
    
    # Save successful classifications using standard format
    successful_classifications = [r['classification'] for r in results if r['success']]
    if successful_classifications:
        classifier = LLMAutomationClassifier()
        classifier.save_results(successful_classifications, output_dir="outputs/test_10")
    
    # Save detailed test analysis
    test_analysis = []
    for result in results:
        if result['success']:
            cls = result['classification']
            test_analysis.append({
                'expected_type': result['expected_type'],
                'problem_group_id': result['problem_group'].problem_group_id,
                'description': result['problem_group'].representative_short_description,
                'classification': cls.automation_category,
                'confidence': cls.confidence_score,
                'account_check': cls.account_check,
                'step_1_hardware': cls.step_1_hardware_involved,
                'step_2_diagnosis': cls.step_2_requires_diagnosis,
                'step_3_command': cls.step_3_maps_to_command,
                'step_4_human': cls.step_4_needs_human_input,
                'reasoning': cls.reasoning
            })
        else:
            test_analysis.append({
                'expected_type': result['expected_type'],
                'problem_group_id': result['problem_group'].problem_group_id,
                'description': result['problem_group'].representative_short_description,
                'classification': 'ERROR',
                'error': result.get('error', 'Unknown error')
            })
    
    with open(output_dir / "test_10_analysis.json", 'w') as f:
        json.dump(test_analysis, f, indent=2)

def analyze_test_results(results):
    """Analyze and display test results"""
    
    print(f"\n" + "=" * 60)
    print("📊 10-GROUP TEST ANALYSIS")
    print("=" * 60)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"✅ Successful classifications: {len(successful)}/10")
    print(f"❌ Failed classifications: {len(failed)}/10")
    
    if failed:
        print(f"\n⚠️  FAILURES:")
        for result in failed:
            print(f"  - {result['expected_type']}: {result.get('error', 'Unknown error')}")
    
    if successful:
        print(f"\n🎯 CLASSIFICATION BREAKDOWN:")
        categories = {}
        confidences = []
        
        for result in successful:
            cls = result['classification']
            cat = cls.automation_category
            categories[cat] = categories.get(cat, 0) + 1
            confidences.append(cls.confidence_score)
        
        for cat, count in categories.items():
            print(f"  {cat}: {count} groups")
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        print(f"  Average confidence: {avg_confidence:.3f}")
        
        print(f"\n🔍 DETAILED RESULTS:")
        for result in successful:
            pg = result['problem_group']
            cls = result['classification']
            expected = result['expected_type']
            
            print(f"\n  {expected} → {cls.automation_category}")
            print(f"    PG_{pg.problem_group_id}: {pg.representative_short_description}")
            print(f"    Confidence: {cls.confidence_score:.3f}")
            print(f"    Analysis: ACCT={cls.account_check.split(' - ')[0]} | "
                  f"HW={cls.step_1_hardware_involved.split(' - ')[0]} | "
                  f"DIAG={cls.step_2_requires_diagnosis.split(' - ')[0]} | "
                  f"CMD={cls.step_3_maps_to_command.split(' - ')[0]}")
        
        # Check for improvement signs
        print(f"\n✅ IMPROVEMENT INDICATORS:")
        
        # More diverse classifications?
        if len(categories) > 1:
            print("  ✓ Multiple classification types (good diversity)")
        else:
            print("  ⚠ All groups classified the same (may still be over-automating)")
        
        # Confidence variation?
        conf_range = max(confidences) - min(confidences) if len(confidences) > 1 else 0
        if conf_range > 0.2:
            print("  ✓ Confidence scores vary (good discrimination)")
        else:
            print("  ⚠ Confidence scores very similar (may need calibration)")
        
        # Hardware recognition?
        hardware_recognized = any("YES" in result['classification'].step_1_hardware_involved.upper() 
                                for result in successful)
        if hardware_recognized:
            print("  ✓ Hardware issues identified (improvement over previous)")
        else:
            print("  ⚠ No hardware issues identified (check test cases)")
    
    print(f"\n📋 NEXT STEPS:")
    if len(successful) >= 8:
        print("  → Good success rate. Review detailed results for quality.")
        print("  → If classifications look reasonable, proceed to full 209-group analysis.")
    else:
        print("  → High failure rate. Check prompt engineering and error handling.")
    
    print(f"\n📁 Detailed results saved in: outputs/test_10/")

if __name__ == "__main__":
    try:
        results = run_10_group_test()
        
        success_count = len([r for r in results if r['success']])
        print(f"\n🎉 10-group test complete: {success_count}/10 successful")
        
        if success_count >= 8:
            print("✅ Ready to proceed to full 209-group analysis if results look good!")
        else:
            print("⚠️  Review failures before proceeding")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()