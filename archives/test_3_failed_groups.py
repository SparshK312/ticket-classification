#!/usr/bin/env python3
"""
Test 3 Failed Groups - Focus on NOT_AUTOMATABLE Classification

Tests only the 3 groups that were misclassified as PARTIALLY_AUTOMATABLE
when they should have been NOT_AUTOMATABLE:
- PG_14: "store manager requests screen adjustment" 
- PG_73: "project 4 pc - screen flickering constantly"
- PG_137: "helpdesk 2 not turning on"
"""

import pandas as pd
from llm_automation_classifier import LLMAutomationClassifier
import json
from pathlib import Path

def select_failed_test_groups():
    """Select the 3 specific groups that failed NOT_AUTOMATABLE classification"""
    
    groups_df = pd.read_csv("outputs/improved_problem_groups.csv")
    
    # The 3 specific groups that were misclassified
    failed_group_ids = [
        (14, "Manual Config - Should be NOT_AUTOMATABLE", "store manager requests screen adjustment"),
        (73, "Hardware Failure - Should be NOT_AUTOMATABLE", "project 4 pc - screen flickering constantly"),
        (137, "Physical Hardware - Should be NOT_AUTOMATABLE", "helpdesk 2 not turning on")
    ]
    
    test_groups = []
    
    for group_id, expected_category, description in failed_group_ids:
        matching_group = groups_df[groups_df['problem_group_id'] == group_id]
        if not matching_group.empty:
            test_groups.append((expected_category, matching_group.iloc[0]))
        else:
            print(f"Warning: Could not find group {group_id}: {description}")
    
    return test_groups

def run_focused_test():
    """Run focused test on the 3 specific failed groups"""
    
    print("🎯 FOCUSED TEST - 3 Failed NOT_AUTOMATABLE Groups")
    print("=" * 60)
    print("Testing improved classification with enhanced NOT_AUTOMATABLE logic...")
    
    # Select failed groups
    test_groups = select_failed_test_groups()
    
    print(f"\n📋 Testing the 3 groups that were misclassified:")
    for i, (category, group) in enumerate(test_groups, 1):
        print(f"{i}. {category}")
        print(f"   PG_{group['problem_group_id']}: '{group['representative_short_description']}'")
        print(f"   Size: {group['group_size']} tickets, Quality: {group['quality_score']:.3f}")
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
    
    print(f"🤖 Running ENHANCED 5-step classification approach...")
    print("(New logic: Hardware FAILURE + Management REQUEST → NOT_AUTOMATABLE)")
    
    # Run classifications
    results = []
    for i, (category, problem_group) in enumerate(test_problem_groups, 1):
        print(f"\n--- Test {i}/3: {category} ---")
        print(f"PG_{problem_group.problem_group_id}: {problem_group.representative_short_description}")
        
        try:
            classification = classifier._classify_single_group(problem_group)
            results.append({
                'expected_type': category,
                'problem_group': problem_group,
                'classification': classification,
                'success': True
            })
            
            # Show result
            actual_category = classification.automation_category
            if "NOT_AUTOMATABLE" in category and actual_category == "NOT_AUTOMATABLE":
                status = "✅ FIXED"
            elif "NOT_AUTOMATABLE" in category and actual_category != "NOT_AUTOMATABLE":
                status = "❌ STILL WRONG"
            else:
                status = "✅"
                
            print(f"{status} {actual_category} (confidence: {classification.confidence_score:.3f})")
            print(f"   Priority: {classification.business_priority}, Complexity: {classification.implementation_complexity}")
            print(f"   Enhanced Analysis Steps:")
            print(f"     Account Check: {classification.account_check}")
            print(f"     Hardware: {classification.step_1_hardware_involved}")
            print(f"     Management: {classification.step_2_management_request}")
            print(f"     Diagnosis: {classification.step_3_requires_diagnosis}")
            print(f"     Command: {classification.step_4_maps_to_command}")
            print(f"     Human Input: {classification.step_5_needs_human_input}")
            
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
    save_focused_results(results)
    
    # Analyze results
    analyze_focused_results(results)
    
    return results

def save_focused_results(results):
    """Save focused test results for review"""
    
    output_dir = Path("outputs/test_3_failed")
    output_dir.mkdir(exist_ok=True)
    
    # Save successful classifications using standard format
    successful_classifications = [r['classification'] for r in results if r['success']]
    if successful_classifications:
        classifier = LLMAutomationClassifier()
        classifier.save_results(successful_classifications, output_dir="outputs/test_3_failed")
    
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
                'step_2_management': cls.step_2_management_request,
                'step_3_diagnosis': cls.step_3_requires_diagnosis,
                'step_4_command': cls.step_4_maps_to_command,
                'step_5_human': cls.step_5_needs_human_input,
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
    
    with open(output_dir / "focused_test_analysis.json", 'w') as f:
        json.dump(test_analysis, f, indent=2)

def analyze_focused_results(results):
    """Analyze and display focused test results"""
    
    print(f"\n" + "=" * 60)
    print("📊 FOCUSED TEST ANALYSIS - NOT_AUTOMATABLE FIXES")
    print("=" * 60)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"✅ Successful classifications: {len(successful)}/3")
    print(f"❌ Failed classifications: {len(failed)}/3")
    
    if failed:
        print(f"\n⚠️  FAILURES:")
        for result in failed:
            print(f"  - {result['expected_type']}: {result.get('error', 'Unknown error')}")
    
    if successful:
        print(f"\n🎯 CLASSIFICATION RESULTS:")
        
        fixed_count = 0
        still_wrong_count = 0
        
        for result in successful:
            cls = result['classification']
            expected = result['expected_type']
            pg = result['problem_group']
            
            # Check if classification was fixed
            if "NOT_AUTOMATABLE" in expected and cls.automation_category == "NOT_AUTOMATABLE":
                status = "✅ FIXED"
                fixed_count += 1
            elif "NOT_AUTOMATABLE" in expected and cls.automation_category != "NOT_AUTOMATABLE":
                status = "❌ STILL WRONG"
                still_wrong_count += 1
            else:
                status = "✅ OK"
            
            print(f"\n  {status}: {expected}")
            print(f"    PG_{pg.problem_group_id}: {pg.representative_short_description}")
            print(f"    ACTUAL: {cls.automation_category} (confidence: {cls.confidence_score:.3f})")
            print(f"    Hardware: {cls.step_1_hardware_involved}")
            print(f"    Management: {cls.step_2_management_request}")
            print(f"    Reasoning: {cls.reasoning[:150]}...")
        
        print(f"\n📈 FIX SUCCESS RATE:")
        print(f"  ✅ Fixed classifications: {fixed_count}/3")
        print(f"  ❌ Still wrong: {still_wrong_count}/3")
        print(f"  🎯 Fix success rate: {(fixed_count/3)*100:.1f}%")
        
        if fixed_count == 3:
            print(f"\n🎉 ALL CLASSIFICATIONS FIXED!")
            print(f"✅ Ready to proceed to full 209-group analysis")
        elif fixed_count >= 2:
            print(f"\n🔧 MOSTLY FIXED - good progress")
            print(f"⚠️  Review remaining {still_wrong_count} issue(s)")
        else:
            print(f"\n❌ CLASSIFICATION LOGIC STILL NEEDS WORK")
            print(f"🔧 Review prompt engineering and validation logic")
    
    print(f"\n📁 Detailed results saved in: outputs/test_3_failed/")

if __name__ == "__main__":
    try:
        results = run_focused_test()
        
        success_count = len([r for r in results if r['success']])
        fixed_count = len([r for r in results if r['success'] and 
                          "NOT_AUTOMATABLE" in r['expected_type'] and 
                          r['classification'].automation_category == "NOT_AUTOMATABLE"])
        
        print(f"\n🎉 Focused test complete: {success_count}/3 successful, {fixed_count}/3 fixed")
        
        if fixed_count == 3:
            print("✅ All NOT_AUTOMATABLE classifications fixed! Ready for full analysis.")
        else:
            print("⚠️  Some classifications still need adjustment.")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()