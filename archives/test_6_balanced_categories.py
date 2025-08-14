#!/usr/bin/env python3
"""
Focused Test: 5 Manually Selected Problem Groups
Test diverse examples that clearly represent different automation categories
"""

import sys
from pathlib import Path
from llm_automation_classifier import LLMAutomationClassifier, ProblemGroup
import pandas as pd
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_manual_selection_test():
    """Test 6 manually selected problem groups (2 per automation category)"""
    
    print("🎯 MANUAL SELECTION TEST - 6 Balanced Automation Examples")
    print("=" * 70)
    print("Testing manually selected groups: 2 FULLY, 2 PARTIALLY, 2 NOT_AUTOMATABLE...")
    print()
    
    # Manually selected test cases with expected classifications (2 per category)
    test_cases = [
        # FULLY_AUTOMATABLE (2 examples)
        {
            'problem_group_id': 149,
            'expected_category': 'FULLY_AUTOMATABLE',
            'description': 'till - cashier locked on till - till 4',
            'reasoning': 'Clear account unlock - maps to Unlock-ADAccount command'
        },
        {
            'problem_group_id': 68,
            'expected_category': 'FULLY_AUTOMATABLE',
            'description': 'account - password reset',
            'reasoning': 'Clear password reset - maps to Reset-ADAccountPassword command'
        },
        # PARTIALLY_AUTOMATABLE (2 examples)
        {
            'problem_group_id': 1,
            'expected_category': 'PARTIALLY_AUTOMATABLE',
            'description': 'vision - orange bar error',
            'reasoning': 'Error investigation - script can gather logs, human analyzes error'
        },
        {
            'problem_group_id': 5,
            'expected_category': 'PARTIALLY_AUTOMATABLE',
            'description': 'chip and pin error on project 4', 
            'reasoning': 'Hardware troubleshooting - automated diagnostics, human intervention needed'
        },
        # NOT_AUTOMATABLE (2 examples)
        {
            'problem_group_id': 14,
            'expected_category': 'NOT_AUTOMATABLE',
            'description': 'store manager requests screen adjustment',
            'reasoning': 'Management request - requires human judgment and physical adjustment'
        },
        {
            'problem_group_id': 208,
            'expected_category': 'NOT_AUTOMATABLE',
            'description': 'vision- screen is completely blank, no quotes no leads nothing is appearing.',
            'reasoning': 'Hardware failure - requires physical investigation and repair'
        }
    ]
    
    print("📋 Testing the 6 manually selected balanced groups:")
    for i, case in enumerate(test_cases, 1):
        print(f"{i}. {case['expected_category']} - Should be {case['expected_category']}")
        print(f"   PG_{case['problem_group_id']}: '{case['description']}'")
        print(f"   Reasoning: {case['reasoning']}")
        print()
    
    # Initialize classifier
    classifier = LLMAutomationClassifier()
    
    # Load problem groups data
    try:
        problem_groups_df = pd.read_csv('outputs/improved_problem_groups.csv')
        problem_details_df = pd.read_csv('outputs/improved_problem_group_details.csv')
        
        print(f"📊 Loaded {len(problem_groups_df)} problem groups")
        print()
        
    except FileNotFoundError as e:
        print(f"❌ Error loading data files: {e}")
        return []
    
    results = []
    
    # Test each selected group
    for i, case in enumerate(test_cases, 1):
        pg_id = case['problem_group_id']
        expected = case['expected_category']
        
        print(f"--- Test {i}/6: {expected} - Should be {expected} ---")
        print(f"PG_{pg_id}: {case['description']}")
        
        try:
            # Find the problem group data
            pg_row = problem_groups_df[problem_groups_df['problem_group_id'] == pg_id]
            if pg_row.empty:
                print(f"❌ Problem group {pg_id} not found")
                continue
                
            pg_data = pg_row.iloc[0]
            
            # Get ticket descriptions
            ticket_indices = [int(x.strip()) for x in pg_data['all_ticket_indices'].split(',')]
            detail_rows = problem_details_df[problem_details_df['ticket_index'].isin(ticket_indices)]
            ticket_descriptions = detail_rows['short_description'].tolist()
            
            # Create ProblemGroup object
            problem_group = ProblemGroup(
                problem_group_id=pg_data['problem_group_id'],
                group_size=pg_data['group_size'],
                quality_score=pg_data['quality_score'],
                representative_short_description=pg_data['representative_short_description'],
                all_ticket_indices=pg_data['all_ticket_indices'],
                ticket_descriptions=ticket_descriptions,
                primary_topic=str(pg_data['primary_topic']),
                is_high_quality=pg_data['is_high_quality']
            )
            
            # Classify the group
            classification = classifier._classify_single_group(problem_group)
            
            # Analyze result
            actual_category = classification.automation_category
            success = actual_category == expected
            
            if success:
                status = "✅ CORRECT"
            else:
                status = "❌ WRONG"
                
            print(f"{status} {actual_category} (confidence: {classification.confidence_score:.3f})")
            print(f"   Priority: {classification.business_priority}, Complexity: {classification.implementation_complexity}")
            print(f"   Enhanced Analysis Steps:")
            print(f"     Account Check: {classification.account_check}")
            print(f"     Hardware: {classification.step_1_hardware_involved}")
            print(f"     Management: {classification.step_2_management_request}")
            print(f"     Diagnosis: {classification.step_3_requires_diagnosis}")
            print(f"     Command: {classification.step_4_maps_to_command}")
            print(f"     Human Input: {classification.step_5_needs_human_input}")
            print(f"   Reasoning: {classification.reasoning[:100]}...")
            print()
            
            results.append({
                'test_case': case,
                'problem_group': problem_group,
                'classification': classification,
                'success': success,
                'expected': expected,
                'actual': actual_category
            })
            
        except Exception as e:
            print(f"❌ Classification failed: {e}")
            results.append({
                'test_case': case,
                'problem_group': None,
                'classification': None,
                'success': False,
                'error': str(e),
                'expected': expected,
                'actual': 'ERROR'
            })
    
    # Save and analyze results
    save_manual_selection_results(results)
    analyze_manual_selection_results(results)
    
    return results

def save_manual_selection_results(results):
    """Save manual selection test results"""
    
    output_dir = Path("outputs/test_6_balanced")
    output_dir.mkdir(exist_ok=True)
    
    # Save successful classifications using standard format
    successful_classifications = [r['classification'] for r in results if r['success'] and r['classification']]
    if successful_classifications:
        classifier = LLMAutomationClassifier()
        classifier.save_results(successful_classifications, output_dir="outputs/test_6_balanced")
    
    # Save detailed test analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = output_dir / f"manual_selection_analysis_{timestamp}.json"
    
    test_analysis = []
    for result in results:
        if result['classification']:
            cls = result['classification']
            test_analysis.append({
                'test_case': result['test_case'],
                'expected_category': result['expected'],
                'actual_category': result['actual'],
                'success': result['success'],
                'confidence': cls.confidence_score,
                'reasoning': cls.reasoning,
                'analysis_steps': {
                    'account_check': cls.account_check,
                    'hardware': cls.step_1_hardware_involved,
                    'management': cls.step_2_management_request,
                    'diagnosis': cls.step_3_requires_diagnosis,
                    'command': cls.step_4_maps_to_command,
                    'human_input': cls.step_5_needs_human_input
                },
                'business_metrics': {
                    'priority': cls.business_priority,
                    'complexity': cls.implementation_complexity,
                    'ticket_count': cls.ticket_count
                }
            })
        else:
            test_analysis.append({
                'test_case': result['test_case'],
                'expected_category': result['expected'],
                'actual_category': result['actual'],
                'success': result['success'],
                'error': result.get('error', 'Unknown error')
            })
    
    # Convert any numpy int64 values to regular ints for JSON serialization
    def convert_int64(obj):
        if hasattr(obj, 'dtype') and 'int' in str(obj.dtype):
            return int(obj)
        return obj
    
    # Clean the test_analysis data
    clean_analysis = json.loads(json.dumps(test_analysis, default=convert_int64))
    
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(clean_analysis, f, indent=2, ensure_ascii=False)
    
    print(f"📁 Detailed results saved in: {analysis_file}")

def analyze_manual_selection_results(results):
    """Analyze and report on manual selection test results"""
    
    print("=" * 65)
    print("📊 MANUAL SELECTION TEST ANALYSIS - Diverse Categories")
    print("=" * 65)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"✅ Successful classifications: {len(successful)}/{len(results)}")
    print(f"❌ Failed classifications: {len(failed)}/{len(results)}")
    print()
    
    print("🎯 CLASSIFICATION RESULTS BY CATEGORY:")
    print()
    
    # Group by expected category
    by_category = {}
    for result in results:
        expected = result['expected']
        if expected not in by_category:
            by_category[expected] = []
        by_category[expected].append(result)
    
    for category in ['FULLY_AUTOMATABLE', 'PARTIALLY_AUTOMATABLE', 'NOT_AUTOMATABLE']:
        if category in by_category:
            results_in_category = by_category[category]
            correct = [r for r in results_in_category if r['success']]
            
            print(f"  📂 {category}:")
            print(f"    Results: {len(correct)}/{len(results_in_category)} correct")
            
            for result in results_in_category:
                if result['classification']:
                    status = "✅" if result['success'] else "❌"
                    cls = result['classification']
                    case = result['test_case']
                    
                    print(f"    {status} PG_{case['problem_group_id']}: {case['description'][:50]}...")
                    print(f"      Expected: {result['expected']}, Got: {result['actual']} (conf: {cls.confidence_score:.3f})")
                    if not result['success']:
                        print(f"      Analysis: Hardware={cls.step_1_hardware_involved[:20]}..., Command={cls.step_4_maps_to_command[:20]}...")
                else:
                    print(f"    ❌ PG_{result['test_case']['problem_group_id']}: ERROR - {result.get('error', 'Unknown')}")
            print()
    
    # Overall success rate
    success_rate = len(successful) / len(results) * 100
    print(f"📈 OVERALL SUCCESS RATE: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 EXCELLENT - Ready for full 209-group analysis!")
    elif success_rate >= 60:
        print("✅ GOOD - Consider minor tweaks, then proceed to full analysis")
    else:
        print("⚠️  NEEDS IMPROVEMENT - Review logic before full analysis")
    
    print(f"\n📁 Detailed results saved in: outputs/test_6_balanced/")

if __name__ == "__main__":
    try:
        results = run_manual_selection_test()
        print(f"\n🎉 Manual selection test complete: {len([r for r in results if r['success']])}/{len(results)} successful")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 