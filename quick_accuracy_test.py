#!/usr/bin/env python3
"""
Quick Accuracy Test - Focus on Known Issues

Tests the specific accuracy problems identified in the implementation plan:
1. Hardware vs software misclassification  
2. Context-blind automation analysis
3. Vision order context issues

Author: Claude Code Assistant
Date: 2025-08-14
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_known_accuracy_issues():
    """Test the specific accuracy issues identified in the plan."""
    print("=" * 60)
    print("QUICK ACCURACY TEST - KNOWN ISSUES")
    print("=" * 60)
    
    try:
        from src.two_tier_classifier.core.pipeline_controller import ThreeTierClassifier
        
        # Initialize classifier
        classifier = ThreeTierClassifier(enable_automation_analysis=True)
        print("✅ ThreeTierClassifier initialized successfully")
        
        # Test cases from implementation plan - using correct existing category names
        test_cases = [
            {
                'text': "replace broken CPU on physical server",
                'expected_business': "General Support",  # Hardware issues go to General Support
                'expected_automation': "NOT_AUTOMATABLE",
                'expected_percentage_max': 25,
                'issue': "Hardware classification (should work correctly)"
            },
            {
                'text': "vision order locked cannot modify quantities urgent",
                'expected_business': "Vision Orders & Inventory",  # Vision orders go to Vision category
                'expected_automation': "PARTIALLY_AUTOMATABLE", 
                'expected_percentage_range': (40, 70),  # NOT 95% like account unlock
                'issue': "Context-aware automation (Vision vs Account context)"
            },
            {
                'text': "cashier sarah locked out till 3 customers waiting",
                'expected_business': "Till Operations",
                'expected_automation': "FULLY_AUTOMATABLE",  # This one should work correctly
                'expected_percentage_min': 85,
                'issue': "Account unlock context (should work correctly)"
            },
            {
                'text': "printer driver installation required",
                'expected_business': "Printing Services",  # Should be printing, not general
                'expected_automation': "PARTIALLY_AUTOMATABLE",
                'expected_percentage_range': (50, 75),
                'issue': "Printing service classification (should work correctly)"
            },
            {
                'text': "mobile device screen cracked needs replacement",
                'expected_business': "Mobile Devices",  # Mobile hardware goes to Mobile Devices
                'expected_automation': "NOT_AUTOMATABLE",
                'expected_percentage_max': 20,
                'issue': "Mobile device hardware classification (should work correctly)"
            }
        ]
        
        print(f"\n🔍 Testing {len(test_cases)} known accuracy issue cases:")
        
        correct_business = 0
        correct_automation = 0
        total_tests = len(test_cases)
        
        for i, case in enumerate(test_cases, 1):
            result = classifier.classify(
                text=case['text'],
                include_level2=True,
                include_level3=True
            )
            
            # Check business category accuracy
            business_correct = result.business_category == case['expected_business']
            if business_correct:
                correct_business += 1
            
            # Check automation category accuracy
            automation_correct = result.automation_category == case['expected_automation']
            
            # Check percentage accuracy
            percentage_valid = True
            if 'expected_percentage_min' in case:
                percentage_valid = result.automation_percentage >= case['expected_percentage_min']
            elif 'expected_percentage_max' in case:
                percentage_valid = result.automation_percentage <= case['expected_percentage_max']
            elif 'expected_percentage_range' in case:
                min_p, max_p = case['expected_percentage_range']
                percentage_valid = min_p <= result.automation_percentage <= max_p
            
            if automation_correct and percentage_valid:
                correct_automation += 1
            
            print(f"\n{i}. Issue: {case['issue']}")
            print(f"   Input: \"{case['text']}\"")
            print(f"   Expected Business: {case['expected_business']}")
            print(f"   Actual Business: {result.business_category} {'✅' if business_correct else '❌'}")
            print(f"   Expected Automation: {case['expected_automation']}")
            print(f"   Actual Automation: {result.automation_category} ({result.automation_percentage}%) {'✅' if automation_correct and percentage_valid else '❌'}")
            print(f"   Processing: {result.total_processing_time_ms:.1f}ms")
            
            if not business_correct or not (automation_correct and percentage_valid):
                print(f"   🔍 ROOT CAUSE ANALYSIS:")
                if not business_correct:
                    print(f"      - Level 1 misclassified: {result.business_category} vs expected {case['expected_business']}")
                if not automation_correct:
                    print(f"      - Level 3 automation wrong: {result.automation_category} vs expected {case['expected_automation']}")
                if automation_correct and not percentage_valid:
                    print(f"      - Level 3 percentage wrong: {result.automation_percentage}% outside expected range")
                if result.level3_result:
                    print(f"      - Automation layer used: {result.level3_result.layer_used}")
                    print(f"      - Automation reasoning: {result.automation_reasoning}")
        
        # Calculate accuracy metrics
        business_accuracy = correct_business / total_tests
        automation_accuracy = correct_automation / total_tests
        overall_accuracy = (correct_business + correct_automation) / (total_tests * 2)
        
        print(f"\n📊 Accuracy Results:")
        print(f"   Business Category Accuracy: {business_accuracy:.1%} ({correct_business}/{total_tests})")
        print(f"   Automation Analysis Accuracy: {automation_accuracy:.1%} ({correct_automation}/{total_tests})")
        print(f"   Overall Accuracy: {overall_accuracy:.1%} ({correct_business + correct_automation}/{total_tests * 2})")
        print(f"   Target: 80%+ for production deployment")
        
        # Diagnosis
        print(f"\n🎯 DIAGNOSIS:")
        if business_accuracy < 0.8:
            print(f"   ❌ LEVEL 1 ISSUE: Business classification accuracy {business_accuracy:.1%} < 80% target")
            print(f"      → Need to retrain Level 1 classifier for better category distinction")
        else:
            print(f"   ✅ Level 1: Business classification meeting target")
            
        if automation_accuracy < 0.8:
            print(f"   ❌ LEVEL 3 ISSUE: Automation analysis accuracy {automation_accuracy:.1%} < 80% target")
            print(f"      → Need context-aware automation analysis improvements")
        else:
            print(f"   ✅ Level 3: Automation analysis meeting target")
        
        if overall_accuracy >= 0.8:
            print(f"   🎉 OVERALL: System ready for production deployment!")
        else:
            print(f"   ⚠️  OVERALL: System needs {0.8 - overall_accuracy:.1%} improvement before production")
        
        return {
            'business_accuracy': business_accuracy,
            'automation_accuracy': automation_accuracy,
            'overall_accuracy': overall_accuracy,
            'ready_for_production': overall_accuracy >= 0.8
        }
        
    except Exception as e:
        print(f"❌ Quick accuracy test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_known_accuracy_issues()
    if result:
        exit_code = 0 if result['ready_for_production'] else 1
    else:
        exit_code = 2
    sys.exit(exit_code)