#!/usr/bin/env python3
"""
Honest Robustness Test - No Gaming Allowed

Tests the system on completely novel tickets not in training data.
Uses business logic validation rather than predetermined expectations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_novel_tickets_robustness():
    """Test system on novel tickets with business logic validation."""
    print("=" * 60)
    print("HONEST ROBUSTNESS TEST - NOVEL TICKETS")
    print("=" * 60)
    
    try:
        from src.two_tier_classifier.core.pipeline_controller import ThreeTierClassifier
        
        # Initialize classifier
        classifier = ThreeTierClassifier(enable_automation_analysis=True)
        print("✅ ThreeTierClassifier initialized successfully")
        
        # NOVEL test cases - tickets that wouldn't be in training data
        # Testing business logic rather than exact category matches
        novel_test_cases = [
            {
                'text': "new quantum computer server motherboard needs replacement urgently",
                'business_logic_test': "Should route to team capable of hardware replacement",
                'expected_automation_logic': "Physical replacement = NOT_AUTOMATABLE",
                'category_validation': lambda cat: cat in ["General Support", "Hardware Support", "Infrastructure"],
                'automation_validation': lambda auto: auto == "NOT_AUTOMATABLE"
            },
            {
                'text': "blockchain application keeps crashing during cryptocurrency processing",
                'business_logic_test': "Should route to software/application support team",
                'expected_automation_logic': "Application debugging = PARTIALLY_AUTOMATABLE",  
                'category_validation': lambda cat: cat in ["Software & Application Issues", "General Support"],
                'automation_validation': lambda auto: auto in ["PARTIALLY_AUTOMATABLE", "NOT_AUTOMATABLE"]
            },
            {
                'text': "AI cashier bot locked out of quantum till system customers complaining",
                'business_logic_test': "Should route to till operations (customer impact priority)",
                'expected_automation_logic': "Account unlock in till context = likely FULLY_AUTOMATABLE",
                'category_validation': lambda cat: cat in ["Till Operations", "User Account Management"],
                'automation_validation': lambda auto: auto in ["FULLY_AUTOMATABLE", "PARTIALLY_AUTOMATABLE"]
            },
            {
                'text': "holographic printer driver for 3D food printing installation needed",
                'business_logic_test': "Should route to printing services team",
                'expected_automation_logic': "Driver installation = PARTIALLY_AUTOMATABLE",
                'category_validation': lambda cat: cat in ["Printing Services", "Software & Application Issues"],
                'automation_validation': lambda auto: auto == "PARTIALLY_AUTOMATABLE"
            },
            {
                'text': "virtual reality headset display flickering during metaverse shopping experience",
                'business_logic_test': "Should route to hardware/device support team", 
                'expected_automation_logic': "Hardware troubleshooting = PARTIALLY_AUTOMATABLE",
                'category_validation': lambda cat: cat in ["Mobile Devices", "General Support", "Hardware Support"],
                'automation_validation': lambda auto: auto in ["PARTIALLY_AUTOMATABLE", "NOT_AUTOMATABLE"]
            },
            {
                'text': "employee cannot access neural network email system after brain implant upgrade",
                'business_logic_test': "Should route to email/communications or account management",
                'expected_automation_logic': "Account/access issue = likely PARTIALLY_AUTOMATABLE",
                'category_validation': lambda cat: cat in ["Email & Communications", "User Account Management"],
                'automation_validation': lambda auto: auto in ["PARTIALLY_AUTOMATABLE", "FULLY_AUTOMATABLE"]
            }
        ]
        
        print(f"\n🔍 Testing {len(novel_test_cases)} completely novel tickets:")
        print("   (These tickets use technology/concepts not in training data)")
        
        business_logic_correct = 0
        automation_logic_correct = 0
        total_tests = len(novel_test_cases)
        
        for i, case in enumerate(novel_test_cases, 1):
            try:
                result = classifier.classify(
                    text=case['text'],
                    include_level2=True,
                    include_level3=True
                )
                
                # Test business routing logic
                business_correct = case['category_validation'](result.business_category)
                if business_correct:
                    business_logic_correct += 1
                
                # Test automation logic  
                automation_correct = case['automation_validation'](result.automation_category)
                if automation_correct:
                    automation_logic_correct += 1
                
                print(f"\n{i}. Novel Ticket Test:")
                print(f"   Input: \"{case['text'][:80]}{'...' if len(case['text']) > 80 else ''}\"")
                print(f"   Business Logic: {case['business_logic_test']}")
                print(f"   → Routed to: {result.business_category} {'✅' if business_correct else '❌'}")
                print(f"   Automation Logic: {case['expected_automation_logic']}")
                print(f"   → Classified as: {result.automation_category} ({result.automation_percentage}%) {'✅' if automation_correct else '❌'}")
                print(f"   Processing: {result.total_processing_time_ms:.1f}ms")
                
                if not business_correct or not automation_correct:
                    print(f"   🔍 ROBUSTNESS ANALYSIS:")
                    if not business_correct:
                        print(f"      - Business routing may not follow expected logic")
                        print(f"      - Routed to '{result.business_category}' - is this operationally correct?")
                    if not automation_correct:
                        print(f"      - Automation classification may not follow expected logic") 
                        print(f"      - Reasoning: {result.automation_reasoning}")
                        
            except Exception as e:
                print(f"\n{i}. ❌ SYSTEM FAILURE on novel ticket:")
                print(f"   Input: \"{case['text'][:80]}{'...' if len(case['text']) > 80 else ''}\"")
                print(f"   Error: {e}")
                # Both logic tests fail if system crashes
                continue
        
        # Calculate robustness metrics
        business_robustness = business_logic_correct / total_tests
        automation_robustness = automation_logic_correct / total_tests
        overall_robustness = (business_logic_correct + automation_logic_correct) / (total_tests * 2)
        
        print(f"\n📊 HONEST ROBUSTNESS RESULTS:")
        print(f"   Business Logic Robustness: {business_robustness:.1%} ({business_logic_correct}/{total_tests})")
        print(f"   Automation Logic Robustness: {automation_robustness:.1%} ({automation_logic_correct}/{total_tests})")
        print(f"   Overall System Robustness: {overall_robustness:.1%}")
        print(f"   Production Robustness Target: 70%+ (lower bar for novel cases)")
        
        # Honest assessment
        print(f"\n🎯 HONEST ASSESSMENT:")
        if overall_robustness >= 0.8:
            print(f"   🎉 EXCELLENT: System demonstrates strong robustness on novel cases")
        elif overall_robustness >= 0.7:
            print(f"   ✅ GOOD: System shows reasonable robustness, ready for production with monitoring")
        elif overall_robustness >= 0.5:
            print(f"   ⚠️  MODERATE: System has some robustness but needs improvement")
        else:
            print(f"   ❌ POOR: System lacks robustness for novel cases - more work needed")
        
        print(f"\n🔍 REALITY CHECK:")
        print(f"   - These novel tickets test business logic, not memorization")
        print(f"   - Results show whether system can handle future unseen tickets")
        print(f"   - This is the true test of production readiness")
        
        return {
            'business_robustness': business_robustness,
            'automation_robustness': automation_robustness, 
            'overall_robustness': overall_robustness,
            'production_ready': overall_robustness >= 0.7
        }
        
    except Exception as e:
        print(f"❌ Honest robustness test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_novel_tickets_robustness()
    if result:
        exit_code = 0 if result['production_ready'] else 1
        print(f"\nFinal Answer: {'PRODUCTION READY' if result['production_ready'] else 'NEEDS MORE WORK'}")
    else:
        exit_code = 2
    sys.exit(exit_code)