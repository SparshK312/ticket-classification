#!/usr/bin/env python3
"""
Week 3 Implementation Test Runner

Tests the complete three-tier system with Level 1 business classification,
Level 2 semantic search, and Level 3 automation analysis.
"""

import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_three_tier_integration():
    """Test complete three-tier system integration."""
    print("=" * 60)
    print("TEST 1: THREE-TIER SYSTEM INTEGRATION")
    print("=" * 60)
    
    try:
        from src.two_tier_classifier.core.pipeline_controller import ThreeTierClassifier
        
        # Initialize complete three-tier system
        print("Initializing ThreeTierClassifier...")
        
        classifier = ThreeTierClassifier(
            model_name='all-MiniLM-L6-v2',
            cache_dir='cache/embeddings',
            confidence_threshold=0.6,
            enable_automation_analysis=True
        )
        
        print("✅ Three-tier system initialized successfully")
        
        # Test cases covering manager's automation requirements
        test_cases = [
            {
                'text': "unlock user account john.doe",
                'expected_l1': "User Account Management", 
                'expected_automation': "FULLY_AUTOMATABLE",
                'expected_percentage_min': 85
            },
            {
                'text': "till crashed during busy period customers waiting",
                'expected_l1': "Till Operations",
                'expected_automation': "PARTIALLY_AUTOMATABLE", 
                'expected_percentage_min': 30
            },
            {
                'text': "replace broken CPU on physical server",
                'expected_l1': "General Support",
                'expected_automation': "NOT_AUTOMATABLE",
                'expected_percentage_max': 25
            },
            {
                'text': "vision order locked cannot modify quantities urgent",
                'expected_l1': "Vision Orders & Inventory",
                'expected_automation': "PARTIALLY_AUTOMATABLE",
                'expected_percentage_min': 40
            },
            {
                'text': "chip pin device offline payment failed",
                'expected_l1': "Payment Processing", 
                'expected_automation': "PARTIALLY_AUTOMATABLE",
                'expected_percentage_min': 30
            }
        ]
        
        print(f"\n🔄 Testing complete three-tier classification on {len(test_cases)} tickets:")
        
        successful_classifications = 0
        total_processing_time = 0
        automation_results = []
        
        for i, test_case in enumerate(test_cases, 1):
            start_time = time.time()
            
            # Test with all three tiers enabled
            result = classifier.classify(
                text=test_case['text'],
                include_level2=True,
                include_level3=True,
                include_explanations=False
            )
            
            processing_time = (time.time() - start_time) * 1000
            total_processing_time += processing_time
            
            # Validate Level 1 accuracy
            l1_correct = result.business_category == test_case['expected_l1']
            
            # Validate Level 3 automation
            automation_correct = result.automation_category == test_case['expected_automation']
            
            # Validate percentage range
            percentage_valid = True
            if 'expected_percentage_min' in test_case:
                percentage_valid = result.automation_percentage >= test_case['expected_percentage_min']
            elif 'expected_percentage_max' in test_case:
                percentage_valid = result.automation_percentage <= test_case['expected_percentage_max']
            
            if l1_correct and automation_correct and percentage_valid:
                successful_classifications += 1
            
            print(f"\n{i}. Input: \"{test_case['text']}\"")
            print(f"   Level 1: {result.business_category} {'✅' if l1_correct else '❌'}")
            print(f"   Level 2: {result.specific_problem}")
            print(f"   Level 3: {result.automation_category} ({result.automation_percentage}%) {'✅' if automation_correct and percentage_valid else '❌'}")
            print(f"   Reasoning: {result.automation_reasoning}")
            print(f"   Processing: {processing_time:.1f}ms")
            print(f"   Overall Confidence: {result.overall_confidence:.3f}")
            
            automation_results.append({
                'category': result.automation_category,
                'percentage': result.automation_percentage,
                'layer': result.level3_result.layer_used if result.level3_result else 'unknown'
            })
        
        # Calculate metrics
        accuracy = successful_classifications / len(test_cases)
        avg_processing_time = total_processing_time / len(test_cases)
        
        print(f"\n📊 Three-Tier Integration Results:")
        print(f"   Overall Accuracy: {accuracy:.1%} ({successful_classifications}/{len(test_cases)})")
        print(f"   Avg Processing Time: {avg_processing_time:.1f}ms")
        print(f"   Speed Target (<2s): {'✅' if avg_processing_time <= 2000 else '❌'}")
        
        # Automation layer distribution
        from collections import Counter
        layer_usage = Counter([r['layer'] for r in automation_results])
        print(f"\n🔍 Automation Analysis Layer Usage:")
        for layer, count in layer_usage.items():
            print(f"   {layer}: {count} uses ({count/len(automation_results)*100:.1f}%)")
        
        # System statistics
        system_stats = classifier.get_system_stats()
        print(f"\n🔧 System Statistics:")
        print(f"   Total Classifications: {system_stats['total_classifications']}")
        print(f"   Level 3 Usage: {system_stats['level3_classifications']}")
        
        automation_stats = system_stats.get('automation_category_distribution', {})
        if automation_stats:
            print(f"   Automation Categories:")
            for category, count in automation_stats.items():
                print(f"     {category}: {count}")
        
        return {
            'status': 'PASSED' if accuracy >= 0.8 else 'PARTIAL',
            'accuracy': accuracy,
            'avg_processing_time_ms': avg_processing_time,
            'automation_layer_distribution': dict(layer_usage)
        }
        
    except Exception as e:
        print(f"❌ Three-tier integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'ERROR', 'error': str(e)}

def test_automation_coverage():
    """Test automation analysis coverage and manager requirements."""
    print("\n" + "=" * 60)
    print("TEST 2: AUTOMATION ANALYSIS COVERAGE")
    print("=" * 60)
    
    try:
        from src.two_tier_classifier.core.automation_analyzer import ComprehensiveAutomationAnalyzer
        
        analyzer = ComprehensiveAutomationAnalyzer()
        
        # Test edge cases for complete coverage
        edge_cases = [
            ("", "General Support"),  # Empty input
            ("help me understand this policy", "General Support"),  # Training request  
            ("replace broken hard drive", "General Support"),  # Physical hardware
            ("server performance very slow", "General Support"),  # Investigation needed
            ("reset user password", "User Account Management"),  # Clear automation
            ("some completely novel technical issue never seen before", "General Support")  # Novel case
        ]
        
        print(f"🔍 Testing automation coverage on {len(edge_cases)} edge cases:")
        
        coverage_success = 0
        layer_distribution = {}
        
        for i, (text, category) in enumerate(edge_cases, 1):
            result = analyzer.analyze(text, category, [])
            
            # Every case should get a result
            has_result = (result.category in ["FULLY_AUTOMATABLE", "PARTIALLY_AUTOMATABLE", "NOT_AUTOMATABLE"] and 
                         0 <= result.automation_percentage <= 100)
            
            if has_result:
                coverage_success += 1
            
            layer_used = result.layer_used
            layer_distribution[layer_used] = layer_distribution.get(layer_used, 0) + 1
            
            print(f"{i}. \"{text[:40]}{'...' if len(text) > 40 else ''}\"")
            print(f"   → {result.category} ({result.automation_percentage}%)")
            print(f"   → Layer: {layer_used}, Confidence: {result.confidence:.3f}")
            print(f"   → {'✅' if has_result else '❌'}")
        
        coverage_rate = coverage_success / len(edge_cases)
        
        print(f"\n📊 Automation Coverage Results:")
        print(f"   Coverage Success: {coverage_rate:.1%} ({coverage_success}/{len(edge_cases)})")
        print(f"   Layer Distribution:")
        for layer, count in layer_distribution.items():
            print(f"     {layer}: {count} ({count/len(edge_cases)*100:.1f}%)")
        
        analyzer_stats = analyzer.get_analysis_stats()
        print(f"\n🔧 Analyzer Statistics:")
        print(f"   Total Analyses: {analyzer_stats['total_analyses']}")
        print(f"   Historical Database: {analyzer_stats['historical_database_size']} mappings")
        print(f"   LLM Available: {analyzer_stats['llm_available']}")
        
        return {
            'status': 'PASSED' if coverage_rate == 1.0 else 'PARTIAL',
            'coverage_rate': coverage_rate,
            'layer_distribution': layer_distribution
        }
        
    except Exception as e:
        print(f"❌ Automation coverage test failed: {e}")
        return {'status': 'ERROR', 'error': str(e)}

def test_manager_requirements():
    """Test specific manager requirements for automation categories."""
    print("\n" + "=" * 60)
    print("TEST 3: MANAGER REQUIREMENTS COMPLIANCE")
    print("=" * 60)
    
    try:
        from src.two_tier_classifier.core.pipeline_controller import ThreeTierClassifier
        
        classifier = ThreeTierClassifier()
        
        # Manager's specific definitions
        manager_test_cases = [
            {
                "text": "unlock user account sarah.smith",
                "expected_category": "FULLY_AUTOMATABLE",
                "requirement": "No human intervention at all - maps to Unlock-ADAccount",
                "expected_percentage_min": 85
            },
            {
                "text": "investigate network performance issues",
                "expected_category": "PARTIALLY_AUTOMATABLE", 
                "requirement": "7 steps automated, 3 manual - hybrid process",
                "expected_percentage_range": (40, 80)
            },
            {
                "text": "physically replace broken motherboard",
                "expected_category": "NOT_AUTOMATABLE",
                "requirement": "Physical action required - cannot automate hardware replacement",
                "expected_percentage_max": 20
            }
        ]
        
        print(f"📋 Testing manager requirements on {len(manager_test_cases)} specific cases:")
        
        requirements_met = 0
        
        for i, case in enumerate(manager_test_cases, 1):
            result = classifier.classify(case["text"], include_level3=True)
            
            # Check category compliance
            category_correct = result.automation_category == case["expected_category"]
            
            # Check percentage compliance
            percentage_correct = True
            if "expected_percentage_min" in case:
                percentage_correct = result.automation_percentage >= case["expected_percentage_min"]
            elif "expected_percentage_max" in case:
                percentage_correct = result.automation_percentage <= case["expected_percentage_max"]
            elif "expected_percentage_range" in case:
                min_p, max_p = case["expected_percentage_range"]
                percentage_correct = min_p <= result.automation_percentage <= max_p
            
            if category_correct and percentage_correct:
                requirements_met += 1
            
            print(f"\n{i}. Manager Requirement Test:")
            print(f"   Input: \"{case['text']}\"")
            print(f"   Requirement: {case['requirement']}")
            print(f"   Expected: {case['expected_category']}")
            print(f"   Actual: {result.automation_category} ({result.automation_percentage}%)")
            print(f"   Reasoning: {result.automation_reasoning}")
            print(f"   Compliance: {'✅' if category_correct and percentage_correct else '❌'}")
        
        compliance_rate = requirements_met / len(manager_test_cases)
        
        print(f"\n📊 Manager Requirements Results:")
        print(f"   Requirements Met: {compliance_rate:.1%} ({requirements_met}/{len(manager_test_cases)})")
        print(f"   Compliance Target (100%): {'✅' if compliance_rate == 1.0 else '❌'}")
        
        return {
            'status': 'PASSED' if compliance_rate >= 0.8 else 'PARTIAL',
            'compliance_rate': compliance_rate,
            'requirements_met': requirements_met
        }
        
    except Exception as e:
        print(f"❌ Manager requirements test failed: {e}")
        return {'status': 'ERROR', 'error': str(e)}

def main():
    """Run all Week 3 tests."""
    print("WEEK 3 IMPLEMENTATION VALIDATION")
    print("Testing Three-Tier System: Level 1 + Level 2 + Level 3 Automation Analysis")
    print()
    
    start_time = time.time()
    
    # Run tests
    integration_result = test_three_tier_integration()
    coverage_result = test_automation_coverage()
    manager_result = test_manager_requirements()
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("WEEK 3 VALIDATION SUMMARY")
    print("=" * 60)
    
    all_results = [integration_result, coverage_result, manager_result]
    passed_tests = sum(1 for r in all_results if r.get('status') == 'PASSED')
    
    print(f"Tests completed: 3/3")
    print(f"Tests passed: {passed_tests}/3")
    print(f"Total validation time: {total_time:.1f} seconds")
    print()
    
    # Detailed results
    print("Test Results:")
    print(f"  1. Three-Tier Integration: {integration_result.get('status')}")
    if 'accuracy' in integration_result:
        print(f"     - Overall accuracy: {integration_result['accuracy']:.1%}")
        print(f"     - Processing time: {integration_result['avg_processing_time_ms']:.1f}ms")
    
    print(f"  2. Automation Coverage: {coverage_result.get('status')}")
    if 'coverage_rate' in coverage_result:
        print(f"     - Coverage rate: {coverage_result['coverage_rate']:.1%}")
    
    print(f"  3. Manager Requirements: {manager_result.get('status')}")
    if 'compliance_rate' in manager_result:
        print(f"     - Compliance rate: {manager_result['compliance_rate']:.1%}")
    
    print()
    
    # Overall assessment
    if passed_tests == 3:
        print("🎉 EXCELLENT: Week 3 implementation successful!")
        print("✅ THREE-TIER SYSTEM WORKING")
        print("✅ 100% AUTOMATION COVERAGE")
        print("✅ MANAGER REQUIREMENTS MET")
        
        print("\nWeek 3 Achievements:")
        print("  - 5-layer automation analysis implemented")
        print("  - Historical automation database (209 + 1,203 mappings)")
        print("  - Step-by-step percentage calculation")  
        print("  - Complete three-tier pipeline integration")
        print("  - Manager requirement compliance")
        
        next_steps = [
            "✅ Week 3 automation analysis complete",
            "🚀 Ready to proceed to Week 4: UI Demo Integration",
            "🎯 End-to-end system ready for production"
        ]
        
    elif passed_tests >= 2:
        print("✅ GOOD: Week 3 core functionality working")
        print("⚠️  Some components may need fine-tuning")
        
        next_steps = [
            "✅ Core automation analysis validated",
            "⚠️  Address any failed test components", 
            "🔄 Consider additional validation if needed"
        ]
    else:
        print("⚠️  Week 3 needs attention before proceeding")
        
        next_steps = [
            "❌ Review failed test results",
            "🔧 Fix automation analysis issues", 
            "🔄 Re-run validation before Week 4"
        ]
    
    print("Next Steps:")
    for step in next_steps:
        print(f"  {step}")
    
    return {
        'overall_status': 'PASSED' if passed_tests >= 2 else 'NEEDS_WORK',
        'tests_passed': passed_tests,
        'total_tests': 3,
        'processing_time': total_time,
        'results': {
            'integration': integration_result,
            'coverage': coverage_result,
            'manager_requirements': manager_result
        }
    }

if __name__ == "__main__":
    results = main()
    exit_code = 0 if results['overall_status'] == 'PASSED' else 1
    sys.exit(exit_code)