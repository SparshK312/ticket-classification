#!/usr/bin/env python3
"""
Test script for LLM Automation Classifier

Tests the classifier with a small sample of problem groups to verify:
1. Ollama connection and model availability
2. Problem group loading
3. LLM classification functionality
4. Output generation

Run this before executing the full analysis on all 209 groups.
"""

import sys
import logging
from pathlib import Path
import traceback

# Import our classifier
from llm_automation_classifier import LLMAutomationClassifier, ProblemGroup
from automation_config import OLLAMA_CONFIG, ANALYSIS_CONFIG

def test_ollama_connection():
    """Test 1: Verify Ollama connection and model availability"""
    print("🔗 Test 1: Ollama Connection and Model Availability")
    print("-" * 50)
    
    try:
        classifier = LLMAutomationClassifier(
            ollama_host=OLLAMA_CONFIG["host"],
            model=OLLAMA_CONFIG["model"]
        )
        print("✅ Ollama connection successful")
        print(f"✅ Model {OLLAMA_CONFIG['model']} is available")
        return classifier
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Ollama is running: ollama serve")
        print("2. Verify model is installed: ollama list")
        print("3. If not installed: ollama pull llama3.1:8b")
        return None

def test_data_loading(classifier):
    """Test 2: Verify problem group data loading"""
    print("\n📁 Test 2: Problem Group Data Loading")
    print("-" * 50)
    
    try:
        problem_groups = classifier.load_problem_groups()
        print(f"✅ Loaded {len(problem_groups)} problem groups")
        
        # Show sample data
        if problem_groups:
            sample = problem_groups[0]
            print(f"✅ Sample group: PG_{sample.problem_group_id}")
            print(f"   - Size: {sample.group_size} tickets")
            print(f"   - Quality: {sample.quality_score:.3f}")
            print(f"   - Description: {sample.representative_short_description[:60]}...")
            print(f"   - High Quality: {sample.is_high_quality}")
        
        return problem_groups
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure plan1.md outputs exist in outputs/ directory")
        print("2. Check file paths in automation_config.py")
        traceback.print_exc()
        return None

def test_single_classification(classifier, problem_groups):
    """Test 3: Single problem group classification"""
    print("\n🤖 Test 3: Single Problem Group Classification")
    print("-" * 50)
    
    if not problem_groups:
        print("❌ No problem groups available for testing")
        return None
    
    # Select a good test candidate (high quality, medium size)
    test_group = None
    for group in problem_groups:
        if group.is_high_quality and 3 <= group.group_size <= 20:
            test_group = group
            break
    
    if not test_group:
        test_group = problem_groups[0]  # Fallback to first group
    
    print(f"🎯 Testing with: PG_{test_group.problem_group_id}")
    print(f"   - {test_group.group_size} tickets")
    print(f"   - Quality: {test_group.quality_score:.3f}")
    print(f"   - Description: {test_group.representative_short_description}")
    
    try:
        result = classifier._classify_single_group(test_group)
        
        print("✅ Classification successful!")
        print(f"   - Category: {result.automation_category}")
        print(f"   - Confidence: {result.confidence_score:.3f}")
        print(f"   - Priority: {result.business_priority}")
        print(f"   - Complexity: {result.implementation_complexity}")
        print(f"   - Processing time: {result.processing_time_seconds:.2f}s")
        print(f"   - Reasoning: {result.reasoning[:100]}...")
        
        return [result]
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        traceback.print_exc()
        return None

def test_output_generation(classifier, results):
    """Test 4: Output file generation"""
    print("\n💾 Test 4: Output File Generation")
    print("-" * 50)
    
    if not results:
        print("❌ No results available for output testing")
        return False
    
    try:
        # Create test output directory
        test_output_dir = "outputs/test"
        Path(test_output_dir).mkdir(exist_ok=True)
        
        # Save results (this will create timestamped files)
        classifier.save_results(results, output_dir=test_output_dir)
        
        # Check if files were created
        output_files = list(Path(test_output_dir).glob("*"))
        if output_files:
            print(f"✅ Generated {len(output_files)} output files:")
            for file in output_files:
                print(f"   - {file.name}")
        else:
            print("❌ No output files generated")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Output generation failed: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run all tests in sequence"""
    print("🧪 LLM Automation Classifier - Comprehensive Test Suite")
    print("=" * 60)
    
    # Test 1: Ollama Connection
    classifier = test_ollama_connection()
    if not classifier:
        print("\n❌ Critical failure: Cannot proceed without Ollama connection")
        return False
    
    # Test 2: Data Loading
    problem_groups = test_data_loading(classifier)
    if not problem_groups:
        print("\n❌ Critical failure: Cannot proceed without problem group data")
        return False
    
    # Test 3: Classification
    results = test_single_classification(classifier, problem_groups)
    if not results:
        print("\n❌ Warning: Classification test failed, but other components may work")
    
    # Test 4: Output Generation
    if results:
        output_success = test_output_generation(classifier, results)
    else:
        print("\n⏭️  Skipping output generation test (no classification results)")
        output_success = False
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("-" * 60)
    
    if classifier:
        print("✅ Ollama connection: PASS")
    else:
        print("❌ Ollama connection: FAIL")
    
    if problem_groups:
        print("✅ Data loading: PASS")
    else:
        print("❌ Data loading: FAIL")
    
    if results:
        print("✅ LLM classification: PASS")
    else:
        print("❌ LLM classification: FAIL")
    
    if output_success:
        print("✅ Output generation: PASS")
    else:
        print("❌ Output generation: FAIL")
    
    # Overall assessment
    critical_tests_passed = classifier is not None and problem_groups is not None
    
    if critical_tests_passed and results:
        print("\n🎉 ALL TESTS PASSED - Ready for full analysis!")
        print("\nTo run the complete analysis:")
        print("python llm_automation_classifier.py")
        return True
    elif critical_tests_passed:
        print("\n⚠️  PARTIAL SUCCESS - Basic setup works, LLM issues detected")
        print("Check Ollama model and try again")
        return False
    else:
        print("\n❌ CRITICAL FAILURES - Fix issues before proceeding")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)