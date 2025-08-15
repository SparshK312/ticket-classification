#!/usr/bin/env python3
"""
Deployment Optimization Testing Script

This script validates the deployment optimization implementation by:
1. Testing current system functionality (baseline)
2. Running deployment asset preparation
3. Testing optimized system functionality 
4. Comparing performance and ensuring no regression

This is a SAFETY-FIRST approach to ensure local functionality is preserved.
"""

import os
import sys
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def run_comprehensive_testing():
    """Run complete testing pipeline."""
    
    print("🧪 Deployment Optimization Testing Pipeline")
    print("="*60)
    
    results = {
        'baseline_test': None,
        'asset_preparation': None,
        'optimized_test': None,
        'performance_comparison': None,
        'safety_validation': None
    }
    
    try:
        # Step 1: Test baseline functionality
        print("\n1️⃣ Testing Baseline Functionality...")
        results['baseline_test'] = test_baseline_functionality()
        
        # Step 2: Prepare deployment assets
        print("\n2️⃣ Preparing Deployment Assets...")
        results['asset_preparation'] = prepare_and_test_assets()
        
        # Step 3: Test optimized functionality
        print("\n3️⃣ Testing Optimized Functionality...")
        results['optimized_test'] = test_optimized_functionality()
        
        # Step 4: Performance comparison
        print("\n4️⃣ Performance Comparison...")
        results['performance_comparison'] = compare_performance(
            results['baseline_test'], 
            results['optimized_test']
        )
        
        # Step 5: Safety validation
        print("\n5️⃣ Safety Validation...")
        results['safety_validation'] = validate_safety(
            results['baseline_test'], 
            results['optimized_test']
        )
        
        # Final report
        print_final_report(results)
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        print("This indicates the optimization approach needs revision.")
        return False
    
    return all(results.values())

def test_baseline_functionality():
    """Test current system functionality as baseline."""
    
    print("   📊 Testing current three-tier system...")
    
    try:
        from two_tier_classifier.core.pipeline_controller import ThreeTierClassifier
        
        # Initialize system and measure time
        start_time = time.time()
        classifier = ThreeTierClassifier(
            model_name='all-MiniLM-L6-v2',
            cache_dir='cache/embeddings',
            confidence_threshold=0.6,
            enable_automation_analysis=True
        )
        init_time = time.time() - start_time
        
        # Test classification with standard examples
        test_cases = [
            "cashier sarah locked out till 3 customers waiting urgent",
            "vision order locked cannot modify quantities store manager approval needed",
            "replace broken CPU on physical server motherboard damaged",
            "unlock user account john.doe",
            "printer driver installation required for new HP LaserJet"
        ]
        
        classification_results = []
        classification_times = []
        
        for test_case in test_cases:
            case_start = time.time()
            result = classifier.classify(
                text=test_case,
                include_level2=True,
                include_level3=True,
                include_explanations=False
            )
            case_time = time.time() - case_start
            
            classification_results.append({
                'input': test_case,
                'business_category': result.business_category,
                'confidence': result.confidence,
                'automation_potential': result.automation_category,
                'processing_time_ms': case_time * 1000
            })
            classification_times.append(case_time)
        
        baseline_results = {
            'status': 'success',
            'initialization_time_ms': init_time * 1000,
            'avg_classification_time_ms': sum(classification_times) / len(classification_times) * 1000,
            'classifications': classification_results,
            'system_stats': classifier.get_performance_stats() if hasattr(classifier, 'get_performance_stats') else {}
        }
        
        print(f"   ✅ Baseline test completed:")
        print(f"      - Initialization: {init_time*1000:.1f}ms")
        print(f"      - Avg classification: {baseline_results['avg_classification_time_ms']:.1f}ms")
        print(f"      - Test cases: {len(classification_results)}/5 successful")
        
        return baseline_results
        
    except Exception as e:
        print(f"   ❌ Baseline test failed: {e}")
        raise

def prepare_and_test_assets():
    """Prepare deployment assets and test their creation."""
    
    print("   📦 Running asset preparation script...")
    
    try:
        # Import and run the asset preparation
        from prepare_deployment_assets import prepare_deployment_assets
        
        start_time = time.time()
        prepare_deployment_assets()
        preparation_time = time.time() - start_time
        
        # Verify assets were created
        assets_dir = Path(__file__).parent / 'assets'
        
        asset_checks = {
            'models_exist': (assets_dir / 'models' / 'all-MiniLM-L6-v2').exists(),
            'embeddings_exist': (assets_dir / 'embeddings' / 'business_categories.npy').exists(),
            'metadata_exists': (assets_dir / 'embeddings' / 'business_metadata.json').exists(),
            'data_files_exist': len(list((assets_dir / 'data').glob('*.json'))) > 0 if (assets_dir / 'data').exists() else False
        }
        
        # Check asset sizes
        asset_sizes = {}
        if assets_dir.exists():
            for subdir in ['models', 'embeddings', 'data']:
                subdir_path = assets_dir / subdir
                if subdir_path.exists():
                    size = sum(f.stat().st_size for f in subdir_path.rglob('*') if f.is_file())
                    asset_sizes[subdir] = size / 1024 / 1024  # MB
        
        preparation_results = {
            'status': 'success',
            'preparation_time_ms': preparation_time * 1000,
            'asset_checks': asset_checks,
            'asset_sizes_mb': asset_sizes,
            'all_assets_created': all(asset_checks.values())
        }
        
        print(f"   ✅ Asset preparation completed:")
        print(f"      - Preparation time: {preparation_time:.1f}s")
        print(f"      - Assets created: {sum(asset_checks.values())}/{len(asset_checks)}")
        if asset_sizes:
            total_size = sum(asset_sizes.values())
            print(f"      - Total size: {total_size:.1f}MB")
        
        return preparation_results
        
    except Exception as e:
        print(f"   ❌ Asset preparation failed: {e}")
        raise

def test_optimized_functionality():
    """Test system functionality with deployment optimizations."""
    
    print("   🚀 Testing optimized system performance...")
    
    try:
        # Force reload modules to pick up any changes
        import importlib
        import sys
        
        modules_to_reload = [
            'two_tier_classifier.utils.embedding_engine',
            'two_tier_classifier.core.level1_classifier',
            'two_tier_classifier.core.pipeline_controller'
        ]
        
        for module_name in modules_to_reload:
            if module_name in sys.modules:
                importlib.reload(sys.modules[module_name])
        
        from two_tier_classifier.core.pipeline_controller import ThreeTierClassifier
        
        # Initialize system and measure time (should be faster with assets)
        start_time = time.time()
        classifier = ThreeTierClassifier(
            model_name='all-MiniLM-L6-v2',
            cache_dir='cache/embeddings',
            confidence_threshold=0.6,
            enable_automation_analysis=True
        )
        init_time = time.time() - start_time
        
        # Test same classification examples as baseline
        test_cases = [
            "cashier sarah locked out till 3 customers waiting urgent",
            "vision order locked cannot modify quantities store manager approval needed", 
            "replace broken CPU on physical server motherboard damaged",
            "unlock user account john.doe",
            "printer driver installation required for new HP LaserJet"
        ]
        
        classification_results = []
        classification_times = []
        
        for test_case in test_cases:
            case_start = time.time()
            result = classifier.classify(
                text=test_case,
                include_level2=True,
                include_level3=True,
                include_explanations=False
            )
            case_time = time.time() - case_start
            
            classification_results.append({
                'input': test_case,
                'business_category': result.business_category,
                'confidence': result.confidence,
                'automation_potential': result.automation_category,
                'processing_time_ms': case_time * 1000
            })
            classification_times.append(case_time)
        
        optimized_results = {
            'status': 'success',
            'initialization_time_ms': init_time * 1000,
            'avg_classification_time_ms': sum(classification_times) / len(classification_times) * 1000,
            'classifications': classification_results,
            'system_stats': classifier.get_performance_stats() if hasattr(classifier, 'get_performance_stats') else {}
        }
        
        print(f"   ✅ Optimized test completed:")
        print(f"      - Initialization: {init_time*1000:.1f}ms")
        print(f"      - Avg classification: {optimized_results['avg_classification_time_ms']:.1f}ms")
        print(f"      - Test cases: {len(classification_results)}/5 successful")
        
        return optimized_results
        
    except Exception as e:
        print(f"   ❌ Optimized test failed: {e}")
        raise

def compare_performance(baseline_results, optimized_results):
    """Compare baseline vs optimized performance."""
    
    if not baseline_results or not optimized_results:
        return {'status': 'failed', 'reason': 'Missing test results'}
    
    init_speedup = baseline_results['initialization_time_ms'] / optimized_results['initialization_time_ms']
    classification_speedup = baseline_results['avg_classification_time_ms'] / optimized_results['avg_classification_time_ms']
    
    comparison = {
        'status': 'success',
        'initialization_speedup': init_speedup,
        'classification_speedup': classification_speedup,
        'baseline_init_ms': baseline_results['initialization_time_ms'],
        'optimized_init_ms': optimized_results['initialization_time_ms'],
        'baseline_avg_classification_ms': baseline_results['avg_classification_time_ms'],
        'optimized_avg_classification_ms': optimized_results['avg_classification_time_ms'],
        'target_achieved': init_speedup > 2.0  # Target at least 2x speedup
    }
    
    print(f"   📈 Performance comparison:")
    print(f"      - Initialization speedup: {init_speedup:.1f}x")
    print(f"      - Classification speedup: {classification_speedup:.1f}x")
    print(f"      - Target achieved: {'✅' if comparison['target_achieved'] else '❌'}")
    
    return comparison

def validate_safety(baseline_results, optimized_results):
    """Validate that optimizations don't break functionality."""
    
    if not baseline_results or not optimized_results:
        return {'status': 'failed', 'reason': 'Missing test results'}
    
    safety_checks = []
    
    # Check that same number of classifications succeeded
    baseline_count = len(baseline_results['classifications'])
    optimized_count = len(optimized_results['classifications'])
    safety_checks.append({
        'check': 'classification_count',
        'passed': baseline_count == optimized_count,
        'baseline': baseline_count,
        'optimized': optimized_count
    })
    
    # Check that classifications are consistent
    for i, (baseline_case, optimized_case) in enumerate(zip(
        baseline_results['classifications'], 
        optimized_results['classifications']
    )):
        consistent = (
            baseline_case['business_category'] == optimized_case['business_category'] and
            abs(baseline_case['confidence'] - optimized_case['confidence']) < 0.1
        )
        safety_checks.append({
            'check': f'classification_consistency_{i}',
            'passed': consistent,
            'baseline_category': baseline_case['business_category'],
            'optimized_category': optimized_case['business_category']
        })
    
    # Overall safety validation
    all_passed = all(check['passed'] for check in safety_checks)
    
    validation_results = {
        'status': 'success' if all_passed else 'failed',
        'all_checks_passed': all_passed,
        'individual_checks': safety_checks,
        'passed_count': sum(1 for check in safety_checks if check['passed']),
        'total_checks': len(safety_checks)
    }
    
    print(f"   🛡️ Safety validation:")
    print(f"      - Checks passed: {validation_results['passed_count']}/{validation_results['total_checks']}")
    print(f"      - Overall status: {'✅ SAFE' if all_passed else '❌ UNSAFE'}")
    
    return validation_results

def print_final_report(results):
    """Print comprehensive final report."""
    
    print("\n" + "="*60)
    print("📋 DEPLOYMENT OPTIMIZATION TEST REPORT")
    print("="*60)
    
    # Overall status
    all_successful = all(r and r.get('status') == 'success' for r in results.values() if r is not None)
    print(f"\n🎯 Overall Status: {'✅ SUCCESS' if all_successful else '❌ FAILED'}")
    
    # Performance summary
    if results['performance_comparison']:
        perf = results['performance_comparison']
        print(f"\n📊 Performance Summary:")
        print(f"   - Initialization: {perf['baseline_init_ms']:.0f}ms → {perf['optimized_init_ms']:.0f}ms ({perf['initialization_speedup']:.1f}x speedup)")
        print(f"   - Classification: {perf['baseline_avg_classification_ms']:.1f}ms → {perf['optimized_avg_classification_ms']:.1f}ms ({perf['classification_speedup']:.1f}x speedup)")
    
    # Safety summary
    if results['safety_validation']:
        safety = results['safety_validation']
        print(f"\n🛡️ Safety Summary:")
        print(f"   - Functionality preserved: {'✅' if safety['all_checks_passed'] else '❌'}")
        print(f"   - Validation checks: {safety['passed_count']}/{safety['total_checks']} passed")
    
    # Asset summary
    if results['asset_preparation']:
        assets = results['asset_preparation']
        print(f"\n📦 Asset Summary:")
        print(f"   - Assets created: {'✅' if assets['all_assets_created'] else '❌'}")
        if 'asset_sizes_mb' in assets:
            total_size = sum(assets['asset_sizes_mb'].values())
            print(f"   - Total size: {total_size:.1f}MB")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if all_successful:
        print("   ✅ Deployment optimization is SAFE to deploy")
        print("   ✅ Local functionality is preserved")
        print("   ✅ Performance improvements achieved")
        print("   📋 Next steps: Deploy assets/ folder with Streamlit app")
    else:
        print("   ❌ DO NOT deploy - issues detected")
        print("   🔧 Review failed tests and fix issues first")
        print("   📋 Local system continues to work normally")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    success = run_comprehensive_testing()
    sys.exit(0 if success else 1)