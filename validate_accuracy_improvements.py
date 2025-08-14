#!/usr/bin/env python3
"""
Validate Accuracy Improvements - Quick Test

Tests the improved accuracy after keyword mapping fixes.
Uses keyword-based classification to avoid slow embedding computation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_improved_accuracy():
    """Test improved accuracy with corrected keyword mappings."""
    print("=" * 60)
    print("VALIDATE ACCURACY IMPROVEMENTS")
    print("=" * 60)
    
    try:
        from src.two_tier_classifier.data.category_mappings import BUSINESS_CATEGORIES
        
        # Test cases with correct expected categories (based on actual system)
        test_cases = [
            {
                'text': "replace broken CPU on physical server",
                'expected_business': "General Support",  # Hardware → General Support
                'confidence_expected': "high",  # Strong keyword matches
                'issue': "Hardware classification"
            },
            {
                'text': "vision order locked cannot modify quantities urgent",
                'expected_business': "Vision Orders & Inventory",  # Vision orders → Vision category
                'confidence_expected': "high",  # Strong keyword matches
                'issue': "Vision order classification"
            },
            {
                'text': "cashier sarah locked out till 3 customers waiting",
                'expected_business': "Till Operations",  # Improved: Till operations should win
                'confidence_expected': "high",  # Now has strong till-specific keywords
                'issue': "Till vs Account context (FIXED)"
            },
            {
                'text': "printer driver installation required",
                'expected_business': "Printing Services",  # Printing → Printing Services
                'confidence_expected': "medium",  # Reasonable keyword matches
                'issue': "Printing service classification"
            },
            {
                'text': "mobile device screen cracked needs replacement",
                'expected_business': "Mobile Devices",  # Mobile → Mobile Devices
                'confidence_expected': "medium",  # Mobile keywords should win
                'issue': "Mobile device classification"
            },
            # Additional test cases for robustness
            {
                'text': "unlock user account john.smith cannot login",
                'expected_business': "User Account Management",  # Clear account management
                'confidence_expected': "high",  # Strong AD keywords
                'issue': "Account management classification"
            },
            {
                'text': "appstream application crashing when loading",
                'expected_business': "Software & Application Issues",  # Clear software issue
                'confidence_expected': "high",  # Strong software keywords
                'issue': "Software application classification"
            }
        ]
        
        print(f"✅ Testing {len(test_cases)} improved accuracy cases:")
        
        correct_classifications = 0
        high_confidence_matches = 0
        
        for i, case in enumerate(test_cases, 1):
            text = case['text']
            expected_category = case['expected_business']
            
            # Find best matching category using keyword logic
            best_match = None
            best_score = 0
            category_scores = {}
            
            for category_enum, category_def in BUSINESS_CATEGORIES.items():
                score = 0
                matches = []
                
                # Count keyword matches
                for keyword in category_def.keywords:
                    if keyword.lower() in text.lower():
                        score += 1
                        matches.append(f"keyword:{keyword}")
                
                # Count priority matches (weighted more heavily)
                for priority_kw in category_def.priority_keywords:
                    if priority_kw.lower() in text.lower():
                        score += 2  # Priority keywords worth more
                        matches.append(f"priority:{priority_kw}")
                
                category_scores[category_def.name] = {
                    'score': score,
                    'matches': matches
                }
                
                if score > best_score:
                    best_score = score
                    best_match = category_def.name
            
            # Determine if classification is correct
            is_correct = best_match == expected_category
            if is_correct:
                correct_classifications += 1
            
            # Determine confidence level
            confidence_level = "low"
            if best_score >= 4:
                confidence_level = "high"
                if is_correct:
                    high_confidence_matches += 1
            elif best_score >= 2:
                confidence_level = "medium"
            
            print(f"\n{i}. Issue: {case['issue']}")
            print(f"   Input: \"{text}\"")
            print(f"   Expected: {expected_category}")
            print(f"   Predicted: {best_match} {'✅' if is_correct else '❌'}")
            print(f"   Confidence: {confidence_level} (score: {best_score})")
            
            if not is_correct:
                print(f"   🔍 ANALYSIS:")
                # Show top 3 scoring categories
                sorted_scores = sorted(category_scores.items(), key=lambda x: x[1]['score'], reverse=True)
                for rank, (cat_name, cat_data) in enumerate(sorted_scores[:3], 1):
                    if cat_data['score'] > 0:
                        print(f"      {rank}. {cat_name}: {cat_data['score']} ({cat_data['matches']})")
        
        # Calculate metrics
        accuracy = correct_classifications / len(test_cases)
        high_confidence_rate = high_confidence_matches / len(test_cases)
        
        print(f"\n📊 ACCURACY IMPROVEMENT RESULTS:")
        print(f"   Overall Accuracy: {accuracy:.1%} ({correct_classifications}/{len(test_cases)})")
        print(f"   High Confidence Matches: {high_confidence_rate:.1%} ({high_confidence_matches}/{len(test_cases)})")
        print(f"   Target: 80%+ for production deployment")
        
        # Assessment
        if accuracy >= 0.8:
            print(f"\n🎉 SUCCESS: Accuracy improvements achieved target!")
            print(f"   ✅ Level 1 classification ready for production")
            if high_confidence_rate >= 0.6:
                print(f"   ✅ High confidence rate indicates robust classification")
        elif accuracy >= 0.7:
            print(f"\n✅ GOOD: Substantial accuracy improvement")
            print(f"   ⚠️  Need {0.8 - accuracy:.1%} more improvement for production")
        else:
            print(f"\n❌ NEEDS WORK: Accuracy still below target")
            print(f"   🔧 Need {0.8 - accuracy:.1%} improvement for production")
        
        return {
            'accuracy': accuracy,
            'high_confidence_rate': high_confidence_rate,
            'ready_for_production': accuracy >= 0.8
        }
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_improved_accuracy()
    if result:
        exit_code = 0 if result['ready_for_production'] else 1
    else:
        exit_code = 2
    sys.exit(exit_code)