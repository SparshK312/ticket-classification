#!/usr/bin/env python3
"""
Three-Tier Classification Engine for Demo UI
Integrates our production ThreeTierClassifier with the Streamlit demo interface
"""

import sys
import os
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
import logging

# Add src to path for imports
current_dir = Path(__file__).parent
repo_root = current_dir.parent
src_path = repo_root / 'src'
sys.path.insert(0, str(src_path))

# Import our production three-tier system
try:
    from two_tier_classifier.core.pipeline_controller import ThreeTierClassifier, ThreeTierResult
except ImportError as e:
    print(f"❌ Failed to import ThreeTierClassifier: {e}")
    print(f"Attempted to import from: {src_path}")
    print("Make sure the three-tier system is properly implemented")
    sys.exit(1)

@dataclass
class DemoResult:
    """Adapter result structure for demo UI compatibility"""
    method: str
    problem_statement: str
    confidence_score: float
    category: str
    details: Dict
    automation_potential: str = "Unknown"
    automation_confidence: float = 0.0
    automation_reasoning: str = ""
    
    # Additional fields for UI display
    business_category: str = ""
    routing_team: str = ""
    priority_level: str = ""
    sla_hours: int = 0
    automation_percentage: int = 0

class ThreeTierDemoEngine:
    """
    Demo-compatible wrapper around our production ThreeTierClassifier.
    
    Provides the same interface as the original TicketClassificationEngine
    but uses our advanced three-tier system underneath.
    """
    
    def __init__(self, data_loader=None, use_embeddings: bool = True, use_llm: bool = False):
        """Initialize the three-tier demo engine"""
        self.logger = logging.getLogger(__name__)
        self.data_loader = data_loader  # For compatibility, but not needed
        
        try:
            # Initialize our production three-tier classifier
            self.classifier = ThreeTierClassifier(
                model_name='all-MiniLM-L6-v2',
                cache_dir=str(repo_root / 'cache' / 'embeddings'),
                confidence_threshold=0.6,
                enable_automation_analysis=True
            )
            
            self.logger.info("✅ ThreeTierClassifier initialized successfully for demo")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize ThreeTierClassifier: {e}")
            raise
    
    def classify_ticket(self, description: str) -> DemoResult:
        """
        Main classification method - adapts ThreeTierResult to DemoResult format
        """
        self.logger.info(f"🎯 Demo classifying: {description[:50]}...")
        
        if not description.strip():
            return DemoResult(
                method="Error",
                problem_statement="Empty Description", 
                confidence_score=0.0,
                category="Error",
                details={'error': 'No description provided'}
            )
        
        try:
            # Use our production three-tier classifier
            three_tier_result = self.classifier.classify(
                text=description,
                include_level2=True,
                include_level3=True,
                include_explanations=False
            )
            
            # Convert ThreeTierResult to DemoResult for UI compatibility
            demo_result = self._convert_to_demo_result(three_tier_result, description)
            
            self.logger.info(f"✅ Demo classification complete: {demo_result.method}")
            return demo_result
            
        except Exception as e:
            self.logger.error(f"❌ Classification failed: {e}")
            return DemoResult(
                method="Error",
                problem_statement=f"Classification Error: {str(e)[:100]}",
                confidence_score=0.0,
                category="Error",
                details={'error': str(e)}
            )
    
    def _convert_to_demo_result(self, three_tier_result: ThreeTierResult, original_input: str) -> DemoResult:
        """Convert ThreeTierResult to demo-compatible DemoResult"""
        
        # Determine classification method based on processing times and results
        method = self._determine_classification_method(three_tier_result)
        
        # Get problem statement - use specific problem if available, otherwise business category
        problem_statement = (
            three_tier_result.specific_problem 
            if three_tier_result.specific_problem 
            else f"{three_tier_result.business_category} - {original_input[:100]}"
        )
        
        # Map automation category to demo format
        automation_potential = three_tier_result.automation_category or "Unknown"
        automation_percentage = three_tier_result.automation_percentage or 0
        
        # Create detailed information for UI
        details = {
            # Level 1 Information
            'business_category': three_tier_result.business_category,
            'routing_team': three_tier_result.routing_team,
            'priority_level': three_tier_result.priority_level,
            'sla_hours': three_tier_result.sla_hours,
            'urgency_score': three_tier_result.urgency_score,
            
            # Level 2 Information
            'specific_problem': three_tier_result.specific_problem,
            'problem_confidence': three_tier_result.problem_confidence,
            'similar_problems_count': len(three_tier_result.similar_problems or []),
            
            # Level 3 Information
            'automation_method': three_tier_result.level3_result.layer_used if three_tier_result.level3_result else 'none',
            'automation_percentage': automation_percentage,
            'step_breakdown': three_tier_result.step_breakdown,
            
            # Performance Information
            'total_processing_time_ms': three_tier_result.total_processing_time_ms,
            'level1_time_ms': three_tier_result.level1_time_ms,
            'level2_time_ms': three_tier_result.level2_time_ms, 
            'level3_time_ms': three_tier_result.level3_time_ms,
            
            # Recommendation
            'recommendation': three_tier_result.recommendation,
            'overall_confidence': three_tier_result.overall_confidence,
            
            # Demo-specific additions
            'implementation_assessment': self._generate_implementation_assessment(three_tier_result),
            'roi_estimate': self._generate_roi_estimate(three_tier_result),
            'business_priority': self._determine_business_priority(three_tier_result),
            'implementation_complexity': self._determine_implementation_complexity(three_tier_result)
        }
        
        return DemoResult(
            method=method,
            problem_statement=problem_statement,
            confidence_score=three_tier_result.overall_confidence,
            category=three_tier_result.business_category,
            automation_potential=automation_potential,
            automation_confidence=three_tier_result.automation_confidence or 0.0,
            automation_reasoning=three_tier_result.automation_reasoning or "No automation analysis available",
            details=details,
            
            # Additional fields for enhanced UI display
            business_category=three_tier_result.business_category,
            routing_team=three_tier_result.routing_team,
            priority_level=three_tier_result.priority_level,
            sla_hours=three_tier_result.sla_hours,
            automation_percentage=automation_percentage
        )
    
    def _determine_classification_method(self, result: ThreeTierResult) -> str:
        """Determine which method was primarily used for classification"""
        
        # Check processing times to understand which level did most work
        if result.level2_time_ms > 0 and result.specific_problem:
            if result.automation_category == "FULLY_AUTOMATABLE" and "pattern" in (result.automation_reasoning or "").lower():
                return "Hardcoded Pattern"
            else:
                return "Semantic Clustering"
        elif result.level1_time_ms > 5:  # If Level 1 took significant time
            return "Business Classification"
        else:
            return "Three-Tier Analysis"
    
    def _generate_implementation_assessment(self, result: ThreeTierResult) -> str:
        """Generate implementation assessment for demo"""
        
        if result.automation_category == "FULLY_AUTOMATABLE":
            if "unlock" in (result.automation_reasoning or "").lower():
                return "Technical implementation using Unlock-ADAccount PowerShell command"
            elif "restart" in (result.automation_reasoning or "").lower():
                return "Service restart automation via PowerShell or system scripts"
            else:
                return "Automation scripts can handle this problem with minimal human oversight"
                
        elif result.automation_category == "PARTIALLY_AUTOMATABLE":
            return f"Hybrid approach: automated data gathering and diagnostics, manual analysis required. Estimated {result.automation_percentage}% of workflow can be automated"
            
        elif result.automation_category == "NOT_AUTOMATABLE":
            return "Requires human intervention or physical actions that cannot be automated"
            
        return "Implementation assessment requires additional analysis"
    
    def _generate_roi_estimate(self, result: ThreeTierResult) -> str:
        """Generate ROI estimate for demo"""
        
        priority = result.priority_level
        automation_pct = result.automation_percentage or 0
        
        if result.automation_category == "FULLY_AUTOMATABLE":
            if priority == "CRITICAL":
                return f"High ROI: Save 45-60 minutes per ticket, {automation_pct}% automation achievable"
            else:
                return f"Moderate ROI: Save 15-30 minutes per ticket, {automation_pct}% automation achievable"
                
        elif result.automation_category == "PARTIALLY_AUTOMATABLE":
            return f"Variable ROI: Save {automation_pct}% of manual effort, approximately {automation_pct * 0.3:.0f} minutes per ticket"
            
        elif result.automation_category == "NOT_AUTOMATABLE":
            return "Limited ROI: Focus on process optimization and knowledge base creation"
            
        return "ROI assessment requires business case analysis"
    
    def _determine_business_priority(self, result: ThreeTierResult) -> str:
        """Determine business priority for demo"""
        
        if result.priority_level == "CRITICAL":
            return "HIGH"
        elif result.priority_level == "HIGH":
            return "HIGH"
        elif result.automation_category == "FULLY_AUTOMATABLE":
            return "MEDIUM"
        else:
            return "LOW"
    
    def _determine_implementation_complexity(self, result: ThreeTierResult) -> str:
        """Determine implementation complexity for demo"""
        
        if result.automation_category == "FULLY_AUTOMATABLE":
            if "unlock" in (result.automation_reasoning or "").lower():
                return "SIMPLE"
            else:
                return "MODERATE"
                
        elif result.automation_category == "PARTIALLY_AUTOMATABLE":
            automation_pct = result.automation_percentage or 0
            if automation_pct >= 70:
                return "MODERATE" 
            else:
                return "COMPLEX"
                
        elif result.automation_category == "NOT_AUTOMATABLE":
            return "NOT_APPLICABLE"
            
        return "MODERATE"

def test_three_tier_demo_engine():
    """Test the three-tier demo engine"""
    print("🧪 Testing Three-Tier Demo Engine Integration...")
    
    try:
        # Initialize engine
        engine = ThreeTierDemoEngine(use_embeddings=True, use_llm=False)
        print("✅ Engine initialized successfully")
        
        # Test cases that should work with our system
        test_cases = [
            "unlock user account john.doe",
            "till crashed during busy period customers waiting", 
            "replace broken CPU on physical server",
            "vision order locked cannot modify quantities urgent",
            "chip pin device offline payment failed"
        ]
        
        print(f"\n🔍 Testing {len(test_cases)} classification scenarios:")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test {i}: {test_case} ---")
            
            try:
                result = engine.classify_ticket(test_case)
                
                print(f"✅ Method: {result.method}")
                print(f"✅ Problem: {result.problem_statement}")
                print(f"✅ Business Category: {result.business_category}")
                print(f"✅ Routing Team: {result.routing_team}")
                print(f"✅ Confidence: {result.confidence_score:.1%}")
                print(f"✅ Automation: {result.automation_potential} ({result.automation_percentage}%)")
                print(f"✅ Processing Time: {result.details.get('total_processing_time_ms', 0):.1f}ms")
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
        
        print("\n🎉 Three-tier demo engine integration test complete!")
        return True
        
    except Exception as e:
        print(f"❌ Demo engine initialization failed: {e}")
        return False

if __name__ == "__main__":
    test_three_tier_demo_engine()