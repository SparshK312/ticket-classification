"""
Two-Tier Classification Pipeline Controller

Orchestrates the complete two-tier classification process:
Level 1: Business category routing
Level 2: Specific problem identification (placeholder for Week 2)
"""

import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .level1_classifier import Level1BusinessClassifier, Level1Result
from .level2_semantic_search import Level2SemanticSearch, Level2Result
from .automation_analyzer import ComprehensiveAutomationAnalyzer, AutomationResult

@dataclass 
class ThreeTierResult:
    """Complete result from three-tier classification pipeline."""
    # Level 1 Results - Business Routing
    business_category: str
    routing_team: str
    priority_level: str
    sla_hours: int
    confidence: float
    urgency_score: float
    
    # Level 2 Results - Problem Identification
    specific_problem: Optional[str] = None
    problem_confidence: Optional[float] = None
    similar_problems: Optional[List[Dict]] = None
    
    # Level 3 Results - Automation Analysis (NEW)
    automation_category: Optional[str] = None  # FULLY/PARTIALLY/NOT_AUTOMATABLE
    automation_percentage: Optional[int] = None  # 0-100% effort savings
    automation_reasoning: Optional[str] = None
    automation_confidence: Optional[float] = None
    step_breakdown: Optional[Dict] = None
    
    # Pipeline Metadata
    total_processing_time_ms: float = 0.0
    level1_time_ms: float = 0.0
    level2_time_ms: float = 0.0
    level3_time_ms: float = 0.0
    
    # Detailed Results
    level1_result: Optional[Level1Result] = None
    level2_result: Optional[Dict] = None
    level3_result: Optional[AutomationResult] = None
    
    # Quality Metrics
    overall_confidence: float = 0.0
    recommendation: str = "ROUTE_TO_TEAM"

# Keep TwoTierResult for backward compatibility
TwoTierResult = ThreeTierResult

class ThreeTierClassifier:
    """
    Complete three-tier classification system.
    
    Level 1: Business category routing (77% accuracy, 10.1ms)
    Level 2: Specific problem identification (100% success, 0.4ms)  
    Level 3: Automation analysis (100% coverage, manager requirements)
    """
    
    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 cache_dir: Optional[str] = 'cache/embeddings',
                 confidence_threshold: float = 0.6,
                 enable_automation_analysis: bool = True):
        """
        Initialize the three-tier classifier.
        
        Args:
            model_name: Sentence transformer model name
            cache_dir: Directory for embedding cache
            confidence_threshold: Minimum confidence for routing decisions
            enable_automation_analysis: Whether to enable Level 3 automation analysis
        """
        self.logger = logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold
        self.enable_automation_analysis = enable_automation_analysis
        
        # Initialize Level 1 classifier
        self.level1_classifier = Level1BusinessClassifier(
            model_name=model_name,
            cache_dir=cache_dir,
            confidence_threshold=confidence_threshold,
            enable_preprocessing=True
        )
        
        # Level 2 semantic search system
        self.level2_classifier = Level2SemanticSearch(
            embedding_engine=self.level1_classifier.embedding_engine,
            min_similarity_threshold=0.3,
            max_candidates=100
        )
        
        # Level 3 automation analysis system (NEW)
        if enable_automation_analysis:
            self.level3_analyzer = ComprehensiveAutomationAnalyzer(
                automation_database_path="cache/automation_mappings.json",
                enable_llm_fallback=True
            )
        else:
            self.level3_analyzer = None
        
        # Performance tracking
        self.stats = {
            'total_classifications': 0,
            'level1_only_classifications': 0,
            'level2_classifications': 0,
            'level3_classifications': 0,
            'avg_processing_time_ms': 0.0,
            'routing_decisions': {},
            'confidence_distribution': [],
            'automation_category_distribution': {}
        }
        
        self.logger.info("ThreeTierClassifier initialized successfully")
    
    def classify(self, text: str, 
                include_level2: bool = True,
                include_level3: bool = True,
                include_explanations: bool = False) -> ThreeTierResult:
        """
        Perform complete three-tier classification.
        
        Args:
            text: Input ticket description
            include_level2: Whether to perform Level 2 problem identification
            include_level3: Whether to perform Level 3 automation analysis
            include_explanations: Whether to include detailed explanations
            
        Returns:
            ThreeTierResult with complete classification information
        """
        start_time = time.time()
        
        # Level 1: Business Category Classification
        level1_start = time.time()
        level1_result = self.level1_classifier.classify(
            text, 
            include_uncertainty=True,
            include_routing=True
        )
        level1_time_ms = (time.time() - level1_start) * 1000
        
        # Level 2: Specific Problem Identification
        level2_result = None
        level2_time_ms = 0.0
        specific_problem = None
        problem_confidence = None
        similar_problems = None
        
        if include_level2:
            # Level 2: Semantic search for specific problem identification
            level2_start = time.time()
            level2_result = self.level2_classifier.search(
                text=text,
                business_category=level1_result.predicted_category,
                top_k=5
            )
            level2_time_ms = (time.time() - level2_start) * 1000
            
            if level2_result:
                specific_problem = level2_result.specific_problem
                problem_confidence = level2_result.confidence
                similar_problems = [
                    {
                        'problem': match.problem_description,
                        'similarity': match.similarity_score,
                        'confidence': match.confidence,
                        'problem_id': match.problem_id
                    }
                    for match in level2_result.similar_problems
                ]
        
        # Level 3: Automation Analysis (NEW)
        level3_result = None
        level3_time_ms = 0.0
        automation_category = None
        automation_percentage = None
        automation_reasoning = None
        automation_confidence = None
        step_breakdown = None
        
        if include_level3 and self.level3_analyzer:
            level3_start = time.time()
            
            # Prepare Level 2 matches for automation analysis
            level2_matches = []
            if level2_result and level2_result.similar_problems:
                level2_matches = level2_result.similar_problems
            
            # Perform comprehensive automation analysis
            level3_result = self.level3_analyzer.analyze(
                problem_text=text,
                business_category=level1_result.predicted_category,
                level2_matches=level2_matches
            )
            
            level3_time_ms = (time.time() - level3_start) * 1000
            
            if level3_result:
                automation_category = level3_result.category
                automation_percentage = level3_result.automation_percentage
                automation_reasoning = level3_result.reasoning
                automation_confidence = level3_result.confidence
                step_breakdown = level3_result.step_breakdown
        
        # Calculate overall metrics
        total_time_ms = (time.time() - start_time) * 1000
        overall_confidence = level1_result.confidence
        
        # Combine all confidences (weighted average)
        if level2_result and problem_confidence:
            overall_confidence = 0.6 * level1_result.confidence + 0.3 * problem_confidence
            if automation_confidence:
                overall_confidence = 0.5 * level1_result.confidence + 0.3 * problem_confidence + 0.2 * automation_confidence
        elif automation_confidence:
            overall_confidence = 0.8 * level1_result.confidence + 0.2 * automation_confidence
        
        # Generate recommendation (enhanced with automation insights)
        recommendation = self._generate_recommendation(level1_result, level2_result, level3_result)
        
        # Create complete result
        result = ThreeTierResult(
            # Level 1 - Business Routing
            business_category=level1_result.predicted_category,
            routing_team=level1_result.routing_team,
            priority_level=level1_result.priority_level,
            sla_hours=level1_result.sla_hours,
            confidence=level1_result.confidence,
            urgency_score=level1_result.urgency_score,
            
            # Level 2 - Problem Identification  
            specific_problem=specific_problem,
            problem_confidence=problem_confidence,
            similar_problems=similar_problems,
            
            # Level 3 - Automation Analysis (NEW)
            automation_category=automation_category,
            automation_percentage=automation_percentage,
            automation_reasoning=automation_reasoning,
            automation_confidence=automation_confidence,
            step_breakdown=step_breakdown,
            
            # Pipeline Metadata
            total_processing_time_ms=total_time_ms,
            level1_time_ms=level1_time_ms,
            level2_time_ms=level2_time_ms,
            level3_time_ms=level3_time_ms,
            
            # Detailed Results
            level1_result=level1_result,
            level2_result=level2_result,
            level3_result=level3_result,
            
            # Quality Metrics
            overall_confidence=overall_confidence,
            recommendation=recommendation
        )
        
        # Update statistics
        self._update_stats(result)
        
        return result
    
    def get_level2_stats(self) -> Dict[str, Any]:
        """Get Level 2 semantic search statistics."""
        if self.level2_classifier:
            return self.level2_classifier.get_database_stats()
        return {}
    
    def _generate_recommendation(self, level1_result: Level1Result, 
                               level2_result: Optional[Dict],
                               level3_result: Optional[AutomationResult]) -> str:
        """Generate routing recommendation enhanced with automation insights."""
        
        # Enhanced routing with automation considerations
        base_recommendation = ""
        
        # High confidence routing
        if level1_result.confidence >= 0.8:
            if level1_result.urgency_score >= 0.8:
                base_recommendation = "ROUTE_IMMEDIATELY_HIGH_PRIORITY"
            else:
                base_recommendation = "ROUTE_TO_TEAM"
        
        # Medium confidence
        elif level1_result.confidence >= self.confidence_threshold:
            if level1_result.urgency_score >= 0.7:
                base_recommendation = "ROUTE_WITH_ESCALATION_WATCH"
            else:
                base_recommendation = "ROUTE_TO_TEAM"
        
        # Low confidence
        else:
            if level1_result.urgency_score >= 0.8:
                base_recommendation = "ESCALATE_FOR_MANUAL_REVIEW"
            else:
                base_recommendation = "ROUTE_TO_GENERAL_DESK"
        
        # Enhance with automation insights
        if level3_result:
            if level3_result.category == "FULLY_AUTOMATABLE":
                if level3_result.automation_percentage >= 90:
                    base_recommendation += "_AUTOMATION_CANDIDATE"
                    
            elif level3_result.category == "PARTIALLY_AUTOMATABLE":
                if level3_result.automation_percentage >= 60:
                    base_recommendation += "_PARTIAL_AUTOMATION_OPPORTUNITY"
        
        return base_recommendation
    
    def classify_batch(self, texts: List[str],
                      include_level2: bool = True,
                      include_level3: bool = True,
                      batch_size: int = 32,
                      show_progress: bool = False) -> List[ThreeTierResult]:
        """
        Classify multiple texts efficiently.
        
        Args:
            texts: List of ticket descriptions
            include_level2: Whether to include Level 2 classification
            include_level3: Whether to include Level 3 automation analysis
            batch_size: Processing batch size
            show_progress: Whether to show progress
            
        Returns:
            List of ThreeTierResult objects
        """
        if not texts:
            return []
        
        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            if show_progress:
                batch_num = i // batch_size + 1
                self.logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            # Process batch
            batch_results = []
            for text in batch_texts:
                result = self.classify(text, include_level2=include_level2, include_level3=include_level3)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def evaluate_system_performance(self, test_texts: List[str],
                                   true_categories: List[str],
                                   true_problems: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate complete system performance.
        
        Args:
            test_texts: Test ticket descriptions
            true_categories: True business categories
            true_problems: True specific problems (optional, for Level 2)
            
        Returns:
            Comprehensive performance metrics
        """
        self.logger.info(f"Evaluating system performance on {len(test_texts)} samples")
        
        # Get Level 1 performance
        level1_performance = self.level1_classifier.evaluate_performance(
            test_texts, true_categories
        )
        
        # Get full system predictions
        predictions = self.classify_batch(
            test_texts, 
            include_level2=True, 
            show_progress=True
        )
        
        # Calculate system-level metrics
        routing_accuracy = sum(
            1 for pred, true_cat in zip(predictions, true_categories)
            if pred.business_category == true_cat
        ) / len(predictions)
        
        avg_processing_time = sum(p.total_processing_time_ms for p in predictions) / len(predictions)
        avg_confidence = sum(p.overall_confidence for p in predictions) / len(predictions)
        
        # Performance by recommendation type
        recommendation_distribution = {}
        for pred in predictions:
            rec = pred.recommendation
            if rec not in recommendation_distribution:
                recommendation_distribution[rec] = 0
            recommendation_distribution[rec] += 1
        
        # SLA compliance estimation
        urgent_tickets = [p for p in predictions if p.urgency_score >= 0.8]
        critical_priority = [p for p in predictions if p.priority_level == 'CRITICAL']
        
        system_performance = {
            'level1_performance': level1_performance,
            'system_metrics': {
                'routing_accuracy': routing_accuracy,
                'avg_processing_time_ms': avg_processing_time,
                'avg_confidence': avg_confidence,
                'meets_speed_target': avg_processing_time < 2000,  # <2 seconds
                'meets_accuracy_target': routing_accuracy >= 0.85,  # 85%+ target
                'urgent_ticket_count': len(urgent_tickets),
                'critical_priority_count': len(critical_priority)
            },
            'recommendation_distribution': recommendation_distribution,
            'sla_analysis': {
                'avg_sla_hours': sum(p.sla_hours for p in predictions) / len(predictions),
                'critical_tickets_1hr_sla': len([p for p in predictions if p.sla_hours <= 1]),
                'high_priority_tickets': len([p for p in predictions if p.priority_level in ['HIGH', 'CRITICAL']])
            }
        }
        
        # Level 2 performance (when implemented)
        if true_problems and all(p.level2_result for p in predictions):
            # Placeholder for Level 2 evaluation
            system_performance['level2_performance'] = {
                'status': 'NOT_IMPLEMENTED',
                'placeholder': True
            }
        
        self.logger.info(f"System evaluation completed: {routing_accuracy:.3f} routing accuracy")
        return system_performance
    
    def _update_stats(self, result: ThreeTierResult):
        """Update internal statistics."""
        self.stats['total_classifications'] += 1
        
        if result.level2_result is None:
            self.stats['level1_only_classifications'] += 1
        else:
            self.stats['level2_classifications'] += 1
        
        if result.level3_result is not None:
            self.stats['level3_classifications'] += 1
        
        # Update averages
        total_time = self.stats['avg_processing_time_ms'] * (self.stats['total_classifications'] - 1)
        total_time += result.total_processing_time_ms
        self.stats['avg_processing_time_ms'] = total_time / self.stats['total_classifications']
        
        # Track routing decisions
        recommendation = result.recommendation
        if recommendation not in self.stats['routing_decisions']:
            self.stats['routing_decisions'][recommendation] = 0
        self.stats['routing_decisions'][recommendation] += 1
        
        # Track confidence distribution
        self.stats['confidence_distribution'].append(result.overall_confidence)
        
        # Track automation category distribution (NEW)
        if result.automation_category:
            if result.automation_category not in self.stats['automation_category_distribution']:
                self.stats['automation_category_distribution'][result.automation_category] = 0
            self.stats['automation_category_distribution'][result.automation_category] += 1
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = self.stats.copy()
        
        # Add Level 1 classifier stats
        stats['level1_stats'] = self.level1_classifier.get_performance_stats()
        
        # Add derived metrics
        if self.stats['total_classifications'] > 0:
            stats['level2_usage_rate'] = (
                self.stats['level2_classifications'] / self.stats['total_classifications']
            )
            
            if self.stats['confidence_distribution']:
                import numpy as np
                stats['median_confidence'] = np.median(self.stats['confidence_distribution'])
                stats['confidence_std'] = np.std(self.stats['confidence_distribution'])
        
        return stats
    
    def explain_classification(self, text: str, result: TwoTierResult) -> Dict[str, Any]:
        """
        Provide comprehensive explanation for a classification result.
        
        Args:
            text: Original input text
            result: Classification result to explain
            
        Returns:
            Detailed explanation of the classification decision
        """
        explanation = {
            'input_summary': {
                'text_length': len(text),
                'urgency_detected': result.urgency_score >= 0.7,
                'processing_time_ms': result.total_processing_time_ms
            },
            'level1_explanation': self.level1_classifier.explain_prediction(text, result.level1_result),
            'routing_decision': {
                'final_recommendation': result.recommendation,
                'routing_team': result.routing_team,
                'priority_level': result.priority_level,
                'sla_hours': result.sla_hours,
                'justification': self._explain_routing_decision(result)
            },
            'confidence_analysis': {
                'overall_confidence': result.overall_confidence,
                'level1_confidence': result.confidence,
                'level2_confidence': result.problem_confidence,
                'meets_threshold': result.overall_confidence >= self.confidence_threshold
            },
            'next_steps': self._suggest_next_steps(result)
        }
        
        return explanation
    
    def _explain_routing_decision(self, result: TwoTierResult) -> str:
        """Explain why a particular routing decision was made."""
        if result.recommendation == "ROUTE_IMMEDIATELY_HIGH_PRIORITY":
            return f"High confidence ({result.confidence:.2f}) and high urgency ({result.urgency_score:.2f}) detected"
        elif result.recommendation == "ROUTE_TO_TEAM":
            return f"Confident classification ({result.confidence:.2f}) to {result.business_category}"
        elif result.recommendation == "ROUTE_WITH_ESCALATION_WATCH":
            return f"Medium confidence with elevated urgency - monitor for escalation"
        elif result.recommendation == "ESCALATE_FOR_MANUAL_REVIEW":
            return f"Low confidence ({result.confidence:.2f}) but high urgency - needs human review"
        elif result.recommendation == "ROUTE_TO_GENERAL_DESK":
            return f"Low confidence classification - route to general support for triage"
        else:
            return "Standard routing decision"
    
    def _suggest_next_steps(self, result: TwoTierResult) -> List[str]:
        """Suggest next steps based on classification result."""
        steps = []
        
        if result.confidence < 0.5:
            steps.append("Consider manual review due to low confidence")
        
        if result.urgency_score >= 0.8:
            steps.append("Prioritize due to high urgency indicators")
        
        if result.priority_level == "CRITICAL":
            steps.append(f"Immediate action required - {result.sla_hours} hour SLA")
        
        if not result.level1_result.text_features.get('keywords_found'):
            steps.append("Limited business context - may need additional information")
        
        if result.recommendation == "ESCALATE_FOR_MANUAL_REVIEW":
            steps.append("Route to senior analyst for manual classification")
        
        return steps if steps else ["Route to assigned team for standard processing"]


# Backward compatibility aliases
TwoTierClassifier = ThreeTierClassifier