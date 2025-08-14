"""
Confidence calibration for classification results.

Provides calibrated confidence scores that reflect true prediction accuracy,
including uncertainty estimation and reliability scoring.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from sklearn.calibration import calibration_curve
from scipy import stats
import warnings

class ConfidenceCalibrator:
    """Advanced confidence calibration for classification results."""
    
    def __init__(self, 
                 calibration_method: str = 'isotonic',
                 min_confidence: float = 0.1,
                 max_confidence: float = 0.95):
        """
        Initialize confidence calibrator.
        
        Args:
            calibration_method: Method for confidence calibration ('isotonic', 'platt', 'temperature')
            min_confidence: Minimum confidence score to return
            max_confidence: Maximum confidence score to return
        """
        self.logger = logging.getLogger(__name__)
        self.calibration_method = calibration_method
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        
        # Calibration data
        self.is_calibrated = False
        self.calibration_data = {
            'raw_scores': [],
            'true_labels': [],
            'calibrated_mapping': None
        }
        
        # Performance tracking
        self.stats = {
            'predictions_calibrated': 0,
            'avg_confidence_before': 0.0,
            'avg_confidence_after': 0.0,
            'calibration_accuracy': 0.0
        }
    
    def calibrate_similarity_scores(self, similarities: List[float], 
                                   text_features: Optional[Dict] = None) -> List[float]:
        """
        Calibrate similarity scores to confidence probabilities.
        
        Args:
            similarities: Raw similarity scores from embedding comparison
            text_features: Additional text features for confidence adjustment
            
        Returns:
            List of calibrated confidence scores
        """
        if not similarities:
            return []
        
        similarities = np.array(similarities)
        
        # Basic confidence mapping from similarities
        confidences = self._similarity_to_confidence(similarities)
        
        # Apply text-based adjustments
        if text_features:
            confidences = self._adjust_for_text_features(confidences, text_features)
        
        # Apply learned calibration if available
        if self.is_calibrated:
            confidences = self._apply_calibration(confidences)
        
        # Ensure bounds
        confidences = np.clip(confidences, self.min_confidence, self.max_confidence)
        
        # Update stats
        self.stats['predictions_calibrated'] += len(confidences)
        
        return confidences.tolist()
    
    def _similarity_to_confidence(self, similarities: np.ndarray) -> np.ndarray:
        """Convert similarity scores to initial confidence estimates."""
        # Sigmoid transformation to map similarities to [0,1]
        # Similarities are typically in [-1, 1] for cosine similarity
        
        # Shift and scale to emphasize high similarities
        # This makes the confidence more discriminative
        shifted = (similarities + 1) / 2  # Map [-1,1] to [0,1]
        
        # Apply power transformation to emphasize high similarities
        powered = np.power(shifted, 2)
        
        # Apply sigmoid for smooth calibration
        # Temperature scaling for better calibration
        temperature = 2.0
        confidences = 1 / (1 + np.exp(-temperature * (powered - 0.5)))
        
        return confidences
    
    def _adjust_for_text_features(self, confidences: np.ndarray, 
                                 text_features: Dict) -> np.ndarray:
        """Adjust confidence based on text characteristics."""
        adjustments = np.ones_like(confidences)
        
        # Length-based adjustment
        if 'text_length' in text_features:
            length = text_features['text_length']
            if length < 5:  # Very short text
                adjustments *= 0.8
            elif length < 10:  # Short text
                adjustments *= 0.9
            elif length > 100:  # Very long text
                adjustments *= 0.95
        
        # Keyword presence adjustment
        if 'keywords_found' in text_features:
            num_keywords = len(text_features['keywords_found'])
            if num_keywords == 0:  # No business keywords
                adjustments *= 0.7
            elif num_keywords >= 3:  # Many keywords
                adjustments *= 1.1
        
        # Urgency indicators adjustment
        if 'urgency_indicators' in text_features:
            urgency_score = text_features['urgency_indicators'].get('urgency_score', 0)
            if urgency_score > 0.7:  # High urgency
                adjustments *= 1.05
            elif urgency_score == 0:  # No urgency indicators
                adjustments *= 0.95
        
        # Variable extraction adjustment
        if 'variables_extracted' in text_features:
            if text_features['variables_extracted']:  # Has structured data
                adjustments *= 1.05
        
        # Processing notes adjustment
        if 'processing_notes' in text_features:
            notes = text_features['processing_notes']
            if 'very_short_output' in notes:
                adjustments *= 0.8
            if 'no_business_keywords_found' in notes:
                adjustments *= 0.7
        
        return confidences * adjustments
    
    def _apply_calibration(self, raw_confidences: np.ndarray) -> np.ndarray:
        """Apply learned calibration mapping."""
        if not self.is_calibrated or self.calibration_data['calibrated_mapping'] is None:
            return raw_confidences
        
        # Apply calibration mapping based on method
        if self.calibration_method == 'isotonic':
            return self._apply_isotonic_calibration(raw_confidences)
        elif self.calibration_method == 'platt':
            return self._apply_platt_calibration(raw_confidences)
        elif self.calibration_method == 'temperature':
            return self._apply_temperature_scaling(raw_confidences)
        else:
            return raw_confidences
    
    def _apply_isotonic_calibration(self, scores: np.ndarray) -> np.ndarray:
        """Apply isotonic regression calibration."""
        # Simple linear interpolation for now
        # In production, use sklearn.isotonic.IsotonicRegression
        return scores  # Placeholder
    
    def _apply_platt_calibration(self, scores: np.ndarray) -> np.ndarray:
        """Apply Platt scaling calibration."""
        # Sigmoid calibration: P(y=1|f) = 1/(1+exp(A*f+B))
        # Placeholder implementation
        return scores
    
    def _apply_temperature_scaling(self, scores: np.ndarray) -> np.ndarray:
        """Apply temperature scaling calibration."""
        temperature = self.calibration_data.get('temperature', 1.0)
        return 1 / (1 + np.exp(-scores / temperature))
    
    def estimate_uncertainty(self, similarities: List[float], 
                           top_k: int = 3) -> Dict:
        """
        Estimate prediction uncertainty based on similarity distribution.
        
        Args:
            similarities: List of similarity scores for all categories
            top_k: Number of top predictions to consider
            
        Returns:
            Dictionary containing uncertainty metrics
        """
        if len(similarities) < 2:
            return {
                'uncertainty_score': 1.0,
                'confidence_gap': 0.0,
                'entropy': 0.0,
                'top_k_agreement': 0.0
            }
        
        similarities = np.array(similarities)
        
        # Sort similarities in descending order
        sorted_sims = np.sort(similarities)[::-1]
        
        # Confidence gap (difference between top 2 predictions)
        confidence_gap = sorted_sims[0] - sorted_sims[1] if len(sorted_sims) > 1 else 1.0
        
        # Convert to probabilities for entropy calculation
        # Use softmax to convert similarities to probabilities
        exp_sims = np.exp(similarities - np.max(similarities))  # Numerical stability
        probs = exp_sims / np.sum(exp_sims)
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))  # Add small epsilon for numerical stability
        max_entropy = np.log(len(similarities))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Top-k agreement (how concentrated the mass is in top-k)
        top_k_mass = np.sum(sorted_sims[:top_k]) / np.sum(similarities) if np.sum(similarities) > 0 else 0
        
        # Overall uncertainty score (0 = certain, 1 = uncertain)
        uncertainty_score = 0.4 * (1 - confidence_gap) + 0.4 * normalized_entropy + 0.2 * (1 - top_k_mass)
        uncertainty_score = np.clip(uncertainty_score, 0, 1)
        
        return {
            'uncertainty_score': float(uncertainty_score),
            'confidence_gap': float(confidence_gap),
            'entropy': float(normalized_entropy),
            'top_k_agreement': float(top_k_mass),
            'prediction_reliability': 1.0 - uncertainty_score
        }
    
    def calibrate_with_threshold(self, similarities: List[float], 
                                threshold: float = 0.5) -> Dict:
        """
        Calibrate prediction with a decision threshold.
        
        Args:
            similarities: Raw similarity scores
            threshold: Decision threshold for classification
            
        Returns:
            Dictionary with calibrated confidence and decision
        """
        if not similarities:
            return {
                'confidence': self.min_confidence,
                'predicted_category': None,
                'above_threshold': False,
                'all_confidences': []
            }
        
        # Get calibrated confidences
        confidences = self.calibrate_similarity_scores(similarities)
        
        # Find best prediction
        best_idx = np.argmax(confidences)
        best_confidence = confidences[best_idx]
        above_threshold = best_confidence >= threshold
        
        return {
            'confidence': best_confidence,
            'predicted_category_index': best_idx,
            'above_threshold': above_threshold,
            'all_confidences': confidences,
            'decision_threshold': threshold
        }
    
    def get_calibration_metrics(self, true_positives: List[bool], 
                               predicted_confidences: List[float],
                               n_bins: int = 10) -> Dict:
        """
        Calculate calibration metrics for model evaluation.
        
        Args:
            true_positives: List of boolean values indicating correct predictions
            predicted_confidences: List of confidence scores
            n_bins: Number of bins for calibration curve
            
        Returns:
            Dictionary containing calibration metrics
        """
        if len(true_positives) != len(predicted_confidences):
            raise ValueError("true_positives and predicted_confidences must have same length")
        
        if len(true_positives) < n_bins:
            n_bins = max(1, len(true_positives) // 2)
        
        # Calculate calibration curve
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                true_positives, predicted_confidences, n_bins=n_bins
            )
        except Exception as e:
            self.logger.warning(f"Could not calculate calibration curve: {e}")
            return {
                'reliability_score': 0.0,
                'calibration_error': 1.0,
                'brier_score': 1.0
            }
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predicted_confidences > bin_lower) & (predicted_confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.array(true_positives)[in_bin].mean()
                avg_confidence_in_bin = np.array(predicted_confidences)[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Brier Score
        brier_score = np.mean((np.array(predicted_confidences) - np.array(true_positives)) ** 2)
        
        # Reliability Score (inverse of calibration error)
        reliability_score = 1.0 - ece
        
        return {
            'reliability_score': float(reliability_score),
            'expected_calibration_error': float(ece),
            'brier_score': float(brier_score),
            'n_samples': len(true_positives),
            'mean_confidence': float(np.mean(predicted_confidences)),
            'mean_accuracy': float(np.mean(true_positives))
        }
    
    def recommend_threshold(self, validation_scores: List[float], 
                           validation_labels: List[bool]) -> Dict:
        """
        Recommend optimal decision threshold based on validation data.
        
        Args:
            validation_scores: Confidence scores from validation set
            validation_labels: True labels for validation set
            
        Returns:
            Dictionary with recommended threshold and metrics
        """
        if len(validation_scores) != len(validation_labels):
            raise ValueError("validation_scores and validation_labels must have same length")
        
        thresholds = np.linspace(0.1, 0.9, 50)
        best_threshold = 0.5
        best_f1 = 0.0
        
        results = []
        
        for threshold in thresholds:
            predictions = [score >= threshold for score in validation_scores]
            
            # Calculate metrics
            tp = sum(p and l for p, l in zip(predictions, validation_labels))
            fp = sum(p and not l for p, l in zip(predictions, validation_labels))
            fn = sum(not p and l for p, l in zip(predictions, validation_labels))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return {
            'recommended_threshold': best_threshold,
            'best_f1_score': best_f1,
            'threshold_analysis': results
        }
    
    def get_confidence_stats(self) -> Dict:
        """Get statistics about confidence calibration performance."""
        return {
            'is_calibrated': self.is_calibrated,
            'calibration_method': self.calibration_method,
            'predictions_calibrated': self.stats['predictions_calibrated'],
            'min_confidence': self.min_confidence,
            'max_confidence': self.max_confidence,
            'avg_confidence_before': self.stats.get('avg_confidence_before', 0.0),
            'avg_confidence_after': self.stats.get('avg_confidence_after', 0.0)
        }