"""
Level 1 Business Classification System

Production-ready classifier for routing tickets to business categories
with 85%+ accuracy target and comprehensive confidence scoring.
"""

import numpy as np
import logging
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from ..utils.text_preprocessor import TextPreprocessor
from ..utils.embedding_engine import EmbeddingEngine
from ..utils.confidence_calibrator import ConfidenceCalibrator
from ..data.category_mappings import BUSINESS_CATEGORIES, BusinessCategory, get_category_by_name
from ..data.routing_logic import get_routing_for_category, calculate_urgency_score
from ..data.original_category_mapping import map_series as map_raw_categories
from ..utils.discriminative_head import DiscriminativeHead

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

@dataclass
class Level1Result:
    """Result from Level 1 business classification."""
    predicted_category: str
    confidence: float
    routing_team: str
    priority_level: str
    sla_hours: int
    urgency_score: float
    processing_time_ms: float
    all_category_scores: Dict[str, float]
    uncertainty_metrics: Dict[str, float]
    text_features: Dict[str, Any]
    routing_info: Dict[str, Any]

class Level1BusinessClassifier:
    """
    Production-ready Level 1 business category classifier.
    
    Achieves 85%+ accuracy for routing tickets to 10 business categories
    with real-time performance (<1 second) and comprehensive confidence scoring.
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 cache_dir: Optional[str] = 'cache/embeddings',
                 confidence_threshold: float = 0.6,
                 enable_preprocessing: bool = True):
        """
        Initialize the Level 1 classifier.
        
        Args:
            model_name: Sentence transformer model name
            cache_dir: Directory for embedding cache
            confidence_threshold: Minimum confidence for positive classification
            enable_preprocessing: Whether to enable advanced text preprocessing
        """
        self.logger = logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold
        self.enable_preprocessing = enable_preprocessing
        
        # Initialize components
        self.text_preprocessor = TextPreprocessor() if enable_preprocessing else None
        self.embedding_engine = EmbeddingEngine(
            model_name=model_name,
            cache_dir=cache_dir,
            normalize_embeddings=True
        )
        self.confidence_calibrator = ConfidenceCalibrator(
            calibration_method='isotonic',
            min_confidence=0.1,
            max_confidence=0.95
        )
        
        # Category setup
        self.business_categories = BUSINESS_CATEGORIES
        self.category_names = [cat.value for cat in BusinessCategory]
        self.category_centroids = None
        self.discriminative_head: Optional[DiscriminativeHead] = None
        self.weight_disc: float = 0.6
        self.weight_cos: float = 0.4
        # Tunable keyword weighting and guard parameters
        self.weight_keyword: float = 0.20
        self.weight_priority: float = 0.10
        self.weight_exclusion: float = 0.15
        self.general_support_penalty: float = 0.02
        self.non_general_margin: float = 0.03
        self.non_general_alt_score: float = 0.15
        self.non_general_top_score_threshold: float = 0.22
        
        # Performance tracking
        self.stats = {
            'classifications_made': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
            'confidence_distribution': [],
            'category_distribution': {},
            'high_confidence_predictions': 0,
            'low_confidence_predictions': 0
        }
        
        # Initialize category representations
        self._initialize_category_representations()

        # Initialize discriminative head if possible
        self._initialize_discriminative_head()
        # Tune blend weight if possible
        self._tune_blend_weight()
        # Tune keyword/guard weights on held-out validation set
        self._tune_keyword_and_guard_weights()
        
        self.logger.info("Level1BusinessClassifier initialized successfully")
    
    def _initialize_category_representations(self):
        """Initialize category representations using keywords and, if available, dataset-derived texts."""
        self.logger.info("Initializing category representations...")
        
        # DEPLOYMENT OPTIMIZATION: Try to load pre-computed embeddings first
        print("🔧 DEBUG: Attempting to load pre-computed embeddings...")
        precomputed_centroids = self._load_precomputed_business_embeddings()
        if precomputed_centroids:
            self.category_centroids = precomputed_centroids
            print(f"✅ DEBUG: Loaded pre-computed business embeddings for {len(precomputed_centroids)} categories")
            self.logger.info(f"✅ Loaded pre-computed business embeddings for {len(precomputed_centroids)} categories")
            return
        
        # FALLBACK: Original computation path (preserves existing functionality)
        print("❌ DEBUG: Pre-computed embeddings not found, using original computation...")
        self.logger.info("Pre-computed embeddings not found, using original computation...")
        
        # 1) Keyword/description-based representatives
        keyword_category_texts: Dict[str, List[str]] = {}
        for _, category_def in self.business_categories.items():
            texts: List[str] = []
            texts.append(category_def.description)
            keywords = category_def.keywords + category_def.priority_keywords
            for i in range(0, len(keywords), 3):
                texts.append(' '.join(keywords[i:i+3]))
            texts.append(' '.join(keywords[:10]))
            keyword_category_texts[category_def.name] = texts

        keyword_centroids = self.embedding_engine.create_category_centroids(keyword_category_texts)

        # 2) Dataset-derived representatives (if data present)
        dataset_centroids: Dict[str, np.ndarray] = {}
        
        # DEPLOYMENT OPTIMIZATION: Try to load pre-computed dataset centroids first
        precomputed_dataset_centroids = self._load_precomputed_dataset_centroids()
        if precomputed_dataset_centroids:
            dataset_centroids = precomputed_dataset_centroids
            print(f"✅ DEBUG: Loaded pre-computed dataset centroids for {len(dataset_centroids)} categories")
            self.logger.info(f"Loaded pre-computed dataset centroids for {len(dataset_centroids)} categories")
        else:
            # FALLBACK: Original dataset centroid computation
            print("🔧 DEBUG: Computing dataset centroids from historical data...")
            data_path = Path('data/processed/consolidated_tickets.csv')
            if pd is not None and data_path.exists():
                try:
                    df = pd.read_csv(data_path, usecols=['Category', 'Short description'])
                    df = df.dropna(subset=['Short description'])
                    # Map raw to business
                    mapped = map_raw_categories(df['Category'].tolist())
                    df = df.assign(BusinessCategory=mapped)
                    # Filter invalid
                    df = df.dropna(subset=['BusinessCategory'])
                    # Limit per category to reduce bias and speed
                    per_category_limit = 500
                    category_to_texts: Dict[str, List[str]] = {}
                    for cat_name in self.category_names:
                        # cat_name is BusinessCategory enum value string
                        # Our definitions use definition.name which matches these strings
                        sample = df[df['BusinessCategory'] == cat_name]['Short description'].astype(str).head(per_category_limit).tolist()
                        if sample:
                            category_to_texts[cat_name] = sample

                    # Create centroids from dataset texts
                    for cat_name, texts in category_to_texts.items():
                        if not texts:
                            continue
                        embeddings = self.embedding_engine.embed_batch(texts[:per_category_limit])
                        if embeddings.size == 0:
                            continue
                        centroid = np.mean(embeddings, axis=0)
                        dataset_centroids[cat_name] = centroid / (np.linalg.norm(centroid) + 1e-12)
                    self.logger.info(f"Dataset-derived centroids created for {len(dataset_centroids)} categories")
                except Exception as e:
                    self.logger.warning(f"Failed to build dataset-derived centroids: {e}")

        # 3) Blend centroids when both available
        blended_centroids: Dict[str, np.ndarray] = {}
        for cat_name in self.category_names:
            kw = keyword_centroids.get(cat_name)
            ds = dataset_centroids.get(cat_name)
            if kw is not None and ds is not None:
                # Weighted blend prioritizing dataset signal
                blended = 0.7 * ds + 0.3 * kw
                blended = blended / (np.linalg.norm(blended) + 1e-12)
                blended_centroids[cat_name] = blended
            elif ds is not None:
                blended_centroids[cat_name] = ds
            elif kw is not None:
                blended_centroids[cat_name] = kw

        self.category_centroids = blended_centroids
        self.logger.info(f"Category representations initialized for {len(self.category_centroids)} categories (blended)")

    def _initialize_discriminative_head(self):
        """Train a light discriminative head on top of embeddings if data is available."""
        try:
            # DEPLOYMENT OPTIMIZATION: Try to load pre-computed discriminative head first
            precomputed_head = self._load_precomputed_discriminative_head()
            if precomputed_head:
                self.discriminative_head = precomputed_head
                self.logger.info("Loaded pre-computed discriminative head")
                return
            
            # FALLBACK: Original discriminative head training (preserves existing functionality)
            head = DiscriminativeHead(
                embedding_fn=lambda texts: self.embedding_engine.embed_batch(texts),
                category_names=self.category_names,
                map_raw_to_business=lambda arr: map_raw_categories(arr),
                max_per_class=3000,
                random_state=42,
            )
            if head.is_available() and head.fit():
                self.discriminative_head = head
                stats = head.get_stats()
                if stats:
                    self.logger.info(f"Discriminative head ready. Val acc: {stats.val_accuracy:.3f}")
            else:
                self.logger.info("Discriminative head not available or not trained; proceeding with centroid-only baseline")
        except Exception as e:
            self.logger.warning(f"Failed to initialize discriminative head: {e}")

    def _tune_blend_weight(self):
        """Data-driven tuning of discriminative vs centroid blend on a held-out sample."""
        if self.discriminative_head is None:
            return
        if pd is None:
            return
        data_path = Path('data/processed/consolidated_tickets.csv')
        if not data_path.exists():
            return
        try:
            df = pd.read_csv(data_path, usecols=['Category', 'Short description'])
            df = df.dropna(subset=['Short description'])
            mapped = map_raw_categories(df['Category'].tolist())
            df = df.assign(BusinessCategory=mapped).dropna(subset=['BusinessCategory'])
            # Sample a manageable validation set
            df = df.sample(n=min(1000, len(df)), random_state=1337)
            texts = df['Short description'].astype(str).tolist()
            labels = df['BusinessCategory'].astype(str).tolist()

            # Precompute cos sims per text for efficiency
            def get_adjusted_sims(text: str) -> List[Tuple[str, float]]:
                sims = self.embedding_engine.classify_by_similarity(text, self.category_centroids)
                return self._compute_adjusted_similarities(text, sims)

            adjusted_cache: List[List[Tuple[str, float]]] = [get_adjusted_sims(t) for t in texts]
            # Precompute discriminative probabilities
            disc_preds = [self.discriminative_head.predict_single(t) for t in texts]
            disc_probs_list: List[Optional[Dict[str, float]]] = []
            for pred in disc_preds:
                if pred is None:
                    disc_probs_list.append(None)
                else:
                    _, _, ranked = pred
                    disc_probs_list.append({c: p for c, p in ranked})

            best_alpha = self.weight_disc
            best_acc = -1.0
            for alpha in [x / 10.0 for x in range(4, 10)]:  # 0.4 .. 0.9
                correct = 0
                for sims, disc_probs, true_label in zip(adjusted_cache, disc_probs_list, labels):
                    # Merge using alpha
                    merged = []
                    for cat, score in sims:
                        if disc_probs and cat in disc_probs:
                            cos_prob = (score + 1.0) / 2.0
                            merged_prob = alpha * disc_probs[cat] + (1 - alpha) * cos_prob
                            merged_score = 2.0 * merged_prob - 1.0
                            merged.append((cat, merged_score))
                        else:
                            merged.append((cat, score))
                    merged.sort(key=lambda x: x[1], reverse=True)
                    pred_cat = merged[0][0]
                    if pred_cat == true_label:
                        correct += 1
                acc = correct / len(texts) if texts else 0.0
                if acc > best_acc:
                    best_acc = acc
                    best_alpha = alpha

            self.weight_disc = float(best_alpha)
            self.weight_cos = 1.0 - self.weight_disc
            self.logger.info(f"Blend weight tuned: weight_disc={self.weight_disc:.2f} (val_acc={best_acc:.3f})")
        except Exception as e:
            self.logger.warning(f"Blend weight tuning failed: {e}")
    
    def classify(self, text: str, 
                include_uncertainty: bool = True,
                include_routing: bool = True) -> Level1Result:
        """
        Classify ticket text to business category.
        
        Args:
            text: Input ticket description
            include_uncertainty: Whether to include uncertainty metrics
            include_routing: Whether to include routing information
            
        Returns:
            Level1Result with classification and routing information
        """
        start_time = time.time()
        
        # Input validation
        if not text or not isinstance(text, str):
            return self._create_fallback_result("Invalid input text", start_time)
        
        # Text preprocessing
        text_features = {}
        processed_text = text
        
        if self.enable_preprocessing and self.text_preprocessor:
            preprocessing_result = self.text_preprocessor.preprocess(text)
            processed_text = preprocessing_result['processed_text']
            text_features = {
                'text_length': len(text),
                'processed_length': len(processed_text),
                'keywords_found': preprocessing_result['keywords_found'],
                'variables_extracted': preprocessing_result['extracted_variables'],
                'processing_notes': preprocessing_result['processing_notes']
            }
            
            # Extract urgency indicators
            urgency_info = self.text_preprocessor.extract_urgency_indicators(text)
            text_features['urgency_indicators'] = urgency_info
        else:
            # Basic features for non-preprocessed text
            text_features = {
                'text_length': len(text),
                'processed_length': len(processed_text),
                'keywords_found': [],
                'variables_extracted': {},
                'processing_notes': [],
                'urgency_indicators': {'urgency_score': 0.0}
            }
        
        # Handle empty processed text
        if not processed_text.strip():
            return self._create_fallback_result("Empty processed text", start_time, text_features)
        
        # Generate embeddings and classify
        try:
            similarities = self.embedding_engine.classify_by_similarity(
                processed_text, 
                self.category_centroids
            )
            
            # Hybrid re-scoring: blend cosine similarity with keyword/priority/exclusion signals
            similarities = self._compute_adjusted_similarities(processed_text, similarities)
            
            # Optionally blend discriminative head probabilities
            disc_probs = None
            if self.discriminative_head is not None:
                pred = self.discriminative_head.predict_single(processed_text)
                if pred is not None:
                    _, _, ranked = pred
                    # Convert ranked list to dict
                    disc_probs = {cat: prob for cat, prob in ranked}

            # Merge centroid scores with discriminative probabilities when available
            merged: List[Tuple[str, float]] = []
            for cat, score in similarities:
                if disc_probs and cat in disc_probs:
                    # Weighted average in probability-like space: map cosine [-1,1]→[0,1]
                    cos_prob = (score + 1.0) / 2.0
                    merged_prob = self.weight_disc * disc_probs[cat] + self.weight_cos * cos_prob
                    merged_score = 2.0 * merged_prob - 1.0
                    merged.append((cat, merged_score))
                else:
                    merged.append((cat, score))
            merged.sort(key=lambda x: x[1], reverse=True)

            # Extract scores and category names after adjustment
            category_scores = {cat: score for cat, score in merged}
            raw_scores = [score for _, score in merged]
            
            # Calibrate confidences
            calibrated_confidences = self.confidence_calibrator.calibrate_similarity_scores(
                raw_scores, text_features
            )
            
            # Create calibrated category scores
            calibrated_scores = {}
            for i, (category, _) in enumerate(similarities):
                calibrated_scores[category] = calibrated_confidences[i]
            
            # Get best prediction with guard against over-predicting General Support
            best_category = merged[0][0]
            best_confidence = calibrated_confidences[0]

            # Prefer specific category over General Support when signals are close
            if best_category == 'General Support':
                # Find best non-general candidate
                best_non_general = None
                for (cat, score), conf in zip(merged, calibrated_confidences):
                    if cat != 'General Support':
                        best_non_general = (cat, score, conf)
                        break
                if best_non_general is not None:
                    top_score = similarities[0][1]
                    non_gen_score = best_non_general[1]
                    # If non-general score is within margin or has reasonable strength, prefer it
                    if (
                        non_gen_score >= top_score - self.non_general_margin
                    ) or (
                        (non_gen_score >= self.non_general_alt_score) and (top_score < self.non_general_top_score_threshold)
                    ):
                        best_category = best_non_general[0]
                        best_confidence = best_non_general[2]
            
            # Calculate uncertainty metrics
            uncertainty_metrics = {}
            if include_uncertainty:
                uncertainty_metrics = self.confidence_calibrator.estimate_uncertainty(raw_scores)
            
            # Get routing information
            routing_info = {}
            if include_routing:
                # Find the business category enum
                category_enum = None
                for cat_enum, cat_def in self.business_categories.items():
                    if cat_def.name == best_category:
                        category_enum = cat_enum
                        break
                
                if category_enum:
                    routing_info = get_routing_for_category(category_enum)
                    
                    # Calculate urgency score
                    urgency_score = calculate_urgency_score(text, category_enum)
                    text_features['urgency_indicators']['final_urgency_score'] = urgency_score
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create result
            result = Level1Result(
                predicted_category=best_category,
                confidence=best_confidence,
                routing_team=routing_info.get('routing_team', 'General Service Desk'),
                priority_level=routing_info.get('priority_level', 'MEDIUM'),
                sla_hours=routing_info.get('sla_hours', 8),
                urgency_score=text_features['urgency_indicators'].get('final_urgency_score', 0.5),
                processing_time_ms=processing_time_ms,
                all_category_scores=calibrated_scores,
                uncertainty_metrics=uncertainty_metrics,
                text_features=text_features,
                routing_info=routing_info
            )
            
            # Update statistics
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Classification failed: {e}")
            return self._create_fallback_result(f"Classification error: {e}", start_time, text_features)

    def _compute_adjusted_similarities(self, processed_text: str, 
                                       similarities: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Blend cosine similarity with keyword-based signals and penalties.

        The adjusted similarity is:
            adj = clip(cos_sim + w_k*kw + w_p*prio - w_e*excl, -1.0, 1.0)
        where kw, prio, excl are in [0,1].
        """
        text_lower = processed_text.lower()
        
        # Weights (tunable)
        weight_keyword = self.weight_keyword
        weight_priority = self.weight_priority
        weight_exclusion = self.weight_exclusion
        
        adjusted: List[Tuple[str, float]] = []
        
        for category_name, cos_sim in similarities:
            # Find definition by name
            cat_def = None
            for _, definition in self.business_categories.items():
                if definition.name == category_name:
                    cat_def = definition
                    break
            
            if cat_def is None:
                adjusted.append((category_name, cos_sim))
                continue
            
            # Keyword matches
            kw_matches = sum(1 for k in cat_def.keywords if k.lower() in text_lower)
            prio_matches = sum(1 for k in cat_def.priority_keywords if k.lower() in text_lower)
            excl_matches = sum(1 for k in cat_def.exclusion_keywords if k.lower() in text_lower)
            
            # Normalize signals to [0,1]
            kw_signal = min(1.0, kw_matches / 5.0)
            prio_signal = min(1.0, prio_matches / 3.0)
            excl_penalty = min(1.0, excl_matches / 2.0)
            
            # Adjust similarity
            adj = cos_sim + weight_keyword * kw_signal + weight_priority * prio_signal - weight_exclusion * excl_penalty
            
            # Slight penalty to General Support to reduce over-selection
            if category_name == 'General Support':
                adj -= self.general_support_penalty
            
            # Clip to cosine-like range
            if adj > 1.0:
                adj = 1.0
            if adj < -1.0:
                adj = -1.0
            
            adjusted.append((category_name, adj))
        
        # Sort by adjusted score desc
        adjusted.sort(key=lambda x: x[1], reverse=True)
        return adjusted

    def _tune_keyword_and_guard_weights(self) -> None:
        """Lightweight sweep over keyword/priority/exclusion weights and guard.

        Uses a held-out random validation subset to select weights that maximize
        accuracy. Keeps search small to maintain startup performance.
        """
        # DEPLOYMENT OPTIMIZATION: Try to load pre-computed parameter tuning first
        precomputed_params = self._load_precomputed_parameter_tuning()
        if precomputed_params:
            self.weight_keyword = precomputed_params.get('weight_keyword', self.weight_keyword)
            self.weight_priority = precomputed_params.get('weight_priority', self.weight_priority)
            self.weight_exclusion = precomputed_params.get('weight_exclusion', self.weight_exclusion)
            self.general_support_penalty = precomputed_params.get('general_support_penalty', self.general_support_penalty)
            self.non_general_margin = precomputed_params.get('non_general_margin', self.non_general_margin)
            
            # Also load blend weights if available
            blend_weights = precomputed_params.get('blend_weights', {})
            if blend_weights:
                self.weight_disc = blend_weights.get('weight_disc', self.weight_disc)
                self.weight_cos = blend_weights.get('weight_cos', self.weight_cos)
            
            self.logger.info("Loaded pre-computed parameter tuning results")
            return
        
        # FALLBACK: Original parameter tuning (preserves existing functionality)
        if pd is None:
            return
        data_path = Path('data/processed/consolidated_tickets.csv')
        if not data_path.exists():
            return
        try:
            df = pd.read_csv(data_path, usecols=['Category', 'Short description'])
            df = df.dropna(subset=['Short description'])
            mapped = map_raw_categories(df['Category'].tolist())
            df = df.assign(BusinessCategory=mapped).dropna(subset=['BusinessCategory'])
            if df.empty:
                return
            # Validation subset
            df_val = df.sample(n=min(1000, len(df)), random_state=2024)
            texts = df_val['Short description'].astype(str).tolist()
            labels = df_val['BusinessCategory'].astype(str).tolist()

            # Small, meaningful grids
            kw_grid = [0.15, 0.20, 0.25]
            prio_grid = [0.05, 0.10, 0.15]
            excl_grid = [0.10, 0.15, 0.20]
            gen_penalty_grid = [0.00, 0.02, 0.04]
            margin_grid = [0.02, 0.03]

            best_cfg = (
                self.weight_keyword,
                self.weight_priority,
                self.weight_exclusion,
                self.general_support_penalty,
                self.non_general_margin,
            )
            best_acc = -1.0

            # Cache adjusted similarities and keyword computations are internal to classify,
            # so we run classify_batch for each setting. Limit search size.
            for wkw in kw_grid:
                for wpr in prio_grid:
                    for wex in excl_grid:
                        for gp in gen_penalty_grid:
                            for mg in margin_grid:
                                # Apply candidate
                                self.weight_keyword = wkw
                                self.weight_priority = wpr
                                self.weight_exclusion = wex
                                self.general_support_penalty = gp
                                self.non_general_margin = mg
                                # Evaluate
                                perf = self.evaluate_performance(texts, labels)
                                acc = perf.get('overall_accuracy', 0.0)
                                if acc > best_acc:
                                    best_acc = acc
                                    best_cfg = (wkw, wpr, wex, gp, mg)

            # Restore best
            (
                self.weight_keyword,
                self.weight_priority,
                self.weight_exclusion,
                self.general_support_penalty,
                self.non_general_margin,
            ) = best_cfg
            self.logger.info(
                f"Keyword/guard tuning: kw={self.weight_keyword:.2f}, prio={self.weight_priority:.2f}, "
                f"excl={self.weight_exclusion:.2f}, gen_penalty={self.general_support_penalty:.2f}, "
                f"non_gen_margin={self.non_general_margin:.2f} (val_acc={best_acc:.3f})"
            )
        except Exception as e:
            self.logger.warning(f"Keyword/guard tuning failed: {e}")
    
    def _create_fallback_result(self, reason: str, start_time: float, 
                               text_features: Optional[Dict] = None) -> Level1Result:
        """Create a fallback result for error cases."""
        processing_time_ms = (time.time() - start_time) * 1000
        
        if text_features is None:
            text_features = {
                'text_length': 0,
                'processed_length': 0,
                'keywords_found': [],
                'variables_extracted': {},
                'processing_notes': [reason],
                'urgency_indicators': {'urgency_score': 0.0}
            }
        
        return Level1Result(
            predicted_category="General Support",
            confidence=self.confidence_calibrator.min_confidence,
            routing_team="General Service Desk",
            priority_level="LOW",
            sla_hours=24,
            urgency_score=0.1,
            processing_time_ms=processing_time_ms,
            all_category_scores={"General Support": self.confidence_calibrator.min_confidence},
            uncertainty_metrics={'uncertainty_score': 1.0},
            text_features=text_features,
            routing_info={}
        )
    
    def classify_batch(self, texts: List[str], 
                      batch_size: int = 32,
                      show_progress: bool = False) -> List[Level1Result]:
        """
        Classify multiple texts efficiently in batches.
        
        Args:
            texts: List of ticket descriptions
            batch_size: Batch size for processing
            show_progress: Whether to show progress information
            
        Returns:
            List of Level1Result objects
        """
        if not texts:
            return []
        
        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            if show_progress:
                batch_num = i // batch_size + 1
                self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
            
            # Process each text in the batch
            batch_results = []
            for text in batch_texts:
                result = self.classify(text, include_uncertainty=False, include_routing=True)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def evaluate_performance(self, test_texts: List[str], 
                           true_categories: List[str]) -> Dict[str, Any]:
        """
        Evaluate classifier performance on test data.
        
        Args:
            test_texts: List of test ticket descriptions
            true_categories: List of true category labels
            
        Returns:
            Dictionary containing performance metrics
        """
        if len(test_texts) != len(true_categories):
            raise ValueError("test_texts and true_categories must have same length")
        
        self.logger.info(f"Evaluating performance on {len(test_texts)} test samples")
        
        # Get predictions
        predictions = self.classify_batch(test_texts, show_progress=True)
        
        # Calculate metrics
        correct_predictions = 0
        confidence_scores = []
        category_performance = {}
        
        for i, (pred, true_cat) in enumerate(zip(predictions, true_categories)):
            is_correct = pred.predicted_category == true_cat
            if is_correct:
                correct_predictions += 1
            
            confidence_scores.append(pred.confidence)
            
            # Per-category metrics
            if true_cat not in category_performance:
                category_performance[true_cat] = {'correct': 0, 'total': 0, 'avg_confidence': 0.0}
            
            category_performance[true_cat]['total'] += 1
            if is_correct:
                category_performance[true_cat]['correct'] += 1
            category_performance[true_cat]['avg_confidence'] += pred.confidence
        
        # Calculate final metrics
        accuracy = correct_predictions / len(test_texts)
        avg_confidence = np.mean(confidence_scores)
        
        # Per-category accuracies
        for cat, perf in category_performance.items():
            perf['accuracy'] = perf['correct'] / perf['total']
            perf['avg_confidence'] /= perf['total']
        
        # Confidence calibration metrics
        true_positives = [pred.predicted_category == true_cat 
                         for pred, true_cat in zip(predictions, true_categories)]
        
        calibration_metrics = self.confidence_calibrator.get_calibration_metrics(
            true_positives, confidence_scores
        )
        
        performance_metrics = {
            'overall_accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'total_samples': len(test_texts),
            'correct_predictions': correct_predictions,
            'category_performance': category_performance,
            'calibration_metrics': calibration_metrics,
            'meets_target': accuracy >= 0.85,  # 85% target
            'avg_processing_time_ms': np.mean([p.processing_time_ms for p in predictions])
        }
        
        self.logger.info(f"Performance evaluation completed: {accuracy:.3f} accuracy")
        return performance_metrics
    
    def _update_stats(self, result: Level1Result):
        """Update internal statistics."""
        self.stats['classifications_made'] += 1
        self.stats['total_processing_time'] += result.processing_time_ms
        self.stats['avg_processing_time'] = (
            self.stats['total_processing_time'] / self.stats['classifications_made']
        )
        
        # Confidence distribution
        self.stats['confidence_distribution'].append(result.confidence)
        
        # Category distribution
        category = result.predicted_category
        if category not in self.stats['category_distribution']:
            self.stats['category_distribution'][category] = 0
        self.stats['category_distribution'][category] += 1
        
        # Confidence buckets
        if result.confidence >= 0.8:
            self.stats['high_confidence_predictions'] += 1
        elif result.confidence < 0.5:
            self.stats['low_confidence_predictions'] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.stats.copy()
        
        # Add derived metrics
        if self.stats['classifications_made'] > 0:
            stats['high_confidence_rate'] = (
                self.stats['high_confidence_predictions'] / self.stats['classifications_made']
            )
            stats['low_confidence_rate'] = (
                self.stats['low_confidence_predictions'] / self.stats['classifications_made']
            )
            
            if self.stats['confidence_distribution']:
                stats['median_confidence'] = np.median(self.stats['confidence_distribution'])
                stats['confidence_std'] = np.std(self.stats['confidence_distribution'])
        
        # Add component stats
        stats['embedding_stats'] = self.embedding_engine.get_embedding_stats()
        stats['confidence_stats'] = self.confidence_calibrator.get_confidence_stats()
        
        # Add category info
        stats['num_categories'] = len(self.category_centroids)
        stats['category_names'] = list(self.category_centroids.keys())
        
        return stats
    
    def explain_prediction(self, text: str, result: Level1Result) -> Dict[str, Any]:
        """
        Provide explanation for a classification result.
        
        Args:
            text: Original input text
            result: Classification result to explain
            
        Returns:
            Dictionary containing explanation information
        """
        explanation = {
            'input_analysis': {
                'original_text': text,
                'text_length': len(text),
                'processed_features': result.text_features
            },
            'classification_reasoning': {
                'predicted_category': result.predicted_category,
                'confidence': result.confidence,
                'confidence_level': 'HIGH' if result.confidence >= 0.8 else 'MEDIUM' if result.confidence >= 0.6 else 'LOW',
                'uncertainty_score': result.uncertainty_metrics.get('uncertainty_score', 0.0)
            },
            'category_comparison': {
                'top_3_categories': sorted(
                    result.all_category_scores.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3],
                'confidence_gap': max(result.all_category_scores.values()) - sorted(result.all_category_scores.values(), reverse=True)[1] if len(result.all_category_scores) > 1 else 0.0
            },
            'routing_decision': {
                'routing_team': result.routing_team,
                'priority_level': result.priority_level,
                'sla_hours': result.sla_hours,
                'urgency_score': result.urgency_score
            },
            'quality_indicators': {
                'processing_time_ms': result.processing_time_ms,
                'meets_confidence_threshold': result.confidence >= self.confidence_threshold,
                'has_business_keywords': len(result.text_features.get('keywords_found', [])) > 0,
                'text_quality': 'GOOD' if len(result.text_features.get('keywords_found', [])) > 0 and result.text_features.get('text_length', 0) > 10 else 'POOR'
            }
        }
        
        return explanation
    
    def _load_precomputed_discriminative_head(self) -> Optional[DiscriminativeHead]:
        """
        Load pre-computed discriminative head to avoid live training.
        
        Returns:
            Trained DiscriminativeHead object, or None if not available
        """
        try:
            # Look for pre-computed discriminative head in deployment asset locations
            possible_locations = [
                Path.cwd() / 'deployment' / 'assets' / 'embeddings',  # Development
                Path.cwd() / 'assets' / 'embeddings',  # Deployment
                Path(__file__).parent.parent.parent.parent / 'deployment' / 'assets' / 'embeddings',  # Relative to src
            ]
            
            for embeddings_dir in possible_locations:
                head_file = embeddings_dir / 'discriminative_head.pkl'
                
                if head_file.exists():
                    import pickle
                    with open(head_file, 'rb') as f:
                        head = pickle.load(f)
                    
                    # Verify it's a valid discriminative head
                    if hasattr(head, 'predict_single') and hasattr(head, 'is_trained'):
                        self.logger.info(f"Loaded pre-computed discriminative head from {head_file}")
                        return head
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to load pre-computed discriminative head: {e}")
            return None

    def _load_precomputed_parameter_tuning(self) -> Optional[Dict[str, Any]]:
        """
        Load pre-computed parameter tuning results to avoid live tuning.
        
        Returns:
            Dictionary with tuned parameters, or None if not available
        """
        try:
            # Look for pre-computed parameter tuning in deployment asset locations
            possible_locations = [
                Path.cwd() / 'deployment' / 'assets' / 'embeddings',  # Development
                Path.cwd() / 'assets' / 'embeddings',  # Deployment
                Path(__file__).parent.parent.parent.parent / 'deployment' / 'assets' / 'embeddings',  # Relative to src
            ]
            
            for embeddings_dir in possible_locations:
                tuning_file = embeddings_dir / 'parameter_tuning.json'
                
                if tuning_file.exists():
                    with open(tuning_file, 'r') as f:
                        tuning_results = json.load(f)
                    
                    self.logger.info(f"Loaded pre-computed parameter tuning from {tuning_file}")
                    return tuning_results
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to load pre-computed parameter tuning: {e}")
            return None
    
    def _load_precomputed_dataset_centroids(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Load pre-computed dataset centroids to avoid live computation.
        
        Returns:
            Dictionary mapping category names to centroids, or None if not available
        """
        try:
            # Look for pre-computed dataset centroids in deployment asset locations
            possible_locations = [
                Path.cwd() / 'deployment' / 'assets' / 'embeddings',  # Development
                Path.cwd() / 'assets' / 'embeddings',  # Deployment
                Path(__file__).parent.parent.parent.parent / 'deployment' / 'assets' / 'embeddings',  # Relative to src
            ]
            
            for embeddings_dir in possible_locations:
                centroids_file = embeddings_dir / 'dataset_centroids.npy'
                metadata_file = embeddings_dir / 'dataset_metadata.json'
                
                if centroids_file.exists() and metadata_file.exists():
                    # Load centroids and metadata
                    centroids_array = np.load(centroids_file)
                    
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Verify metadata integrity
                    centroid_names = metadata.get('centroid_names', [])
                    
                    if len(centroid_names) != centroids_array.shape[0]:
                        self.logger.warning(f"Dataset centroid count mismatch in {centroids_file}")
                        continue
                    
                    # Create dataset centroids dictionary
                    dataset_centroids = {}
                    for i, category_name in enumerate(centroid_names):
                        dataset_centroids[category_name] = centroids_array[i]
                    
                    self.logger.info(f"Loaded pre-computed dataset centroids from {centroids_file}")
                    return dataset_centroids
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to load pre-computed dataset centroids: {e}")
            return None

    def _load_precomputed_business_embeddings(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Load pre-computed business category embeddings for deployment optimization.
        
        Returns:
            Dictionary mapping category names to embeddings, or None if not available
        """
        try:
            print(f"🔧 DEBUG: Current working directory: {Path.cwd()}")
            
            # Look for pre-computed embeddings in deployment asset locations
            possible_locations = [
                Path.cwd() / 'deployment' / 'assets' / 'embeddings',  # Development
                Path.cwd() / 'assets' / 'embeddings',  # Deployment
                Path(__file__).parent.parent.parent.parent / 'deployment' / 'assets' / 'embeddings',  # Relative to src
            ]
            
            for i, embeddings_dir in enumerate(possible_locations):
                embeddings_file = embeddings_dir / 'business_categories.npy'
                metadata_file = embeddings_dir / 'business_metadata.json'
                
                print(f"🔧 DEBUG: Checking location {i+1}: {embeddings_dir}")
                print(f"🔧 DEBUG: NPY exists: {embeddings_file.exists()}, JSON exists: {metadata_file.exists()}")
                
                if embeddings_file.exists() and metadata_file.exists():
                    print(f"✅ DEBUG: Found assets at {embeddings_dir}")
                    
                    # Load embeddings and metadata
                    embeddings_array = np.load(embeddings_file)
                    print(f"🔧 DEBUG: Loaded embeddings shape: {embeddings_array.shape}")
                    
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Verify metadata integrity
                    business_names = metadata.get('business_names', [])
                    print(f"🔧 DEBUG: Business names count: {len(business_names)}")
                    
                    if len(business_names) != embeddings_array.shape[0]:
                        print(f"❌ DEBUG: Embedding count mismatch: {len(business_names)} names vs {embeddings_array.shape[0]} embeddings")
                        self.logger.warning(f"Embedding count mismatch in {embeddings_file}")
                        continue
                    
                    # Create category centroids dictionary
                    category_centroids = {}
                    for i, category_name in enumerate(business_names):
                        category_centroids[category_name] = embeddings_array[i]
                    
                    print(f"✅ DEBUG: Successfully created centroids for {len(category_centroids)} categories")
                    self.logger.info(f"Loaded pre-computed embeddings from {embeddings_file}")
                    return category_centroids
            
            print("❌ DEBUG: No valid embeddings found in any location")
            return None
            
        except Exception as e:
            print(f"❌ DEBUG: Exception in loading: {e}")
            import traceback
            traceback.print_exc()
            self.logger.warning(f"Failed to load pre-computed embeddings: {e}")
            return None