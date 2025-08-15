"""
Semantic embedding generation for ticket classification.

Optimized for business ticket text with caching and batch processing capabilities.
Uses sentence-transformers for high-quality semantic embeddings.
"""

import numpy as np
import logging
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import time

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

class EmbeddingEngine:
    """High-performance embedding engine with caching and optimization."""
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 cache_dir: Optional[str] = None,
                 normalize_embeddings: bool = True,
                 batch_size: int = 32):
        """
        Initialize embedding engine.
        
        Args:
            model_name: HuggingFace model name for sentence transformers
            cache_dir: Directory for embedding cache (None for no caching)
            normalize_embeddings: Whether to L2 normalize embeddings
            batch_size: Batch size for processing multiple texts
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        
        # Set up caching
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.use_cache = True
        else:
            self.cache_dir = None
            self.use_cache = False
        
        # Initialize model
        self.model = None
        self._load_model()
        
        # Performance tracking
        self.stats = {
            'embeddings_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0.0,
            'avg_embedding_time': 0.0
        }
    
    def _load_model(self):
        """Load the sentence transformer model with deployment optimization."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required. Install with: pip install sentence-transformers")
        
        # Check for bundled deployment model first (deployment optimization)
        bundled_model_path = self._find_bundled_model()
        
        try:
            if bundled_model_path:
                self.logger.info(f"Loading bundled model from: {bundled_model_path}")
                self.model = SentenceTransformer(str(bundled_model_path))
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                self.logger.info(f"✅ Bundled model loaded successfully. Embedding dimension: {self.embedding_dim}")
            else:
                # Fall back to standard loading (works exactly as before)
                self.logger.info(f"Loading sentence transformer model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                self.logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def _find_bundled_model(self) -> Optional[Path]:
        """Find bundled deployment model if available."""
        try:
            # Look for deployment assets in common locations
            possible_locations = [
                Path.cwd() / 'deployment' / 'assets' / 'models' / self.model_name,  # Development
                Path.cwd() / 'assets' / 'models' / self.model_name,  # Deployment  
                Path(__file__).parent.parent.parent.parent / 'deployment' / 'assets' / 'models' / self.model_name,  # Relative to src
            ]
            
            for model_path in possible_locations:
                if model_path.exists() and (model_path / 'config.json').exists():
                    return model_path
            
            return None
        except Exception:
            # If anything goes wrong with bundled model detection, fall back silently
            return None
    
    def embed(self, texts: Union[str, List[str]], 
              use_cache: Optional[bool] = None) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of texts
            use_cache: Whether to use caching (defaults to instance setting)
            
        Returns:
            numpy array of embeddings (single vector or matrix)
        """
        start_time = time.time()
        
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        use_cache = use_cache if use_cache is not None else self.use_cache
        
        embeddings = []
        cache_hits = 0
        cache_misses = 0
        
        # Process texts
        for text in texts:
            # Try cache first
            embedding = None
            if use_cache:
                embedding = self._get_cached_embedding(text)
                if embedding is not None:
                    cache_hits += 1
                else:
                    cache_misses += 1
            
            # Generate embedding if not cached
            if embedding is None:
                embedding = self._generate_embedding(text)
                
                # Cache the result
                if use_cache:
                    self._cache_embedding(text, embedding)
            
            embeddings.append(embedding)
        
        # Convert to numpy array
        embeddings = np.array(embeddings)
        
        # Normalize if requested
        if self.normalize_embeddings:
            embeddings = normalize(embeddings, norm='l2')
        
        # Update stats
        processing_time = time.time() - start_time
        self.stats['embeddings_generated'] += len(texts)
        self.stats['cache_hits'] += cache_hits
        self.stats['cache_misses'] += cache_misses
        self.stats['total_processing_time'] += processing_time
        self.stats['avg_embedding_time'] = self.stats['total_processing_time'] / self.stats['embeddings_generated']
        
        # Return single vector for single input
        if single_input:
            return embeddings[0]
        
        return embeddings
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.embedding_dim)
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            self.logger.error(f"Failed to generate embedding for text: {e}")
            # Return zero vector on error
            return np.zeros(self.embedding_dim)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        # Create hash of text + model name for uniqueness
        content = f"{text}|{self.model_name}|{self.normalize_embeddings}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Retrieve embedding from cache."""
        if not self.use_cache:
            return None
        
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load cached embedding: {e}")
        
        return None
    
    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """Store embedding in cache."""
        if not self.use_cache:
            return
        
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            self.logger.warning(f"Failed to cache embedding: {e}")
    
    def embed_batch(self, texts: List[str], 
                   batch_size: Optional[int] = None) -> np.ndarray:
        """
        Generate embeddings for a large batch of texts efficiently.
        
        Args:
            texts: List of texts to embed
            batch_size: Override default batch size
            
        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        batch_size = batch_size or self.batch_size
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embed(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        
        # Reshape for cosine_similarity function
        emb1 = emb1.reshape(1, -1)
        emb2 = emb2.reshape(1, -1)
        
        return cosine_similarity(emb1, emb2)[0, 0]
    
    def find_most_similar(self, query_text: str, 
                         candidate_texts: List[str],
                         top_k: int = 5) -> List[Tuple[int, str, float]]:
        """
        Find most similar texts to a query.
        
        Args:
            query_text: Text to find similarities for
            candidate_texts: List of candidate texts
            top_k: Number of top results to return
            
        Returns:
            List of (index, text, similarity_score) tuples
        """
        if not candidate_texts:
            return []
        
        # Generate embeddings
        query_embedding = self.embed(query_text).reshape(1, -1)
        candidate_embeddings = self.embed_batch(candidate_texts)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((idx, candidate_texts[idx], similarities[idx]))
        
        return results
    
    def create_category_centroids(self, category_texts: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Create centroid embeddings for each category.
        
        Args:
            category_texts: Dictionary mapping category names to lists of representative texts
            
        Returns:
            Dictionary mapping category names to centroid embeddings
        """
        centroids = {}
        
        for category, texts in category_texts.items():
            if not texts:
                continue
            
            # Generate embeddings for all texts in category
            embeddings = self.embed_batch(texts)
            
            # Calculate centroid (mean embedding)
            centroid = np.mean(embeddings, axis=0)
            
            # Normalize centroid if enabled
            if self.normalize_embeddings:
                centroid = normalize(centroid.reshape(1, -1), norm='l2')[0]
            
            centroids[category] = centroid
        
        return centroids
    
    def classify_by_similarity(self, text: str, 
                              category_centroids: Dict[str, np.ndarray]) -> List[Tuple[str, float]]:
        """
        Classify text by similarity to category centroids.
        
        Args:
            text: Text to classify
            category_centroids: Dictionary mapping category names to centroid embeddings
            
        Returns:
            List of (category, similarity_score) tuples, sorted by similarity
        """
        if not category_centroids:
            return []
        
        # Generate embedding for input text
        text_embedding = self.embed(text).reshape(1, -1)
        
        # Calculate similarities to all centroids
        similarities = []
        for category, centroid in category_centroids.items():
            centroid_reshaped = centroid.reshape(1, -1)
            similarity = cosine_similarity(text_embedding, centroid_reshaped)[0, 0]
            similarities.append((category, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
    
    def get_embedding_stats(self) -> Dict:
        """Get performance statistics for the embedding engine."""
        cache_hit_rate = 0.0
        if self.stats['cache_hits'] + self.stats['cache_misses'] > 0:
            cache_hit_rate = self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
        
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'embeddings_generated': self.stats['embeddings_generated'],
            'cache_enabled': self.use_cache,
            'cache_hit_rate': round(cache_hit_rate, 3),
            'total_processing_time': round(self.stats['total_processing_time'], 2),
            'avg_embedding_time': round(self.stats['avg_embedding_time'], 4),
            'normalize_embeddings': self.normalize_embeddings
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        if not self.use_cache or not self.cache_dir:
            return
        
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            self.logger.info("Embedding cache cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
    
    def warmup_cache(self, texts: List[str]):
        """Pre-generate embeddings for a list of texts to warm up the cache."""
        if not texts:
            return
        
        self.logger.info(f"Warming up cache with {len(texts)} texts...")
        self.embed_batch(texts)
        self.logger.info("Cache warmup completed")