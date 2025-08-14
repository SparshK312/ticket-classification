"""
Level 2 Semantic Search System

Provides specific problem identification within business categories using
semantic similarity search against historical ticket database.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from ..utils.embedding_engine import EmbeddingEngine
from ..data.original_category_mapping import map_raw_category

@dataclass
class ProblemMatch:
    """A matched problem from the database."""
    problem_id: str
    problem_description: str
    original_ticket: str
    business_category: str
    similarity_score: float
    confidence: float
    resolution_hint: Optional[str] = None
    frequency: int = 1
    avg_resolution_time: Optional[float] = None

@dataclass
class Level2Result:
    """Result from Level 2 semantic search."""
    specific_problem: str
    confidence: float
    similar_problems: List[ProblemMatch]
    processing_time_ms: float
    search_method: str
    total_candidates: int
    relevance_score: float

class Level2SemanticSearch:
    """
    Level 2 semantic search for specific problem identification.
    
    Takes a business category from Level 1 and finds the most similar
    specific problems from historical ticket data within that category.
    """
    
    def __init__(self, 
                 embedding_engine: Optional[EmbeddingEngine] = None,
                 problem_database_path: str = "cache/level2_problems.json",
                 min_similarity_threshold: float = 0.3,
                 max_candidates: int = 100):
        """
        Initialize Level 2 semantic search.
        
        Args:
            embedding_engine: Shared embedding engine from Level 1
            problem_database_path: Path to cached problem database
            min_similarity_threshold: Minimum similarity for candidate problems
            max_candidates: Maximum number of candidates to consider
        """
        self.logger = logging.getLogger(__name__)
        self.min_similarity_threshold = min_similarity_threshold
        self.max_candidates = max_candidates
        self.problem_database_path = Path(problem_database_path)
        
        # Use shared embedding engine or create new one
        self.embedding_engine = embedding_engine or EmbeddingEngine(
            model_name='all-MiniLM-L6-v2',
            cache_dir='cache/embeddings',
            normalize_embeddings=True
        )
        
        # Problem database storage
        self.problem_database = {}  # category -> list of problems
        self.problem_embeddings = {}  # category -> numpy array of embeddings
        self.problem_metadata = {}  # category -> list of metadata
        
        # Performance tracking
        self.stats = {
            'searches_performed': 0,
            'avg_search_time_ms': 0.0,
            'cache_hits': 0,
            'database_size': 0
        }
        
        # Initialize problem database
        self._initialize_problem_database()
        
        self.logger.info("Level2SemanticSearch initialized successfully")
    
    def _initialize_problem_database(self):
        """Initialize or load the problem database from historical tickets."""
        self.logger.info("Initializing Level 2 problem database...")
        
        # Try to load cached database first
        if self._load_cached_database():
            self.logger.info(f"Loaded cached problem database with {self.stats['database_size']} problems")
            return
        
        # Build database from scratch
        if not PANDAS_AVAILABLE:
            self.logger.warning("pandas not available, Level 2 will use minimal database")
            self._create_minimal_database()
            return
        
        self._build_database_from_tickets()
    
    def _load_cached_database(self) -> bool:
        """Load cached problem database if it exists."""
        if not self.problem_database_path.exists():
            return False
        
        try:
            with open(self.problem_database_path, 'r') as f:
                cached_data = json.load(f)
            
            self.problem_database = cached_data.get('problems', {})
            
            # Load embeddings (stored as lists, convert to numpy arrays)
            embeddings_data = cached_data.get('embeddings', {})
            for category, embedding_list in embeddings_data.items():
                self.problem_embeddings[category] = np.array(embedding_list)
            
            self.problem_metadata = cached_data.get('metadata', {})
            
            # Update stats
            total_problems = sum(len(problems) for problems in self.problem_database.values())
            self.stats['database_size'] = total_problems
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to load cached database: {e}")
            return False
    
    def _save_database_cache(self):
        """Save problem database to cache."""
        try:
            # Create cache directory
            self.problem_database_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy arrays to lists for JSON serialization
            embeddings_data = {}
            for category, embeddings in self.problem_embeddings.items():
                embeddings_data[category] = embeddings.tolist()
            
            cache_data = {
                'problems': self.problem_database,
                'embeddings': embeddings_data,
                'metadata': self.problem_metadata,
                'stats': self.stats
            }
            
            with open(self.problem_database_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            self.logger.info(f"Saved problem database cache to {self.problem_database_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save database cache: {e}")
    
    def _create_minimal_database(self):
        """Create a minimal problem database for testing."""
        self.logger.info("Creating minimal problem database...")
        
        # Sample problems for each business category
        sample_problems = {
            "Till Operations": [
                "Till crashed during transaction",
                "Till not responding to scanner input", 
                "Till drawer stuck closed",
                "Till display showing error codes",
                "Cash register not printing receipts"
            ],
            "Vision Orders & Inventory": [
                "Vision order stuck in locked status",
                "Cannot amend order quantities in Vision",
                "Vision system timeout during order processing",
                "Inventory levels not updating in Vision",
                "Order approval workflow not working"
            ],
            "Payment Processing": [
                "Chip and pin device offline",
                "Card payment being declined incorrectly",
                "Payment terminal not connecting to network",
                "Contactless payments not working",
                "Card reader not recognizing cards"
            ],
            "Printing Services": [
                "Label printer not printing barcodes",
                "End of day reports not generating",
                "Printer queue stuck with pending jobs",
                "Print quality poor on receipt printer",
                "Printer offline despite network connection"
            ],
            "User Account Management": [
                "Active Directory account locked",
                "User cannot login after password reset",
                "New employee account not created",
                "User permissions not working correctly",
                "Multi-factor authentication not working"
            ],
            "Email & Communications": [
                "Google email not syncing",
                "Cannot access shared mailbox",
                "Email attachment size limit exceeded",
                "Outlook not connecting to server",
                "Email rules not filtering correctly"
            ],
            "Software & Application Issues": [
                "AppStream application not launching",
                "Fusion system running slowly",
                "Application crashes when opening files",
                "Software license expired",
                "Application features not working"
            ],
            "Mobile Devices": [
                "Zebra device battery not charging",
                "Mobile scanner not reading barcodes",
                "Device not connecting to WiFi",
                "Touch screen not responding",
                "Mobile app not syncing data"
            ],
            "Back Office & Financial": [
                "Financial reports showing incorrect data",
                "Back office system reconciliation errors",
                "Invoice processing delays",
                "Payment reconciliation not matching",
                "Financial data export failing"
            ],
            "General Support": [
                "Network connectivity issues",
                "Hardware replacement needed",
                "System performance degradation",
                "User training request",
                "General technical assistance"
            ]
        }
        
        # Generate embeddings for sample problems
        for category, problems in sample_problems.items():
            if not problems:
                continue
                
            self.problem_database[category] = []
            metadata_list = []
            
            # Generate embeddings
            embeddings = self.embedding_engine.embed(problems)
            self.problem_embeddings[category] = embeddings
            
            # Create problem entries
            for i, problem in enumerate(problems):
                problem_entry = {
                    'id': f"{category.lower().replace(' ', '_')}_{i+1:03d}",
                    'description': problem,
                    'original_ticket': problem,  # Same as description for minimal DB
                    'category': category,
                    'frequency': 1,
                    'avg_resolution_time': None
                }
                
                self.problem_database[category].append(problem_entry)
                metadata_list.append(problem_entry)
            
            self.problem_metadata[category] = metadata_list
        
        # Update stats
        total_problems = sum(len(problems) for problems in self.problem_database.values())
        self.stats['database_size'] = total_problems
        
        # Save to cache
        self._save_database_cache()
        
        self.logger.info(f"Created minimal database with {total_problems} sample problems")
    
    def _build_database_from_tickets(self):
        """Build problem database from actual historical tickets."""
        self.logger.info("Building problem database from historical tickets...")
        
        try:
            # Load historical tickets
            data_path = Path("data/processed/consolidated_tickets.csv")
            if not data_path.exists():
                self.logger.warning("Historical ticket data not found, using minimal database")
                self._create_minimal_database()
                return
            
            df = pd.read_csv(data_path)
            self.logger.info(f"Loaded {len(df)} historical tickets")
            
            # Map raw categories to business categories
            business_categories = [map_raw_category(cat) for cat in df['Category']]
            df['BusinessCategory'] = business_categories
            
            # Filter out unmapped categories
            df = df.dropna(subset=['BusinessCategory'])
            df = df.dropna(subset=['Short description'])
            
            # Group by business category
            category_problems = {}
            category_embeddings = {}
            category_metadata = {}
            
            for category in df['BusinessCategory'].unique():
                category_df = df[df['BusinessCategory'] == category]
                
                # Extract problem descriptions (use Short description)
                descriptions = category_df['Short description'].astype(str).tolist()
                
                # Limit problems per category for performance
                if len(descriptions) > 200:
                    descriptions = descriptions[:200]
                    category_df = category_df.head(200)
                
                if not descriptions:
                    continue
                
                # Generate embeddings
                self.logger.info(f"Processing {len(descriptions)} problems for {category}")
                embeddings = self.embedding_engine.embed(descriptions)
                
                # Create problem entries
                problems = []
                metadata = []
                
                for i, (_, row) in enumerate(category_df.iterrows()):
                    if i >= len(descriptions):
                        break
                        
                    problem_entry = {
                        'id': f"{category.lower().replace(' & ', '_').replace(' ', '_')}_{i+1:04d}",
                        'description': descriptions[i],
                        'original_ticket': descriptions[i],
                        'category': category,
                        'frequency': 1,  # Could calculate from duplicates
                        'avg_resolution_time': None  # Could extract from ticket data
                    }
                    
                    problems.append(problem_entry)
                    metadata.append(problem_entry)
                
                category_problems[category] = problems
                category_embeddings[category] = embeddings
                category_metadata[category] = metadata
                
                self.logger.info(f"Created {len(problems)} problems for {category}")
            
            # Store in instance variables
            self.problem_database = category_problems
            self.problem_embeddings = category_embeddings
            self.problem_metadata = category_metadata
            
            # Update stats
            total_problems = sum(len(problems) for problems in category_problems.values())
            self.stats['database_size'] = total_problems
            
            # Save cache
            self._save_database_cache()
            
            self.logger.info(f"Built problem database with {total_problems} problems across {len(category_problems)} categories")
            
        except Exception as e:
            self.logger.error(f"Failed to build database from tickets: {e}")
            self._create_minimal_database()
    
    def search(self, 
               text: str, 
               business_category: str, 
               top_k: int = 5) -> Level2Result:
        """
        Search for similar problems within a business category.
        
        Args:
            text: Input ticket description
            business_category: Business category from Level 1
            top_k: Number of similar problems to return
            
        Returns:
            Level2Result with similar problems and metadata
        """
        start_time = time.time()
        
        # Check if category exists in database
        if business_category not in self.problem_database:
            return Level2Result(
                specific_problem="Unknown problem type",
                confidence=0.1,
                similar_problems=[],
                processing_time_ms=0.0,
                search_method="no_database",
                total_candidates=0,
                relevance_score=0.0
            )
        
        # Get category problems and embeddings
        category_problems = self.problem_database[business_category]
        category_embeddings = self.problem_embeddings[business_category]
        
        if len(category_problems) == 0:
            return Level2Result(
                specific_problem="No similar problems found",
                confidence=0.1,
                similar_problems=[],
                processing_time_ms=(time.time() - start_time) * 1000,
                search_method="empty_database",
                total_candidates=0,
                relevance_score=0.0
            )
        
        # Generate query embedding
        query_embedding = self.embedding_engine.embed([text])[0]
        
        # Calculate similarities
        similarities = np.dot(category_embeddings, query_embedding)
        
        # Filter by minimum similarity threshold
        valid_indices = np.where(similarities >= self.min_similarity_threshold)[0]
        
        if len(valid_indices) == 0:
            # No problems meet threshold, return best matches anyway
            valid_indices = np.argsort(similarities)[-min(top_k, len(similarities)):]
        
        # Sort by similarity (descending)
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
        
        # Limit to top_k
        top_indices = sorted_indices[:top_k]
        
        # Create problem matches
        similar_problems = []
        for idx in top_indices:
            problem = category_problems[idx]
            similarity = float(similarities[idx])
            
            # Calculate confidence based on similarity and position
            confidence = min(0.95, similarity * 0.8 + 0.1)
            
            match = ProblemMatch(
                problem_id=problem['id'],
                problem_description=problem['description'],
                original_ticket=problem['original_ticket'],
                business_category=business_category,
                similarity_score=similarity,
                confidence=confidence,
                frequency=problem.get('frequency', 1),
                avg_resolution_time=problem.get('avg_resolution_time')
            )
            similar_problems.append(match)
        
        # Determine specific problem (highest similarity)
        if similar_problems:
            specific_problem = similar_problems[0].problem_description
            overall_confidence = similar_problems[0].confidence
            relevance_score = float(similarities[top_indices[0]])
        else:
            specific_problem = "No similar problems identified"
            overall_confidence = 0.1
            relevance_score = 0.0
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update stats
        self.stats['searches_performed'] += 1
        avg_time = self.stats['avg_search_time_ms'] * (self.stats['searches_performed'] - 1)
        self.stats['avg_search_time_ms'] = (avg_time + processing_time) / self.stats['searches_performed']
        
        return Level2Result(
            specific_problem=specific_problem,
            confidence=overall_confidence,
            similar_problems=similar_problems,
            processing_time_ms=processing_time,
            search_method="semantic_similarity",
            total_candidates=len(category_problems),
            relevance_score=relevance_score
        )
    
    def get_category_problems(self, business_category: str) -> List[Dict]:
        """Get all problems for a specific business category."""
        return self.problem_database.get(business_category, [])
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        category_stats = {}
        for category, problems in self.problem_database.items():
            category_stats[category] = len(problems)
        
        return {
            'total_categories': len(self.problem_database),
            'total_problems': self.stats['database_size'],
            'category_breakdown': category_stats,
            'searches_performed': self.stats['searches_performed'],
            'avg_search_time_ms': self.stats['avg_search_time_ms'],
            'cache_hits': self.stats['cache_hits']
        }
    
    def rebuild_database(self, force: bool = False):
        """Rebuild the problem database from scratch."""
        if force or not self.problem_database_path.exists():
            self.logger.info("Rebuilding problem database...")
            self.problem_database = {}
            self.problem_embeddings = {}
            self.problem_metadata = {}
            self._build_database_from_tickets()
        else:
            self.logger.info("Database already exists, use force=True to rebuild")