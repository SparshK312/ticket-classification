"""
Configuration settings for IT Ticket Automation Analysis

This module contains all configuration parameters for the ticket analysis system.
Modify these settings to customize the behavior of the analysis pipeline.
"""

import os
from pathlib import Path
from typing import Dict, List


class Config:
    """Configuration class for the ticket analysis system."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    CONFIG_DIR = PROJECT_ROOT / "config"
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
    
    # Default data file
    DEFAULT_DATA_FILE = RAW_DATA_DIR / "tickets.csv"
    
    # Data processing settings
    DATA_PROCESSING = {
        "text_columns": ["description", "comments", "summary"],
        "categorical_columns": ["category", "priority", "status"],
        "datetime_columns": ["created_date", "resolved_date"],
        "min_text_length": 10,
        "max_text_length": 10000,
        "remove_duplicates": True,
        "handle_missing": "drop"  # Options: 'drop', 'fill', 'keep'
    }
    
    # Text preprocessing settings
    TEXT_PREPROCESSING = {
        "lowercase": True,
        "remove_punctuation": False,
        "remove_numbers": False,
        "remove_stopwords": True,
        "stemming": False,
        "lemmatization": True,
        "min_word_length": 2,
        "custom_stopwords": [
            "ticket", "issue", "problem", "help", "please", "thanks"
        ]
    }
    
    # Clustering settings
    CLUSTERING = {
        "embedding_model": "all-MiniLM-L6-v2",  # Sentence transformer model
        "algorithm": "kmeans",  # Options: 'kmeans', 'dbscan', 'hierarchical'
        "max_clusters": 20,
        "min_cluster_size": 5,
        "random_state": 42,
        "silhouette_threshold": 0.3,
        
        # KMeans specific
        "kmeans_init": "k-means++",
        "kmeans_max_iter": 300,
        
        # DBSCAN specific
        "dbscan_eps": 0.5,
        "dbscan_min_samples": 5,
        
        # Hierarchical specific
        "linkage": "ward",
        "distance_threshold": None
    }
    
    # Keyword extraction settings
    KEYWORD_EXTRACTION = {
        "method": "tfidf",  # Options: 'tfidf', 'yake', 'keybert'
        "max_keywords": 15,
        "ngram_range": (1, 3),
        "max_features": 5000,
        "min_df": 2,
        "max_df": 0.8,
        "remove_stopwords": True,
        "technical_terms_boost": 1.5  # Boost technical terms
    }
    
    # Automation analysis settings
    AUTOMATION_ANALYSIS = {
        "high_automation_keywords": [
            "password reset", "account unlock", "software install",
            "permission", "access", "restart", "reboot", "backup",
            "routine", "standard", "common", "frequent"
        ],
        "medium_automation_keywords": [
            "configuration", "setup", "troubleshoot", "diagnose",
            "update", "patch", "maintenance", "monitor"
        ],
        "low_automation_keywords": [
            "custom", "unique", "complex", "investigation",
            "analysis", "design", "development", "consultation"
        ],
        "automation_threshold": {
            "high": 0.7,
            "medium": 0.4,
            "low": 0.0
        }
    }
    
    # Model settings
    MODELS = {
        "sentence_transformer": {
            "model_name": "all-MiniLM-L6-v2",
            "device": "cpu",  # Options: 'cpu', 'cuda'
            "batch_size": 32
        },
        "summarization": {
            "model_name": "facebook/bart-large-cnn",
            "max_length": 150,
            "min_length": 30,
            "length_penalty": 2.0,
            "num_beams": 4
        }
    }
    
    # Output settings
    OUTPUT_SETTINGS = {
        "save_intermediate": True,
        "export_formats": ["json", "csv"],
        "include_visualizations": True,
        "generate_report": True,
        "report_template": "default"
    }
    
    # Logging settings
    LOGGING = {
        "level": "INFO",  # Options: DEBUG, INFO, WARNING, ERROR
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": OUTPUT_DIR / "ticket_analysis.log",
        "max_file_size": "10MB",
        "backup_count": 5
    }
    
    # Performance settings
    PERFORMANCE = {
        "use_multiprocessing": True,
        "n_jobs": -1,  # -1 uses all available cores
        "chunk_size": 1000,
        "memory_limit": "4GB",
        "cache_embeddings": True,
        "cache_dir": PROJECT_ROOT / ".cache"
    }
    
    # Visualization settings
    VISUALIZATION = {
        "figure_size": (12, 8),
        "dpi": 300,
        "style": "seaborn-v0_8",
        "color_palette": "Set2",
        "save_plots": True,
        "show_plots": False
    }
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.OUTPUT_DIR,
            cls.CONFIG_DIR,
            cls.NOTEBOOKS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if they exist
        if os.getenv('EMBEDDING_MODEL'):
            config.CLUSTERING['embedding_model'] = os.getenv('EMBEDDING_MODEL')
        
        if os.getenv('MAX_CLUSTERS'):
            config.CLUSTERING['max_clusters'] = int(os.getenv('MAX_CLUSTERS'))
        
        if os.getenv('LOG_LEVEL'):
            config.LOGGING['level'] = os.getenv('LOG_LEVEL')
        
        return config
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not attr.startswith('_') and not callable(getattr(self, attr))
        }