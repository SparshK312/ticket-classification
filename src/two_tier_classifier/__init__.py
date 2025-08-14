"""
Two-Tier Classification System

Production-ready ticket classification system with Level 1 business routing
and Level 2 specific problem identification.

Target Performance:
- Level 1: 85%+ accuracy for business category routing
- Level 2: 70%+ semantic relevance for problem identification
- End-to-end: <2 seconds response time
"""

__version__ = "1.0.0"
__author__ = "Ticket Classification Team"

from .core.level1_classifier import Level1BusinessClassifier
from .core.pipeline_controller import TwoTierClassifier

__all__ = [
    "Level1BusinessClassifier",
    "TwoTierClassifier",
]