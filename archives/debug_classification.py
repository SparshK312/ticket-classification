#!/usr/bin/env python3
"""
Debug script to understand why the connectivity ticket is being misclassified
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class DebugClassifier:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        
        # Same patterns as in the main script
        self.pattern_rules = {
            'Till': {
                'till_connectivity_issue': {
                    'required_keywords': ['till'],
                    'supporting_keywords': ['server', 'internet', 'offline', 'connection', 'network', 'down', 'unavailable'],
                    'min_supporting': 1,
                    'confidence_boost': ['server', 'offline', 'unavailable', 'network'],
                    'description': 'Till connectivity and network connection problems'
                },
                'till_pos_transaction': {
                    'required_keywords': ['till'],
                    'supporting_keywords': ['pos', 'transaction', 'payment', 'card', 'process', 'receipt'],
                    'min_supporting': 1,
                    'confidence_boost': ['transaction', 'payment', 'card'],
                    'description': 'Point-of-sale transaction processing issues'
                }
            }
        }
        
        # Stop words for IT context
        self.it_stopwords = set(stopwords.words('english')) | {
            'user', 'system', 'issue', 'problem', 'help', 'support', 'ticket', 'request',
            'please', 'need', 'unable', 'cannot', 'can', 'not', 'working', 'work',
            'store', 'customer', 'staff', 'team', 'manager', 'wickes', 'new', 'old'
        }
    
    def extract_technical_keywords(self, text: str) -> list:
        """Extract technical keywords from text."""
        if pd.isna(text) or text == "":
            return []
        
        # Clean and tokenize
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\b\d+\b', ' ', text)  # Remove standalone numbers
        
        tokens = word_tokenize(text)
        
        # Filter and lemmatize
        keywords = []
        for token in tokens:
            if (len(token) > 2 and 
                token not in self.it_stopwords and
                token.isalpha()):
                lemmatized = self.lemmatizer.lemmatize(token)
                keywords.append(lemmatized)
        
        return list(set(keywords))  # Remove duplicates
    
    def create_combined_text(self, short_desc: str, description: str = "") -> str:
        """Combine Short description and Description for analysis (same as main script)."""
        short_desc = str(short_desc)
        description = str(description)
        # Give more weight to short description (it's more concise/focused)
        combined = f"{short_desc} {short_desc} {description}"
        return combined.strip()
    
    def calculate_pattern_score(self, keywords: list, pattern_rule: dict) -> dict:
        """Calculate confidence score for a pattern match."""
        required_keywords = pattern_rule['required_keywords']
        supporting_keywords = pattern_rule['supporting_keywords']
        min_supporting = pattern_rule['min_supporting']
        confidence_boost = pattern_rule.get('confidence_boost', [])
        
        # Check required keywords
        required_matches = [kw for kw in required_keywords if kw in keywords]
        if len(required_matches) == 0:
            return {'score': 0, 'confidence': 'none', 'details': 'No required keywords found'}
        
        # Check supporting keywords
        supporting_matches = [kw for kw in supporting_keywords if kw in keywords]
        if len(supporting_matches) < min_supporting:
            return {'score': 0, 'confidence': 'none', 'details': f'Insufficient supporting keywords: {len(supporting_matches)}/{min_supporting}'}
        
        # Calculate base score
        required_ratio = len(required_matches) / len(required_keywords)
        supporting_ratio = len(supporting_matches) / len(supporting_keywords)
        base_score = (required_ratio * 0.6) + (supporting_ratio * 0.4)
        
        # Apply confidence boosts
        boost_matches = [kw for kw in confidence_boost if kw in keywords]
        confidence_multiplier = 1.0 + (len(boost_matches) * 0.1)
        
        final_score = base_score * confidence_multiplier
        
        # Determine confidence level
        if final_score >= 0.8:
            confidence = 'high'
        elif final_score >= 0.5:
            confidence = 'medium'
        elif final_score >= 0.3:
            confidence = 'low'
        else:
            confidence = 'none'
        
        return {
            'score': round(final_score, 3),
            'confidence': confidence,
            'details': {
                'required_matches': required_matches,
                'supporting_matches': supporting_matches,
                'boost_matches': boost_matches,
                'required_ratio': round(required_ratio, 3),
                'supporting_ratio': round(supporting_ratio, 3),
                'base_score': round(base_score, 3),
                'confidence_multiplier': round(confidence_multiplier, 3)
            }
        }

def main():
    # The problematic ticket
    test_ticket = "till: till 2 is launching on offline mode, server not available"
    
    classifier = DebugClassifier()
    
    print("="*80)
    print("DEBUG: CLASSIFICATION ANALYSIS")
    print("="*80)
    print(f"Test ticket: '{test_ticket}'")
    
    # Test both with raw text and combined text (as main script does)
    print(f"\n--- RAW TEXT ANALYSIS ---")
    keywords_raw = classifier.extract_technical_keywords(test_ticket)
    print(f"Keywords from raw text: {keywords_raw}")
    
    print(f"\n--- COMBINED TEXT ANALYSIS (same as main script) ---")
    combined_text = classifier.create_combined_text(test_ticket, "")
    print(f"Combined text: '{combined_text}'")
    keywords_combined = classifier.extract_technical_keywords(combined_text)
    print(f"Keywords from combined text: {keywords_combined}")
    
    # Use the same keywords as main script
    keywords = keywords_combined
    
    # Test both patterns
    till_patterns = classifier.pattern_rules['Till']
    results = {}
    
    for pattern_name, pattern_rule in till_patterns.items():
        print(f"\n--- TESTING {pattern_name.upper()} ---")
        print(f"Required keywords: {pattern_rule['required_keywords']}")
        print(f"Supporting keywords: {pattern_rule['supporting_keywords']}")
        print(f"Confidence boost keywords: {pattern_rule.get('confidence_boost', [])}")
        
        score_result = classifier.calculate_pattern_score(keywords, pattern_rule)
        results[pattern_name] = score_result
        print(f"Score result: {score_result}")
        
        if score_result['score'] > 0:
            details = score_result['details']
            print(f"  Required matches: {details['required_matches']}")
            print(f"  Supporting matches: {details['supporting_matches']}")
            print(f"  Boost matches: {details['boost_matches']}")
            print(f"  Base score: {details['base_score']}")
            print(f"  Confidence multiplier: {details['confidence_multiplier']}")
            print(f"  Final score: {score_result['score']}")
            print(f"  Confidence: {score_result['confidence']}")
    
    # Show winner
    print(f"\n--- FINAL CLASSIFICATION ---")
    best_pattern = None
    best_score = 0
    
    for pattern_name, result in results.items():
        if result['score'] > best_score and result['confidence'] in ['high', 'medium']:
            best_score = result['score']
            best_pattern = pattern_name
    
    print(f"Best pattern: {best_pattern}")
    print(f"Best score: {best_score}")
    if best_pattern:
        print(f"Classification: {best_pattern} ({results[best_pattern]['confidence']} confidence)")

if __name__ == "__main__":
    main()