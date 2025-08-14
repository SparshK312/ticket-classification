#!/usr/bin/env python3
"""
IMPROVED SEMANTIC GROUPING - Hybrid Approach

Implements recommendations:
1. Scoring/threshold system instead of any() matching
2. Require multiple keywords for pattern matching
3. Centralized, data-driven pattern dictionary
4. Hybrid workflow: hardcoded matcher → isolate remainder → semantic analysis
"""

import pandas as pd
import re
import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
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

class ImprovedSemanticGrouper:
    """Improved semantic grouping with scoring and hybrid approach."""
    
    def __init__(self):
        """Initialize the improved semantic grouper."""
        self.logger = logging.getLogger(__name__)
        self.lemmatizer = WordNetLemmatizer()
        
        # CENTRALIZED PATTERN DICTIONARY - All categories in one place
        self.pattern_rules = {
            'Vision': {
                'vision_order_management': {
                    'required_keywords': ['order'],
                    'supporting_keywords': ['amend', 'locked', 'unable', 'modify', 'change', 'edit', 'update'],
                    'min_supporting': 1,
                    'confidence_boost': ['amend', 'locked'],
                    'description': 'Vision order processing and modification issues'
                },
                'vision_license_issues': {
                    'required_keywords': ['license'],
                    'supporting_keywords': ['fusion', 'error', 'expired', 'activation', 'key'],
                    'min_supporting': 1,
                    'confidence_boost': ['fusion'],
                    'description': 'Vision/Fusion licensing and activation problems'
                },
                'vision_system_access': {
                    'required_keywords': ['vision'],
                    'supporting_keywords': ['login', 'access', 'timeout', 'session', 'connection', 'unable'],
                    'min_supporting': 2,
                    'confidence_boost': ['login', 'timeout'],
                    'description': 'Vision system login and access issues'
                }
            },
            'Till': {
                'till_printer_initialization': {
                    'required_keywords': ['printer'],
                    'supporting_keywords': ['initialize', 'failed', 'setup', 'install', 'till'],
                    'min_supporting': 1,
                    'confidence_boost': ['initialize', 'failed'],
                    'description': 'Till printer initialization and setup failures'
                },
                'till_connectivity_issue': {
                    'required_keywords': ['till'],
                    'supporting_keywords': ['server', 'internet', 'offline', 'connection', 'network', 'down', 'unavailable', 'available'],
                    'min_supporting': 1,
                    'confidence_boost': ['server', 'offline', 'unavailable', 'network', 'available'],
                    'description': 'Till connectivity and network connection problems'
                },
                'till_pos_transaction': {
                    'required_keywords': ['till'],
                    'supporting_keywords': ['pos', 'transaction', 'payment', 'card', 'process', 'receipt'],
                    'min_supporting': 1,
                    'confidence_boost': ['transaction', 'payment', 'card'],
                    'description': 'Point-of-sale transaction processing issues'
                },
                'till_performance_issue': {
                    'required_keywords': ['till'],
                    'supporting_keywords': ['slow', 'hang', 'delay', 'performance', 'lag', 'response'],
                    'min_supporting': 1,
                    'confidence_boost': ['slow', 'hang'],
                    'description': 'Till performance and speed issues'
                }
            },
            'Google': {
                'google_2fa': {
                    'required_keywords': ['2fa'],
                    'supporting_keywords': ['google', 'authentication', 'verify', 'code', 'factor'],
                    'min_supporting': 1,
                    'confidence_boost': ['google', 'authentication'],
                    'description': 'Google two-factor authentication issues'
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
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\b\d+\b', ' ', text)
        
        tokens = word_tokenize(text)
        
        # Filter and lemmatize
        keywords = []
        for token in tokens:
            if (len(token) > 2 and 
                token not in self.it_stopwords and
                token.isalpha()):
                lemmatized = self.lemmatizer.lemmatize(token)
                keywords.append(lemmatized)
        
        return list(set(keywords))
    
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
                'boost_matches': boost_matches
            }
        }
    
    def classify_with_hardcoded_rules(self, keywords: list, category: str) -> dict:
        """Classify ticket using improved hardcoded rules."""
        if category not in self.pattern_rules:
            return {'pattern': None, 'confidence': 'none', 'score': 0, 'details': 'Category not in pattern rules'}
        
        category_patterns = self.pattern_rules[category]
        best_match = None
        best_score = 0
        
        # Test all patterns for this category
        for pattern_name, pattern_rule in category_patterns.items():
            score_result = self.calculate_pattern_score(keywords, pattern_rule)
            
            if score_result['score'] > best_score:
                best_score = score_result['score']
                best_match = {
                    'pattern': pattern_name,
                    'confidence': score_result['confidence'],
                    'score': score_result['score'],
                    'details': score_result['details'],
                    'description': pattern_rule['description']
                }
        
        # Only return matches with minimum confidence
        if best_match and best_match['confidence'] in ['high', 'medium']:
            return best_match
        else:
            return {'pattern': None, 'confidence': 'none', 'score': 0, 'details': 'No high/medium confidence matches'}
    
    def create_combined_text(self, row: pd.Series) -> str:
        """Combine Short description and Description for analysis."""
        short_desc = str(row.get('Short description', ''))
        description = str(row.get('Description', ''))
        combined = f"{short_desc} {short_desc} {description}"
        return combined.strip()
    
    def process_tickets_hybrid(self, df: pd.DataFrame) -> dict:
        """Process tickets using hybrid approach."""
        self.logger.info(f"Starting hybrid processing of {len(df):,} tickets")
        
        # Prepare data
        df_work = df.copy()
        df_work['combined_text'] = df_work.apply(self.create_combined_text, axis=1)
        df_work['technical_keywords'] = df_work['combined_text'].apply(self.extract_technical_keywords)
        
        # Phase 1: Hardcoded classification
        classified_tickets = []
        unclassified_tickets = []
        
        for idx, row in df_work.iterrows():
            keywords = row['technical_keywords']
            category = row['Category']
            
            # Try hardcoded classification
            classification = self.classify_with_hardcoded_rules(keywords, category)
            
            if classification['pattern'] is not None:
                classified_tickets.append({
                    'ticket_index': idx,
                    'category': category,
                    'short_description': row['Short description'],
                    'keywords': keywords,
                    'classification': classification,
                    'method': 'hardcoded_rules'
                })
            else:
                unclassified_tickets.append({
                    'ticket_index': idx,
                    'category': category,
                    'short_description': row['Short description'],
                    'keywords': keywords,
                    'classification_attempt': classification,
                    'method': 'needs_semantic_analysis'
                })
        
        results = {
            'total_tickets': len(df_work),
            'classified_by_rules': len(classified_tickets),
            'needs_semantic_analysis': len(unclassified_tickets),
            'classification_rate': round((len(classified_tickets) / len(df_work)) * 100, 2),
            'classified_tickets': classified_tickets,
            'unclassified_tickets': unclassified_tickets
        }
        
        self.logger.info(f"Hardcoded rules classified {len(classified_tickets):,} tickets ({results['classification_rate']}%)")
        self.logger.info(f"{len(unclassified_tickets):,} tickets need semantic analysis")
        
        return results

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def main():
    """Main improved semantic grouping pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # File paths
    input_file = Path('data/processed/consolidated_tickets.csv')
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    if not input_file.exists():
        print(f"❌ Error: Consolidated data file not found at {input_file}")
        return
    
    print("="*80)
    print("IMPROVED SEMANTIC GROUPING - HYBRID APPROACH")
    print("="*80)
    
    # Load consolidated data
    logger.info(f"Loading consolidated ticket data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df):,} consolidated tickets")
    
    # Initialize improved grouper
    grouper = ImprovedSemanticGrouper()
    
    print(f"\n📊 PATTERN RULES LOADED:")
    total_patterns = sum(len(patterns) for patterns in grouper.pattern_rules.values())
    print(f"   Categories with rules: {len(grouper.pattern_rules)}")
    print(f"   Total patterns: {total_patterns}")
    
    for category, patterns in grouper.pattern_rules.items():
        print(f"   {category}: {len(patterns)} patterns")
    
    # Phase 1: Hybrid processing
    print(f"\n🔍 PHASE 1: HARDCODED RULE CLASSIFICATION")
    results = grouper.process_tickets_hybrid(df)
    
    print(f"\n📊 CLASSIFICATION RESULTS:")
    print(f"   Total tickets: {results['total_tickets']:,}")
    print(f"   Classified by rules: {results['classified_by_rules']:,} ({results['classification_rate']}%)")
    print(f"   Need semantic analysis: {results['needs_semantic_analysis']:,}")
    
    # Show some example classifications
    if results['classified_tickets']:
        print(f"\n🎯 EXAMPLE CLASSIFICATIONS:")
        for i, ticket in enumerate(results['classified_tickets'][:5], 1):
            print(f"   {i}. [{ticket['category']}] '{ticket['short_description']}'")
            print(f"      → {ticket['classification']['pattern']} ({ticket['classification']['confidence']} confidence)")
    
    # Save results
    print(f"\n💾 SAVING RESULTS")
    classification_file = output_dir / 'improved_classification_results.json'
    with open(classification_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"   📄 Results saved to: {classification_file}")
    
    print(f"\n✅ IMPROVED SEMANTIC GROUPING COMPLETE")
    print(f"   🎯 {results['classification_rate']:.1f}% of tickets classified with high confidence")
    print(f"   🔍 {results['needs_semantic_analysis']:,} tickets queued for semantic analysis")
    
    logger.info("Improved semantic grouping completed successfully!")

if __name__ == "__main__":
    main()