#!/usr/bin/env python3
"""
IMPROVED SEMANTIC GROUPING - Hybrid Approach

Implements your recommendations:
1. Scoring/threshold system instead of any() matching
2. Require multiple keywords for pattern matching
3. Centralized, data-driven pattern dictionary
4. Hybrid workflow: hardcoded matcher → isolate remainder → semantic analysis

This addresses false positives and creates a more robust classification system.
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
                    'required_keywords': ['order'],  # Must have this
                    'supporting_keywords': ['amend', 'locked', 'unable', 'modify', 'change', 'edit', 'update'],
                    'min_supporting': 1,  # Need at least 1 supporting keyword
                    'confidence_boost': ['amend', 'locked'],  # These increase confidence
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
                    'min_supporting': 2,  # Need 2+ supporting keywords
                    'confidence_boost': ['login', 'timeout'],
                    'description': 'Vision system login and access issues'
                },
                'vision_performance': {
                    'required_keywords': ['vision'],
                    'supporting_keywords': ['slow', 'timeout', 'hang', 'response', 'delay', 'performance'],
                    'min_supporting': 1,
                    'confidence_boost': ['slow', 'timeout'],
                    'description': 'Vision system performance problems'
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
                'till_pos_transaction': {
                    'required_keywords': ['till'],
                    'supporting_keywords': ['pos', 'transaction', 'payment', 'card', 'process'],
                    'min_supporting': 1,
                    'confidence_boost': ['transaction', 'payment'],
                    'description': 'Point-of-sale transaction processing issues'
                },
                'till_hardware_failure': {
                    'required_keywords': ['till'],
                    'supporting_keywords': ['device', 'hardware', 'connection', 'cable', 'screen', 'failed'],
                    'min_supporting': 1,
                    'confidence_boost': ['hardware', 'device'],
                    'description': 'Till hardware connectivity and device problems'
                }
            },
            'Printing': {
                'printer_hardware': {
                    'required_keywords': ['printer'],
                    'supporting_keywords': ['install', 'setup', 'driver', 'connection', 'device', 'hardware'],
                    'min_supporting': 1,
                    'confidence_boost': ['install', 'driver'],
                    'description': 'Printer hardware and driver installation issues'
                },
                'printing_process': {
                    'required_keywords': ['print'],
                    'supporting_keywords': ['document', 'job', 'queue', 'process', 'failed', 'error'],
                    'min_supporting': 1,
                    'confidence_boost': ['job', 'queue'],
                    'description': 'Print job processing and document printing issues'
                }
            },
            'Chip & Pin': {
                'ped_device_issues': {
                    'required_keywords': ['ped'],
                    'supporting_keywords': ['pin', 'chip', 'device', 'card', 'reader', 'failed'],
                    'min_supporting': 1,
                    'confidence_boost': ['pin', 'chip'],
                    'description': 'Payment Entry Device (PED) hardware issues'
                },
                'payment_authentication': {
                    'required_keywords': ['pin'],
                    'supporting_keywords': ['authentication', 'verify', 'failed', 'error', 'card', 'chip'],
                    'min_supporting': 1,
                    'confidence_boost': ['authentication', 'verify'],
                    'description': 'PIN and chip card authentication problems'
                }
            },
            'Google': {
                'google_2fa': {
                    'required_keywords': ['2fa'],
                    'supporting_keywords': ['google', 'authentication', 'verify', 'code', 'factor'],
                    'min_supporting': 1,
                    'confidence_boost': ['google', 'authentication'],
                    'description': 'Google two-factor authentication issues'
                },
                'google_access': {
                    'required_keywords': ['google'],
                    'supporting_keywords': ['login', 'access', 'account', 'password', 'unable'],
                    'min_supporting': 1,
                    'confidence_boost': ['login', 'access'],
                    'description': 'Google account access and login problems'
                }
            },
            'Active Directory': {
                'ad_password_reset': {
                    'required_keywords': ['password'],
                    'supporting_keywords': ['reset', 'change', 'expired', 'unlock', 'account'],
                    'min_supporting': 1,
                    'confidence_boost': ['reset', 'expired'],
                    'description': 'Active Directory password reset and management'
                },
                'ad_account_access': {
                    'required_keywords': ['account'],
                    'supporting_keywords': ['locked', 'disabled', 'enable', 'unlock', 'access'],
                    'min_supporting': 1,
                    'confidence_boost': ['locked', 'disabled'],
                    'description': 'Active Directory account access and status issues'
                }
            }
        }
        
        # General technical categories (unchanged)
        self.technical_categories = {
            'authentication': ['login', 'password', 'access', 'authentication', '2fa', 'verify', 'signin', 'credential', 'unlock', 'reset', 'locked', 'enable', 'disable'],
            'hardware': ['printer', 'scanner', 'till', 'device', 'hardware', 'cable', 'screen', 'monitor', 'keyboard', 'mouse', 'initialize', 'failed', 'jammed'],
            'network': ['connection', 'network', 'wifi', 'internet', 'connectivity', 'timeout', 'disconnect', 'ping', 'vpn', 'communication'],
            'system_error': ['error', 'crash', 'freeze', 'failure', 'exception', 'bug', 'malfunction', 'corrupt', 'fusion', 'license'],
            'performance': ['slow', 'timeout', 'performance', 'lag', 'delay', 'response', 'hang', 'speed'],
            'license': ['license', 'fusion', 'subscription', 'expired', 'activation', 'key'],
            'data_processing': ['order', 'transaction', 'process', 'update', 'sync', 'import', 'export', 'save', 'amend', 'locked'],
            'user_interface': ['display', 'screen', 'button', 'menu', 'interface', 'gui', 'view', 'window'],
            'configuration': ['setup', 'config', 'setting', 'install', 'configure', 'initialize'],
            'communication': ['email', 'message', 'notification', 'alert', 'phone', 'call']
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
    
    def calculate_pattern_score(self, keywords: list, pattern_rule: dict) -> dict:
        """
        Calculate confidence score for a pattern match.
        
        Args:
            keywords: List of keywords from ticket
            pattern_rule: Pattern rule dictionary
            
        Returns:
            Dict with score, confidence, and details
        """
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
                'supporting_ratio': round(supporting_ratio, 3)
            }
        }
    
    def classify_with_hardcoded_rules(self, keywords: list, category: str) -> dict:\n        \"\"\"\n        Classify ticket using improved hardcoded rules.\n        \n        Args:\n            keywords: List of keywords from ticket\n            category: Ticket category\n            \n        Returns:\n            Classification result with confidence scores\n        \"\"\"\n        if category not in self.pattern_rules:\n            return {'pattern': None, 'confidence': 'none', 'score': 0, 'details': 'Category not in pattern rules'}\n        \n        category_patterns = self.pattern_rules[category]\n        best_match = None\n        best_score = 0\n        all_scores = {}\n        \n        # Test all patterns for this category\n        for pattern_name, pattern_rule in category_patterns.items():\n            score_result = self.calculate_pattern_score(keywords, pattern_rule)\n            all_scores[pattern_name] = score_result\n            \n            if score_result['score'] > best_score:\n                best_score = score_result['score']\n                best_match = {\n                    'pattern': pattern_name,\n                    'confidence': score_result['confidence'],\n                    'score': score_result['score'],\n                    'details': score_result['details'],\n                    'description': pattern_rule['description']\n                }\n        \n        # Only return matches with minimum confidence\n        if best_match and best_match['confidence'] in ['high', 'medium']:\n            best_match['all_scores'] = all_scores\n            return best_match\n        else:\n            return {'pattern': None, 'confidence': 'none', 'score': 0, 'details': 'No high/medium confidence matches', 'all_scores': all_scores}\n    \n    def create_combined_text(self, row: pd.Series) -> str:\n        \"\"\"Combine Short description and Description for analysis.\"\"\"\n        short_desc = str(row.get('Short description', ''))\n        description = str(row.get('Description', ''))\n        \n        # Give more weight to short description (it's more concise/focused)\n        combined = f\"{short_desc} {short_desc} {description}\"\n        return combined.strip()\n    \n    def process_tickets_hybrid(self, df: pd.DataFrame) -> dict:\n        \"\"\"\n        Process tickets using hybrid approach:\n        1. Hardcoded matcher for high-confidence classifications\n        2. Isolate remainder for semantic analysis\n        \n        Args:\n            df: Consolidated tickets dataframe\n            \n        Returns:\n            Processing results with classified and unclassified tickets\n        \"\"\"\n        self.logger.info(f\"Starting hybrid processing of {len(df):,} tickets\")\n        \n        # Prepare data\n        df_work = df.copy()\n        df_work['combined_text'] = df_work.apply(self.create_combined_text, axis=1)\n        df_work['technical_keywords'] = df_work['combined_text'].apply(self.extract_technical_keywords)\n        \n        # Phase 1: Hardcoded classification\n        classified_tickets = []\n        unclassified_tickets = []\n        \n        for idx, row in df_work.iterrows():\n            keywords = row['technical_keywords']\n            category = row['Category']\n            \n            # Try hardcoded classification\n            classification = self.classify_with_hardcoded_rules(keywords, category)\n            \n            if classification['pattern'] is not None:\n                # High-confidence classification\n                classified_tickets.append({\n                    'ticket_index': idx,\n                    'category': category,\n                    'short_description': row['Short description'],\n                    'keywords': keywords,\n                    'classification': classification,\n                    'method': 'hardcoded_rules'\n                })\n            else:\n                # Needs semantic analysis\n                unclassified_tickets.append({\n                    'ticket_index': idx,\n                    'category': category,\n                    'short_description': row['Short description'],\n                    'keywords': keywords,\n                    'classification_attempt': classification,\n                    'method': 'needs_semantic_analysis'\n                })\n        \n        results = {\n            'total_tickets': len(df_work),\n            'classified_by_rules': len(classified_tickets),\n            'needs_semantic_analysis': len(unclassified_tickets),\n            'classification_rate': round((len(classified_tickets) / len(df_work)) * 100, 2),\n            'classified_tickets': classified_tickets,\n            'unclassified_tickets': unclassified_tickets\n        }\n        \n        self.logger.info(f\"Hardcoded rules classified {len(classified_tickets):,} tickets ({results['classification_rate']}%)\")\n        self.logger.info(f\"{len(unclassified_tickets):,} tickets need semantic analysis\")\n        \n        return results\n    \n    def analyze_classification_results(self, results: dict) -> dict:\n        \"\"\"\n        Analyze the classification results to understand patterns.\n        \n        Args:\n            results: Results from process_tickets_hybrid\n            \n        Returns:\n            Analysis of classification patterns\n        \"\"\"\n        classified_tickets = results['classified_tickets']\n        \n        # Pattern frequency analysis\n        pattern_frequency = Counter([ticket['classification']['pattern'] for ticket in classified_tickets])\n        \n        # Category success rates\n        category_stats = defaultdict(lambda: {'total': 0, 'classified': 0})\n        \n        for ticket in classified_tickets:\n            category = ticket['category']\n            category_stats[category]['classified'] += 1\n        \n        for ticket in results['unclassified_tickets']:\n            category = ticket['category']\n            category_stats[category]['total'] += 1\n        \n        # Calculate success rates\n        for category in category_stats:\n            total = category_stats[category]['total'] + category_stats[category]['classified']\n            category_stats[category]['total'] = total\n            category_stats[category]['success_rate'] = round((category_stats[category]['classified'] / total) * 100, 2) if total > 0 else 0\n        \n        # Confidence distribution\n        confidence_dist = Counter([ticket['classification']['confidence'] for ticket in classified_tickets])\n        \n        return {\n            'pattern_frequency': dict(pattern_frequency.most_common()),\n            'category_success_rates': dict(category_stats),\n            'confidence_distribution': dict(confidence_dist),\n            'top_patterns': dict(pattern_frequency.most_common(10))\n        }\n\ndef setup_logging():\n    \"\"\"Setup logging configuration.\"\"\"\n    logging.basicConfig(\n        level=logging.INFO,\n        format='%(asctime)s - %(levelname)s - %(message)s',\n        handlers=[logging.StreamHandler()]\n    )\n\ndef main():\n    \"\"\"Main improved semantic grouping pipeline.\"\"\"\n    setup_logging()\n    logger = logging.getLogger(__name__)\n    \n    # File paths\n    input_file = Path('data/processed/consolidated_tickets.csv')\n    output_dir = Path('outputs')\n    output_dir.mkdir(exist_ok=True)\n    \n    if not input_file.exists():\n        print(f\"❌ Error: Consolidated data file not found at {input_file}\")\n        return\n    \n    print(\"=\"*80)\n    print(\"IMPROVED SEMANTIC GROUPING - HYBRID APPROACH\")\n    print(\"=\"*80)\n    \n    # Load consolidated data\n    logger.info(f\"Loading consolidated ticket data from {input_file}\")\n    df = pd.read_csv(input_file)\n    logger.info(f\"Loaded {len(df):,} consolidated tickets\")\n    \n    # Initialize improved grouper\n    grouper = ImprovedSemanticGrouper()\n    \n    print(f\"\\n📊 PATTERN RULES LOADED:\")\n    total_patterns = sum(len(patterns) for patterns in grouper.pattern_rules.values())\n    print(f\"   Categories with rules: {len(grouper.pattern_rules)}\")\n    print(f\"   Total patterns: {total_patterns}\")\n    \n    for category, patterns in grouper.pattern_rules.items():\n        print(f\"   {category}: {len(patterns)} patterns\")\n    \n    # Phase 1: Hybrid processing\n    print(f\"\\n🔍 PHASE 1: HARDCODED RULE CLASSIFICATION\")\n    results = grouper.process_tickets_hybrid(df)\n    \n    print(f\"\\n📊 CLASSIFICATION RESULTS:\")\n    print(f\"   Total tickets: {results['total_tickets']:,}\")\n    print(f\"   Classified by rules: {results['classified_by_rules']:,} ({results['classification_rate']}%)\")\n    print(f\"   Need semantic analysis: {results['needs_semantic_analysis']:,}\")\n    \n    # Phase 2: Analyze results\n    print(f\"\\n🔍 PHASE 2: RESULT ANALYSIS\")\n    analysis = grouper.analyze_classification_results(results)\n    \n    print(f\"\\n🎯 TOP CLASSIFIED PATTERNS:\")\n    for i, (pattern, count) in enumerate(list(analysis['top_patterns'].items())[:10], 1):\n        print(f\"   {i:2d}. {pattern}: {count:,} tickets\")\n    \n    print(f\"\\n📈 CATEGORY SUCCESS RATES:\")\n    for category, stats in sorted(analysis['category_success_rates'].items(), \n                                key=lambda x: x[1]['success_rate'], reverse=True):\n        print(f\"   {category}: {stats['classified']:,}/{stats['total']:,} ({stats['success_rate']:.1f}%)\")\n    \n    print(f\"\\n🎯 CONFIDENCE DISTRIBUTION:\")\n    for confidence, count in analysis['confidence_distribution'].items():\n        print(f\"   {confidence}: {count:,} tickets\")\n    \n    # Save results\n    print(f\"\\n💾 SAVING RESULTS\")\n    \n    # Save classification results\n    classification_file = output_dir / 'improved_classification_results.json'\n    with open(classification_file, 'w') as f:\n        json.dump({\n            'results': results,\n            'analysis': analysis,\n            'methodology': {\n                'approach': 'Hybrid: Hardcoded Rules + Semantic Analysis Queue',\n                'improvements': [\n                    'Scoring system instead of any() matching',\n                    'Multiple keyword requirements',\n                    'Centralized pattern dictionary',\n                    'Confidence-based classification'\n                ]\n            }\n        }, f, indent=2, default=str)\n    print(f\"   📄 Classification results saved to: {classification_file}\")\n    \n    # Save tickets needing semantic analysis\n    if results['unclassified_tickets']:\n        unclassified_df = pd.DataFrame([\n            {\n                'ticket_index': ticket['ticket_index'],\n                'category': ticket['category'],\n                'short_description': ticket['short_description'],\n                'keywords': ', '.join(ticket['keywords']),\n                'reason_unclassified': ticket['classification_attempt']['details']\n            }\n            for ticket in results['unclassified_tickets']\n        ])\n        \n        unclassified_file = output_dir / 'tickets_for_semantic_analysis.csv'\n        unclassified_df.to_csv(unclassified_file, index=False)\n        print(f\"   📄 Unclassified tickets saved to: {unclassified_file}\")\n    \n    print(f\"\\n✅ IMPROVED SEMANTIC GROUPING COMPLETE\")\n    print(f\"   🎯 {results['classification_rate']:.1f}% of tickets classified with high confidence\")\n    print(f\"   🔍 {results['needs_semantic_analysis']:,} tickets queued for semantic analysis\")\n    print(f\"   📁 Results saved to: {output_dir}\")\n    \n    print(f\"\\n🔄 NEXT STEPS:\")\n    print(f\"   1. Review high-confidence classifications for accuracy\")\n    print(f\"   2. Run semantic analysis on remaining {results['needs_semantic_analysis']:,} tickets\")\n    print(f\"   3. Validate and refine pattern rules based on results\")\n    \n    logger.info(\"Improved semantic grouping completed successfully!\")\n\nif __name__ == \"__main__\":\n    main()