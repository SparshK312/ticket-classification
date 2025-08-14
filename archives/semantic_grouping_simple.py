#!/usr/bin/env python3
"""
PHASE 1: Semantic Grouping - Keyword Extraction & Problem Identification (Simplified)

This script implements keyword extraction and problem identification without scikit-learn:
1. Extract technical keywords from Short description + Description fields
2. Build problem taxonomy (hardware, software, network, authentication, etc.)
3. Group tickets by underlying technical issues, not surface symptoms
4. Start with category-wise analysis, then cross-category patterns

Manager's goal: Group things that are "meaningfully correct" - same problem statement.
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
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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

class SemanticGrouper:
    """Semantic grouping to find core problem statements."""
    
    def __init__(self):
        """Initialize the semantic grouper."""
        self.logger = logging.getLogger(__name__)
        self.lemmatizer = WordNetLemmatizer()
        
        # Technical keywords taxonomy - expanded and refined
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
        
        # Vision-specific problem patterns
        self.vision_patterns = {
            'order_management': ['order', 'amend', 'locked', 'unable', 'modify', 'change', 'edit'],
            'system_access': ['login', 'access', 'timeout', 'session', 'connection'],
            'license_issues': ['fusion', 'license', 'error', 'expired', 'activation'],
            'performance_issues': ['slow', 'timeout', 'hang', 'response', 'delay']
        }
        
        # Till-specific problem patterns  
        self.till_patterns = {
            'printer_issues': ['printer', 'initialize', 'failed', 'setup', 'install'],
            'pos_problems': ['till', 'pos', 'transaction', 'payment', 'card'],
            'hardware_failure': ['device', 'hardware', 'connection', 'cable', 'screen']
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
        
        return keywords
    
    def categorize_keywords(self, keywords: list) -> dict:
        """Categorize keywords by technical domain."""
        categorized = defaultdict(list)
        
        for keyword in keywords:
            for category, category_keywords in self.technical_categories.items():
                if keyword in category_keywords:
                    categorized[category].append(keyword)
        
        return dict(categorized)
    
    def identify_problem_pattern(self, keywords: list, category: str) -> str:
        """Identify specific problem pattern based on keywords and category."""
        if category.lower() == 'vision':
            for pattern, pattern_keywords in self.vision_patterns.items():
                if any(keyword in pattern_keywords for keyword in keywords):
                    return f"vision_{pattern}"
        
        elif category.lower() == 'till':
            for pattern, pattern_keywords in self.till_patterns.items():
                if any(keyword in pattern_keywords for keyword in keywords):
                    return f"till_{pattern}"
        
        # Generic technical categorization
        keyword_categories = self.categorize_keywords(keywords)
        if keyword_categories:
            # Return the most prominent technical category
            return max(keyword_categories.keys(), key=lambda k: len(keyword_categories[k]))
        
        return "other"
    
    def create_combined_text(self, row: pd.Series) -> str:
        """Combine Short description and Description for analysis."""
        short_desc = str(row.get('Short description', ''))
        description = str(row.get('Description', ''))
        
        # Give more weight to short description (it's more concise/focused)
        combined = f"{short_desc} {short_desc} {description}"
        return combined.strip()
    
    def find_similar_tickets_by_keywords(self, tickets_df: pd.DataFrame, min_similarity=0.3) -> list:
        """Find groups of tickets with similar keyword patterns."""
        groups = []
        processed_indices = set()
        
        for idx, ticket in tickets_df.iterrows():
            if idx in processed_indices:
                continue
                
            ticket_keywords = set(ticket['technical_keywords'])
            if len(ticket_keywords) == 0:
                continue
            
            # Find similar tickets
            similar_tickets = [idx]
            
            for other_idx, other_ticket in tickets_df.iterrows():
                if other_idx <= idx or other_idx in processed_indices:
                    continue
                
                other_keywords = set(other_ticket['technical_keywords'])
                if len(other_keywords) == 0:
                    continue
                
                # Calculate keyword similarity (Jaccard similarity)
                intersection = ticket_keywords.intersection(other_keywords)
                union = ticket_keywords.union(other_keywords)
                
                if len(union) > 0:
                    similarity = len(intersection) / len(union)
                    if similarity >= min_similarity:
                        similar_tickets.append(other_idx)
            
            if len(similar_tickets) > 1:  # Group found
                processed_indices.update(similar_tickets)
                
                # Analyze the group
                group_tickets = tickets_df.loc[similar_tickets]
                all_keywords = []
                for keywords in group_tickets['technical_keywords']:
                    all_keywords.extend(keywords)
                
                keyword_freq = Counter(all_keywords)
                
                groups.append({
                    'group_id': len(groups),
                    'ticket_count': len(similar_tickets),
                    'ticket_indices': similar_tickets,
                    'common_keywords': dict(keyword_freq.most_common(10)),
                    'sample_descriptions': group_tickets['Short description'].head(5).tolist(),
                    'problem_patterns': group_tickets['problem_pattern'].value_counts().to_dict()
                })
        
        return groups
    
    def analyze_category_problems(self, df: pd.DataFrame, category: str) -> dict:
        """Analyze problems within a specific category."""
        self.logger.info(f"Analyzing problems in {category} category")
        
        # Filter to category
        category_df = df[df['Category'] == category].copy()
        
        if len(category_df) == 0:
            return {'category': category, 'tickets': 0, 'analysis': {}}
        
        self.logger.info(f"Processing {len(category_df)} tickets in {category}")
        
        # Extract keywords and patterns for each ticket
        category_df['combined_text'] = category_df.apply(self.create_combined_text, axis=1)
        category_df['technical_keywords'] = category_df['combined_text'].apply(self.extract_technical_keywords)
        category_df['keyword_categories'] = category_df['technical_keywords'].apply(self.categorize_keywords)
        category_df['problem_pattern'] = category_df.apply(
            lambda row: self.identify_problem_pattern(row['technical_keywords'], category), axis=1
        )
        
        # Find most common keywords in this category
        all_keywords = []
        for keywords in category_df['technical_keywords']:
            all_keywords.extend(keywords)
        
        keyword_frequency = Counter(all_keywords)
        
        # Find most common problem patterns
        pattern_frequency = Counter(category_df['problem_pattern'])
        
        # Find most common technical problem types
        problem_types = defaultdict(int)
        for keyword_cats in category_df['keyword_categories']:
            for tech_category in keyword_cats.keys():
                problem_types[tech_category] += 1
        
        # Find semantic groups using keyword similarity
        semantic_groups = self.find_similar_tickets_by_keywords(category_df)
        
        return {
            'category': category,
            'total_tickets': len(category_df),
            'top_keywords': dict(keyword_frequency.most_common(20)),
            'problem_patterns': dict(pattern_frequency.most_common(10)),
            'problem_types': dict(problem_types),
            'semantic_groups': {f"group_{i}": group for i, group in enumerate(semantic_groups)},
            'sample_tickets': category_df[['Short description', 'technical_keywords', 'problem_pattern']].head(10).to_dict('records')
        }
    
    def find_cross_category_patterns(self, df: pd.DataFrame, category_analyses: dict) -> dict:
        """Find common problem patterns across categories."""
        self.logger.info("Finding cross-category patterns")
        
        # Aggregate problem types across all categories
        global_problem_types = defaultdict(int)
        global_keywords = defaultdict(int)
        global_patterns = defaultdict(int)
        
        for category, analysis in category_analyses.items():
            if 'problem_types' in analysis:
                for problem_type, count in analysis['problem_types'].items():
                    global_problem_types[problem_type] += count
            
            if 'top_keywords' in analysis:
                for keyword, count in analysis['top_keywords'].items():
                    global_keywords[keyword] += count
            
            if 'problem_patterns' in analysis:
                for pattern, count in analysis['problem_patterns'].items():
                    global_patterns[pattern] += count
        
        # Find technical patterns that span multiple categories
        cross_category_patterns = {}
        
        for problem_type, total_count in global_problem_types.items():
            if total_count >= 5:  # Minimum threshold
                categories_with_problem = []
                for category, analysis in category_analyses.items():
                    if problem_type in analysis.get('problem_types', {}):
                        categories_with_problem.append({
                            'category': category,
                            'count': analysis['problem_types'][problem_type]
                        })
                
                if len(categories_with_problem) >= 2:  # Spans multiple categories
                    cross_category_patterns[problem_type] = {
                        'total_tickets': total_count,
                        'categories_affected': categories_with_problem,
                        'example_keywords': [k for k in self.technical_categories[problem_type]][:5]
                    }
        
        return {
            'global_problem_types': dict(global_problem_types),
            'global_keywords': dict(sorted(global_keywords.items(), key=lambda x: x[1], reverse=True)[:30]),
            'global_patterns': dict(sorted(global_patterns.items(), key=lambda x: x[1], reverse=True)[:20]),
            'cross_category_patterns': cross_category_patterns
        }

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def main():
    """Main semantic grouping pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # File paths
    input_file = Path('data/processed/consolidated_tickets.csv')
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    if not input_file.exists():
        print(f"❌ Error: Consolidated data file not found at {input_file}")
        print("   Please run create_consolidated_dataset.py first.")
        return
    
    print("="*80)
    print("PHASE 1: SEMANTIC GROUPING - KEYWORD EXTRACTION & PROBLEM IDENTIFICATION")
    print("="*80)
    
    # Load consolidated data
    logger.info(f"Loading consolidated ticket data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df):,} consolidated tickets")
    
    # Initialize semantic grouper
    grouper = SemanticGrouper()
    
    # Get top categories for analysis
    top_categories = df['Category'].value_counts().head(10)
    print(f"\n📊 TOP 10 CATEGORIES TO ANALYZE:")
    for i, (category, count) in enumerate(top_categories.items(), 1):
        print(f"   {i:2d}. {category}: {count:,} tickets")
    
    # Phase 1: Category-wise semantic analysis
    print(f"\n🔍 PHASE 1A: CATEGORY-WISE SEMANTIC ANALYSIS")
    category_analyses = {}
    
    for category in top_categories.index[:5]:  # Start with top 5 categories
        print(f"\n   Analyzing {category}...")
        analysis = grouper.analyze_category_problems(df, category)
        category_analyses[category] = analysis
        
        # Print summary for this category
        print(f"      Tickets: {analysis['total_tickets']:,}")
        print(f"      Top keywords: {list(analysis['top_keywords'].keys())[:5]}")
        print(f"      Problem patterns: {list(analysis['problem_patterns'].keys())[:3]}")
        print(f"      Problem types: {list(analysis['problem_types'].keys())}")
        print(f"      Semantic groups: {len(analysis['semantic_groups'])}")
    
    # Phase 1B: Cross-category pattern analysis
    print(f"\n🔍 PHASE 1B: CROSS-CATEGORY PATTERN ANALYSIS")
    cross_patterns = grouper.find_cross_category_patterns(df, category_analyses)
    
    print(f"   Global problem types found: {len(cross_patterns['global_problem_types'])}")
    print(f"   Cross-category patterns: {len(cross_patterns['cross_category_patterns'])}")
    
    # Show top cross-category patterns
    print(f"\n   TOP CROSS-CATEGORY PATTERNS:")
    for i, (pattern, data) in enumerate(list(cross_patterns['cross_category_patterns'].items())[:5], 1):
        affected_categories = [cat['category'] for cat in data['categories_affected']]
        print(f"      {i}. {pattern.upper()}: {data['total_tickets']} tickets across {affected_categories}")
    
    # Show top global keywords
    print(f"\n   TOP GLOBAL KEYWORDS:")
    for i, (keyword, count) in enumerate(list(cross_patterns['global_keywords'].items())[:10], 1):
        print(f"      {i:2d}. '{keyword}': {count} occurrences")
    
    # Save detailed results
    print(f"\n💾 SAVING DETAILED ANALYSIS RESULTS")
    
    # Save category analyses
    for category, analysis in category_analyses.items():
        category_file = output_dir / f'semantic_analysis_{category.lower().replace(" ", "_")}.json'
        with open(category_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"   📄 {category} analysis saved to: {category_file}")
    
    # Save cross-category patterns
    cross_patterns_file = output_dir / 'cross_category_patterns.json'
    with open(cross_patterns_file, 'w') as f:
        json.dump(cross_patterns, f, indent=2, default=str)
    print(f"   📄 Cross-category patterns saved to: {cross_patterns_file}")
    
    # Save complete semantic analysis
    complete_analysis = {
        'analysis_timestamp': pd.Timestamp.now().isoformat(),
        'methodology': {
            'phase': 'PHASE 1 - Semantic Grouping',
            'approach': 'Keyword Extraction + Pattern Matching',
            'data_sources': 'Short description + Description',
            'categories_analyzed': list(category_analyses.keys())
        },
        'category_analyses': category_analyses,
        'cross_category_patterns': cross_patterns,
        'summary': {
            'total_tickets_analyzed': len(df),
            'categories_analyzed': len(category_analyses),
            'global_problem_types': len(cross_patterns['global_problem_types']),
            'cross_category_patterns': len(cross_patterns['cross_category_patterns'])
        }
    }
    
    complete_file = output_dir / 'semantic_grouping_phase1.json'
    with open(complete_file, 'w') as f:
        json.dump(complete_analysis, f, indent=2, default=str)
    print(f"   📄 Complete analysis saved to: {complete_file}")
    
    print(f"\n✅ PHASE 1 COMPLETE: SEMANTIC GROUPING - KEYWORD EXTRACTION")
    print(f"   📊 Analyzed {len(df):,} consolidated tickets")
    print(f"   🔍 Found {len(cross_patterns['global_problem_types'])} global problem types")
    print(f"   🌐 Identified {len(cross_patterns['cross_category_patterns'])} cross-category patterns")
    print(f"   📁 Results saved to: {output_dir}")
    
    print(f"\n🔄 NEXT STEPS:")
    print(f"   1. Review semantic groups within major categories")
    print(f"   2. Validate that groups represent core problem statements")
    print(f"   3. Refine groupings based on business logic")
    print(f"   4. Create final problem statement taxonomy")
    
    logger.info("Phase 1 semantic grouping completed successfully!")

if __name__ == "__main__":
    main()