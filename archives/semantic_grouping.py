#!/usr/bin/env python3
"""
PHASE 1: Semantic Grouping - Keyword Extraction & Problem Identification

This script implements the hybrid approach for finding "core problem statements":
1. Extract technical keywords from Short description + Description fields
2. Build problem taxonomy (hardware, software, network, authentication, etc.)
3. Group tickets by underlying technical issues, not surface symptoms
4. Start with category-wise analysis, then cross-category patterns

Manager's goal: Group things that are "meaningfully correct" - same problem statement.
"""

import pandas as pd
import numpy as np
import re
import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
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
        
        # Technical keywords taxonomy
        self.technical_categories = {
            'authentication': ['login', 'password', 'access', 'authentication', '2fa', 'verify', 'signin', 'credential', 'unlock', 'reset'],
            'hardware': ['printer', 'scanner', 'till', 'device', 'hardware', 'cable', 'screen', 'monitor', 'keyboard', 'mouse'],
            'network': ['connection', 'network', 'wifi', 'internet', 'connectivity', 'timeout', 'disconnect', 'ping', 'vpn'],
            'system_error': ['error', 'crash', 'freeze', 'failure', 'exception', 'bug', 'malfunction', 'corrupt'],
            'performance': ['slow', 'timeout', 'performance', 'lag', 'delay', 'response', 'hang', 'speed'],
            'license': ['license', 'fusion', 'subscription', 'expired', 'activation', 'key'],
            'data_processing': ['order', 'transaction', 'process', 'update', 'sync', 'import', 'export', 'save'],
            'user_interface': ['display', 'screen', 'button', 'menu', 'interface', 'gui', 'view', 'window'],
            'configuration': ['setup', 'config', 'setting', 'install', 'configure', 'initialize'],
            'communication': ['email', 'message', 'notification', 'alert', 'phone', 'call']
        }
        
        # Stop words for IT context
        self.it_stopwords = set(stopwords.words('english')) | {
            'user', 'system', 'issue', 'problem', 'help', 'support', 'ticket', 'request',
            'please', 'need', 'unable', 'cannot', 'can', 'not', 'working', 'work',
            'store', 'customer', 'staff', 'team', 'manager', 'wickes'
        }
    
    def extract_technical_keywords(self, text: str) -> list:
        """
        Extract technical keywords from text.
        
        Args:
            text: Input text (Short description + Description)
            
        Returns:
            List of technical keywords
        """
        if pd.isna(text) or text == "":
            return []
        
        # Clean and tokenize
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove numbers
        
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
        """
        Categorize keywords by technical domain.
        
        Args:
            keywords: List of extracted keywords
            
        Returns:
            Dictionary mapping categories to found keywords
        """
        categorized = defaultdict(list)
        
        for keyword in keywords:
            for category, category_keywords in self.technical_categories.items():
                if keyword in category_keywords:
                    categorized[category].append(keyword)
        
        return dict(categorized)
    
    def create_combined_text(self, row: pd.Series) -> str:
        """
        Combine Short description and Description for analysis.
        
        Args:
            row: DataFrame row
            
        Returns:
            Combined text
        """
        short_desc = str(row.get('Short description', ''))
        description = str(row.get('Description', ''))
        
        # Give more weight to short description (it's more concise/focused)
        combined = f"{short_desc} {short_desc} {description}"
        return combined.strip()
    
    def analyze_category_problems(self, df: pd.DataFrame, category: str) -> dict:
        """
        Analyze problems within a specific category.
        
        Args:
            df: Consolidated tickets dataframe
            category: Category to analyze (e.g., 'Vision', 'Till')
            
        Returns:
            Analysis results for the category
        """
        self.logger.info(f"Analyzing problems in {category} category")
        
        # Filter to category
        category_df = df[df['Category'] == category].copy()
        
        if len(category_df) == 0:
            return {'category': category, 'tickets': 0, 'analysis': {}}
        
        self.logger.info(f"Processing {len(category_df)} tickets in {category}")
        
        # Extract keywords for each ticket
        category_df['combined_text'] = category_df.apply(self.create_combined_text, axis=1)
        category_df['technical_keywords'] = category_df['combined_text'].apply(self.extract_technical_keywords)
        category_df['keyword_categories'] = category_df['technical_keywords'].apply(self.categorize_keywords)
        
        # Find most common keywords in this category
        all_keywords = []
        for keywords in category_df['technical_keywords']:
            all_keywords.extend(keywords)
        
        keyword_frequency = Counter(all_keywords)
        
        # Find most common technical problem types
        problem_types = defaultdict(int)
        for keyword_cats in category_df['keyword_categories']:
            for tech_category in keyword_cats.keys():
                problem_types[tech_category] += 1
        
        # Create TF-IDF vectors for semantic similarity
        texts = category_df['combined_text'].tolist()
        
        if len(texts) > 1:
            vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(texts)
                
                # Perform clustering to find semantic groups
                n_clusters = min(10, max(2, len(texts) // 5))  # Adaptive cluster count
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(tfidf_matrix)
                
                category_df['semantic_cluster'] = cluster_labels
                
                # Analyze each semantic cluster
                semantic_groups = {}
                for cluster_id in range(n_clusters):
                    cluster_tickets = category_df[category_df['semantic_cluster'] == cluster_id]
                    
                    if len(cluster_tickets) > 1:  # Only groups with multiple tickets
                        # Find representative keywords for this cluster
                        cluster_keywords = []
                        for keywords in cluster_tickets['technical_keywords']:
                            cluster_keywords.extend(keywords)
                        
                        cluster_keyword_freq = Counter(cluster_keywords)
                        
                        semantic_groups[f"cluster_{cluster_id}"] = {
                            'ticket_count': len(cluster_tickets),
                            'top_keywords': dict(cluster_keyword_freq.most_common(10)),
                            'sample_descriptions': cluster_tickets['Short description'].head(5).tolist(),
                            'ticket_indices': cluster_tickets.index.tolist(),
                            'dominant_problem_types': [k for k, v in Counter([
                                tech_cat for keyword_cats in cluster_tickets['keyword_categories'] 
                                for tech_cat in keyword_cats.keys()
                            ]).most_common(3)]
                        }
                
            except Exception as e:
                self.logger.warning(f"TF-IDF clustering failed for {category}: {e}")
                semantic_groups = {}
        else:
            semantic_groups = {}
        
        return {
            'category': category,
            'total_tickets': len(category_df),
            'top_keywords': dict(keyword_frequency.most_common(20)),
            'problem_types': dict(problem_types),
            'semantic_groups': semantic_groups,
            'sample_tickets': category_df[['Short description', 'technical_keywords', 'keyword_categories']].head(10).to_dict('records')
        }
    
    def find_cross_category_patterns(self, df: pd.DataFrame, category_analyses: dict) -> dict:
        """
        Find common problem patterns across categories.
        
        Args:
            df: Consolidated tickets dataframe
            category_analyses: Results from category-wise analysis
            
        Returns:
            Cross-category pattern analysis
        """
        self.logger.info("Finding cross-category patterns")
        
        # Aggregate problem types across all categories
        global_problem_types = defaultdict(int)
        global_keywords = defaultdict(int)
        
        for category, analysis in category_analyses.items():
            if 'problem_types' in analysis:
                for problem_type, count in analysis['problem_types'].items():
                    global_problem_types[problem_type] += count
            
            if 'top_keywords' in analysis:
                for keyword, count in analysis['top_keywords'].items():
                    global_keywords[keyword] += count
        
        # Find technical patterns that span multiple categories
        cross_category_patterns = {}
        
        for problem_type, total_count in global_problem_types.items():
            if total_count >= 5:  # Minimum threshold for cross-category pattern
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
                        'example_keywords': [k for k in self.technical_categories[problem_type] 
                                           if k in global_keywords][:5]
                    }
        
        return {
            'global_problem_types': dict(global_problem_types),
            'global_keywords': dict(sorted(global_keywords.items(), key=lambda x: x[1], reverse=True)[:30]),
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
            'approach': 'Hybrid (Keywords + TF-IDF + Clustering)',
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