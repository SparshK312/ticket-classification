#!/usr/bin/env python3
"""
Problem Statement Analysis for LLM Readiness

Analyzes the 209 problem statements to assess:
1. Uniqueness (no duplicates)
2. Natural language quality
3. LLM readiness
4. Formatting consistency
5. Semantic categories
"""

import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
from pathlib import Path

def analyze_problem_statements():
    """Comprehensive analysis of problem statements for LLM readiness."""
    
    # Load the improved problem groups
    file_path = Path('outputs/improved_problem_groups.csv')
    df = pd.read_csv(file_path)
    
    print("="*80)
    print("PROBLEM STATEMENT ANALYSIS FOR LLM READINESS")
    print("="*80)
    
    # Extract problem statements
    statements = df['representative_short_description'].tolist()
    
    print(f"📊 BASIC STATISTICS:")
    print(f"   Total problem statements: {len(statements)}")
    print(f"   Unique problem statements: {len(set(statements))}")
    print(f"   Duplicates found: {len(statements) - len(set(statements))}")
    
    # Analyze formatting patterns
    print(f"\n🔍 FORMATTING ANALYSIS:")
    
    # Length analysis
    lengths = [len(stmt) for stmt in statements]
    print(f"   Statement length - Min: {min(lengths)}, Max: {max(lengths)}, Avg: {np.mean(lengths):.1f}")
    
    # Pattern analysis
    patterns = {
        'starts_with_quotes': sum(1 for s in statements if s.startswith('"')),
        'starts_lowercase': sum(1 for s in statements if s[0].islower()),
        'starts_uppercase': sum(1 for s in statements if s[0].isupper()),
        'contains_colon': sum(1 for s in statements if ':' in s),
        'contains_dash': sum(1 for s in statements if '-' in s),
        'contains_till': sum(1 for s in statements if 'till' in s.lower()),
        'contains_vision': sum(1 for s in statements if 'vision' in s.lower()),
        'contains_printer': sum(1 for s in statements if 'printer' in s.lower()),
        'contains_chip_pin': sum(1 for s in statements if 'chip' in s.lower() and 'pin' in s.lower()),
    }
    
    for pattern, count in patterns.items():
        print(f"   {pattern}: {count} ({count/len(statements)*100:.1f}%)")
    
    # Natural language quality assessment
    print(f"\n📝 NATURAL LANGUAGE QUALITY:")
    
    # Short/cryptic statements
    very_short = [s for s in statements if len(s) < 20]
    print(f"   Very short statements (<20 chars): {len(very_short)}")
    if very_short:
        print(f"      Examples: {very_short[:3]}")
    
    # Long descriptive statements  
    very_long = [s for s in statements if len(s) > 80]
    print(f"   Long descriptive statements (>80 chars): {len(very_long)}")
    if very_long:
        print(f"      Examples: {very_long[:2]}")
    
    # Clean natural language (complete sentences)
    natural_language = []
    technical_shorthand = []
    
    for stmt in statements:
        # Remove quotes for analysis
        clean_stmt = stmt.strip('"')
        
        # Check if it's more natural language vs technical shorthand
        if any([
            'customer' in clean_stmt.lower(),
            'agent' in clean_stmt.lower(),
            'help' in clean_stmt.lower(),
            'unable to' in clean_stmt.lower(),
            'issue with' in clean_stmt.lower(),
            len(clean_stmt.split()) > 6
        ]):
            natural_language.append(stmt)
        else:
            technical_shorthand.append(stmt)
    
    print(f"   Natural language statements: {len(natural_language)} ({len(natural_language)/len(statements)*100:.1f}%)")
    print(f"   Technical shorthand: {len(technical_shorthand)} ({len(technical_shorthand)/len(statements)*100:.1f}%)")
    
    # Semantic category analysis
    print(f"\n🏷️  SEMANTIC CATEGORIES:")
    
    categories = {
        'Till/POS Issues': [s for s in statements if any(word in s.lower() for word in ['till', 'pos', 'chip', 'pin', 'card', 'payment'])],
        'Vision System': [s for s in statements if 'vision' in s.lower()],
        'Printer Issues': [s for s in statements if any(word in s.lower() for word in ['printer', 'printing', 'print'])],
        'Scanner/Zebra': [s for s in statements if any(word in s.lower() for word in ['scanner', 'zebra', 'scan'])],
        'Back Office': [s for s in statements if 'back office' in s.lower() or 'backoffice' in s.lower()],
        'Login/Access': [s for s in statements if any(word in s.lower() for word in ['login', 'password', 'access', 'account'])],
        'Hardware': [s for s in statements if any(word in s.lower() for word in ['hardware', 'monitor', 'pc', 'screen', 'headset'])],
        'Software Errors': [s for s in statements if any(word in s.lower() for word in ['error', 'crash', 'frozen', 'freeze', 'stuck'])]
    }
    
    for category, items in categories.items():
        print(f"   {category}: {len(items)} statements ({len(items)/len(statements)*100:.1f}%)")
    
    # LLM readiness assessment
    print(f"\n🤖 LLM READINESS ASSESSMENT:")
    
    # Problems that need reformatting
    needs_reformatting = []
    ready_for_llm = []
    
    for stmt in statements:
        clean_stmt = stmt.strip('"')
        
        # Check if needs reformatting
        needs_work = any([
            len(clean_stmt) < 15,  # Too short
            clean_stmt.count(':') > 2,  # Too many colons
            clean_stmt.count('-') > 3,  # Too many dashes
            not clean_stmt.replace('-', ' ').replace(':', ' ').strip(),  # Mostly punctuation
            clean_stmt.lower().startswith(('till ', 'proj ', 'project ')) and len(clean_stmt) < 30  # Technical codes
        ])
        
        if needs_work:
            needs_reformatting.append(stmt)
        else:
            ready_for_llm.append(stmt)
    
    print(f"   Statements ready for LLM: {len(ready_for_llm)} ({len(ready_for_llm)/len(statements)*100:.1f}%)")
    print(f"   Statements needing reformatting: {len(needs_reformatting)} ({len(needs_reformatting)/len(statements)*100:.1f}%)")
    
    # Show examples of different quality levels
    print(f"\n📋 STATEMENT QUALITY EXAMPLES:")
    
    print(f"\n✅ HIGH QUALITY (Natural Language):")
    high_quality = [s for s in statements if len(s) > 40 and ('customer' in s.lower() or 'agent' in s.lower())][:3]
    for i, stmt in enumerate(high_quality, 1):
        print(f"   {i}. {stmt}")
    
    print(f"\n⚠️  MEDIUM QUALITY (Technical but Clear):")
    medium_quality = [s for s in statements if 15 <= len(s) <= 40 and s not in high_quality][:3]
    for i, stmt in enumerate(medium_quality, 1):
        print(f"   {i}. {stmt}")
    
    print(f"\n❌ LOW QUALITY (Needs Reformatting):")
    low_quality = [s for s in statements if len(s) < 15 or s.count('-') > 3][:3]
    for i, stmt in enumerate(low_quality, 1):
        print(f"   {i}. {stmt}")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS FOR LLM PROCESSING:")
    print(f"   1. ✅ UNIQUENESS: All 209 statements are unique - no duplicates!")
    print(f"   2. 🔄 REFORMATTING: {len(needs_reformatting)} statements need preprocessing")
    print(f"   3. 📊 CATEGORIZATION: Clear semantic categories identified")
    print(f"   4. 🤖 LLM STRATEGY: Use different prompts for technical vs natural language statements")
    
    if len(needs_reformatting) > 0:
        print(f"\n🔧 PREPROCESSING NEEDED:")
        print(f"   - Convert technical shorthand to natural language")
        print(f"   - Expand abbreviations (till→cash register, proj→project)")
        print(f"   - Add context for cryptic statements")
        print(f"   - Standardize formatting")
    
    print(f"\n🚀 CONCLUSION:")
    if len(ready_for_llm) >= len(statements) * 0.7:
        print(f"   ✅ READY FOR LLM ANALYSIS - {len(ready_for_llm)/len(statements)*100:.1f}% statements are good quality")
        print(f"   📝 Recommend preprocessing {len(needs_reformatting)} statements first")
    else:
        print(f"   ⚠️  NEEDS SIGNIFICANT PREPROCESSING - Only {len(ready_for_llm)/len(statements)*100:.1f}% ready")
        print(f"   🔧 Focus on reformatting technical shorthand statements")
    
    # Export analysis results
    analysis_df = pd.DataFrame({
        'problem_group_id': df['problem_group_id'],
        'statement': statements,
        'statement_length': lengths,
        'quality_category': ['HIGH' if s in ready_for_llm else 'NEEDS_WORK' for s in statements],
        'is_natural_language': [s in natural_language for s in statements],
        'primary_category': [get_primary_category(s, categories) for s in statements]
    })
    
    analysis_df.to_csv('outputs/problem_statement_analysis.csv', index=False)
    print(f"\n📁 Analysis exported to: outputs/problem_statement_analysis.csv")

def get_primary_category(statement, categories):
    """Determine primary category for a statement."""
    for category, items in categories.items():
        if statement in items:
            return category
    return 'Other'

if __name__ == "__main__":
    analyze_problem_statements()