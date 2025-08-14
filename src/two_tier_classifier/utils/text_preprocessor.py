"""
Enhanced text preprocessing for ticket classification.

Provides semantic preprocessing optimized for business ticket text,
including normalization, variable extraction, and noise reduction.
"""

import re
import string
from typing import Dict, List, Optional, Set, Tuple
import logging

class TextPreprocessor:
    """Advanced text preprocessing for ticket classification."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common IT/business stop words specific to tickets
        self.custom_stop_words = {
            'please', 'thanks', 'thank', 'you', 'regards', 'hi', 'hello',
            'could', 'would', 'should', 'may', 'might', 'can', 'will',
            'wickes', 'service', 'desk', 'support', 'team', 'it', 'help',
            'issue', 'problem', 'incident', 'ticket', 'request',
            'need', 'want', 'require', 'looking', 'trying'
        }
        
        # Variable patterns to normalize
        self.variable_patterns = {
            'store_numbers': r'\b(?:store\s*)?(?:number\s*)?(\d{4})\b',
            'employee_ids': r'\b[a-z]{2,5}\d{2,4}\b',
            'project_numbers': r'\bproject\s*(\d+)\b',
            'till_numbers': r'\btill\s*(\d+)\b',
            'order_numbers': r'\border\s*(?:number\s*)?(\d+)\b',
            'sku_numbers': r'\bsku\s*(?:number\s*)?(\d+)\b',
            'ticket_numbers': r'\b(?:inc|req|chg|prb)(\d+)\b',
            'error_codes': r'\berror\s*(?:code\s*)?(\d+)\b',
            'phone_numbers': r'[\+]?[0-9\s\-\(\)]{10,}',
            'email_addresses': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'ip_addresses': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'time_stamps': r'\b\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\b'
        }
        
        # Business-critical keywords that should never be removed
        self.protected_keywords = {
            'till', 'vision', 'print', 'printer', 'order', 'payment', 'chip', 'pin',
            'scanner', 'barcode', 'fusion', 'appstream', 'zebra', 'mobile',
            'account', 'password', 'login', 'email', 'google', 'gmail',
            'active directory', 'ad', 'locked', 'unlock', 'group', 'mfa', '2fa',
            'error', 'failed', 'crashed', 'urgent', 'critical', 'down', 'offline', 'not working'
        }
        
    def preprocess(self, text: str, preserve_numbers: bool = True) -> Dict[str, any]:
        """
        Complete preprocessing pipeline for ticket text.
        
        Args:
            text: Raw ticket description
            preserve_numbers: Whether to preserve specific numbers (store IDs, etc.)
            
        Returns:
            Dictionary containing processed text and extracted metadata
        """
        if not text or not isinstance(text, str):
            return {
                'processed_text': '',
                'original_text': text,
                'extracted_variables': {},
                'keywords_found': [],
                'processing_notes': ['empty_or_invalid_input']
            }
        
        result = {
            'original_text': text,
            'extracted_variables': {},
            'keywords_found': [],
            'processing_notes': []
        }
        
        # Step 1: Extract and normalize variables
        processed_text, variables = self._extract_variables(text, preserve_numbers)
        result['extracted_variables'] = variables
        
        # Step 2: Clean and normalize text
        processed_text = self._clean_text(processed_text)
        
        # Step 3: Extract business keywords
        keywords = self._extract_business_keywords(processed_text)
        result['keywords_found'] = keywords
        
        # Step 4: Remove noise while preserving meaning
        processed_text = self._remove_noise(processed_text)
        
        # Step 5: Final normalization
        processed_text = self._final_normalization(processed_text)
        
        result['processed_text'] = processed_text
        
        # Add processing quality notes
        if len(processed_text.split()) < 3:
            result['processing_notes'].append('very_short_output')
        if not keywords:
            result['processing_notes'].append('no_business_keywords_found')
        if variables:
            result['processing_notes'].append('variables_extracted')
            
        return result
    
    def _extract_variables(self, text: str, preserve_numbers: bool) -> Tuple[str, Dict]:
        """Extract and normalize variable information from text."""
        variables = {}
        processed_text = text.lower()
        
        for var_type, pattern in self.variable_patterns.items():
            matches = re.findall(pattern, processed_text, re.IGNORECASE)
            if matches:
                variables[var_type] = matches
                
                # Replace with normalized tokens if preserving structure
                if preserve_numbers:
                    if var_type == 'store_numbers':
                        processed_text = re.sub(pattern, f'[STORE_NUMBER]', processed_text, flags=re.IGNORECASE)
                    elif var_type == 'employee_ids':
                        processed_text = re.sub(pattern, f'[EMPLOYEE_ID]', processed_text, flags=re.IGNORECASE)
                    elif var_type == 'project_numbers':
                        processed_text = re.sub(pattern, f'project [PROJECT_NUM]', processed_text, flags=re.IGNORECASE)
                    elif var_type == 'till_numbers':
                        processed_text = re.sub(pattern, f'till [TILL_NUM]', processed_text, flags=re.IGNORECASE)
                else:
                    # Remove the variable entirely
                    processed_text = re.sub(pattern, '', processed_text, flags=re.IGNORECASE)
        
        return processed_text, variables
    
    def _clean_text(self, text: str) -> str:
        """Basic text cleaning and normalization."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML/XML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        
        # Remove email domains but keep the word "email"
        text = re.sub(r'@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}', ' ', text)
        
        # Normalize common contractions
        contractions = {
            "can't": "cannot",
            "won't": "will not", 
            "don't": "do not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "hasn't": "has not",
            "haven't": "have not",
            "hadn't": "had not"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _extract_business_keywords(self, text: str) -> List[str]:
        """Extract business-relevant keywords for classification."""
        keywords_found = []
        text_lower = text.lower()
        
        # Check for protected keywords and variations
        for keyword in self.protected_keywords:
            # Exact match
            if keyword in text_lower:
                keywords_found.append(keyword)
            
            # Check for variations (plural, etc.)
            if keyword.endswith('er') and keyword + 's' in text_lower:
                keywords_found.append(keyword + 's')
            elif not keyword.endswith('s') and keyword + 's' in text_lower:
                keywords_found.append(keyword + 's')
        
        # Look for compound business terms
        compound_terms = [
            'chip pin', 'chip and pin', 'active directory', 'unlock ad account', 'back office',
            'end user computing', 'vision order', 'till function',
            'payment device', 'mobile device', 'project machine'
        ]
        
        for term in compound_terms:
            if term in text_lower:
                keywords_found.append(term)
        
        return list(set(keywords_found))  # Remove duplicates
    
    def _remove_noise(self, text: str) -> str:
        """Remove noise while preserving business-critical information."""
        words = text.split()
        filtered_words = []
        
        for word in words:
            # Keep if it's a protected keyword
            if any(protected in word for protected in self.protected_keywords):
                filtered_words.append(word)
                continue
            
            # Keep if it's a normalized token
            if word.startswith('[') and word.endswith(']'):
                filtered_words.append(word)
                continue
            
            # Remove common stop words
            if word in self.custom_stop_words:
                continue
            
            # Remove very short words unless they're important
            if len(word) <= 2 and word not in ['pc', 'id', 'ad', 'it']:
                continue
            
            # Remove punctuation-only words
            if all(c in string.punctuation for c in word):
                continue
            
            # Keep the word
            filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def _final_normalization(self, text: str) -> str:
        """Final text normalization and cleanup."""
        # Remove extra punctuation
        text = re.sub(r'[^\w\s\[\]]', ' ', text)
        
        # Normalize specific business terms
        business_normalizations = {
            'printers': 'printer',
            'tills': 'till',
            'orders': 'order',
            'applications': 'application',
            'devices': 'device',
            'systems': 'system',
            'errors': 'error',
            'problems': 'problem',
            'issues': 'issue'
        }
        
        for plural, singular in business_normalizations.items():
            text = text.replace(plural, singular)
        
        # Final whitespace cleanup
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_urgency_indicators(self, text: str) -> Dict[str, any]:
        """Extract urgency and priority indicators from text."""
        urgency_indicators = {
            'critical': ['critical', 'urgent', 'emergency', 'asap', 'immediately'],
            'high': ['customers waiting', 'store down', 'cannot operate', 'business impact'],
            'time_sensitive': ['today', 'now', 'deadline', 'soon'],
            'operational_impact': ['down', 'offline', 'not working', 'failed', 'crashed']
        }
        
        text_lower = text.lower()
        found_indicators = {}
        urgency_score = 0.0
        
        for category, indicators in urgency_indicators.items():
            found = [ind for ind in indicators if ind in text_lower]
            if found:
                found_indicators[category] = found
                # Weight urgency score based on category
                if category == 'critical':
                    urgency_score += 0.4
                elif category == 'high':
                    urgency_score += 0.3
                elif category == 'time_sensitive':
                    urgency_score += 0.2
                elif category == 'operational_impact':
                    urgency_score += 0.1
        
        return {
            'urgency_score': min(1.0, urgency_score),
            'indicators_found': found_indicators,
            'has_urgency_keywords': len(found_indicators) > 0
        }
    
    def get_preprocessing_stats(self, results: List[Dict]) -> Dict:
        """Get statistics about preprocessing results for monitoring."""
        if not results:
            return {}
        
        total_processed = len(results)
        avg_input_length = sum(len(r['original_text']) for r in results) / total_processed
        avg_output_length = sum(len(r['processed_text']) for r in results) / total_processed
        
        # Count processing issues
        empty_outputs = sum(1 for r in results if len(r['processed_text'].strip()) == 0)
        no_keywords = sum(1 for r in results if len(r['keywords_found']) == 0)
        variables_extracted = sum(1 for r in results if r['extracted_variables'])
        
        return {
            'total_processed': total_processed,
            'avg_input_length': round(avg_input_length, 1),
            'avg_output_length': round(avg_output_length, 1),
            'compression_ratio': round(avg_output_length / avg_input_length, 2),
            'empty_outputs': empty_outputs,
            'no_keywords_found': no_keywords,
            'variables_extracted': variables_extracted,
            'quality_score': round(1.0 - (empty_outputs + no_keywords) / (total_processed * 2), 2)
        }