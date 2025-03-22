#init.py
"""
Utils package for Journal Analyzer.

This package includes utility modules for:
- PDF text extraction
- NLP processing of Chinese text
- Sentiment analysis
- Data visualization
"""

# Import main utility functions for easier access
from .pdf_extractor import extract_text_from_file, extract_date_from_text, split_journal_into_entries, extract_text_with_ocr
from .nlp_processor import (
    process_chinese_text,
    analyze_dimensions,
    extract_mood_keywords,
    extract_chinese_content,
    extract_key_themes,
    generate_journal_insights,
    translate_to_english,
    translate_keywords
)
from .sentiment import analyze_sentiment
from .visualizer import (
    generate_word_cloud,
    generate_sentiment_chart,
    generate_dimension_chart,
    generate_trend_analysis
)

__all__ = [
    'extract_text_from_file',
    'extract_date_from_text',
    'split_journal_into_entries',
    'extract_text_with_ocr',
    'process_chinese_text',
    'analyze_dimensions',
    'extract_mood_keywords',
    'extract_chinese_content',
    'extract_key_themes',
    'generate_journal_insights',
    'translate_to_english',
    'translate_keywords',
    'analyze_sentiment',
    'generate_word_cloud',
    'generate_sentiment_chart',
    'generate_dimension_chart',
    'generate_trend_analysis'
]

