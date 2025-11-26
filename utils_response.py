"""
Utility functions for formatting database results into natural language.
"""

import logging
from typing import Dict, List

log = logging.getLogger(__name__)


def format_db_results_for_llm(results: List[Dict], max_rows: int = 50) -> str:
    """
    Convert database results to readable text format for LLM context.
    
    Args:
        results: List of dictionaries (rows from database)
        max_rows: Maximum number of rows to include
        
    Returns:
        Formatted string representation of results
    """
    if not results or len(results) == 0:
        return "(no data found)"
    
    rows_to_show = min(len(results), max_rows)
    lines = []
    
    for idx, row in enumerate(results[:rows_to_show]):
        parts = []
        for col, val in row.items():
            # Truncate long values
            if isinstance(val, str) and len(val) > 50:
                val = val[:47] + "..."
            parts.append(f"{col}: {val}")
        lines.append("  - " + " | ".join(parts))
    
    if len(results) > max_rows:
        lines.append(f"  ... ({len(results) - max_rows} more rows)")
    
    return "\n".join(lines)


def detect_language(text: str) -> str:
    """
    Detect language of input text.
    Returns ISO 639-1 code (e.g., 'en', 'fr', 'ar')
    """
    try:
        from langdetect import detect
        lang = detect(text)
        # Map common languages
        lang_map = {
            'fr': 'fr',  # French
            'en': 'en',  # English
            'ar': 'ar',  # Arabic
            'es': 'es',  # Spanish
        }
        return lang_map.get(lang, 'en')  # Default to English
    except Exception as e:
        log.warning(f"Language detection failed: {e}. Defaulting to English.")
        return 'en'