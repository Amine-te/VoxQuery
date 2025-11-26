"""
Natural Language Response Generator
Converts SQL query results into natural language responses
"""

import time
import requests
import logging
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


class NLResponseGenerator:
    """
    Generates natural language responses from database query results
    """
    
    def __init__(
        self,
        ollama_model: str = "llama3:8b",
        ollama_url: str = "http://localhost:11434/api/generate",
        max_results: int = 10,
        timeout: int = 120,
        verbose: bool = False
    ):
        """
        Initialize NL Response Generator
        
        Args:
            ollama_model: Name of Ollama model to use
            ollama_url: URL of Ollama API endpoint
            max_results: Maximum number of results to include in context
            timeout: Request timeout in seconds
            verbose: Enable verbose logging
        """
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.max_results = max_results
        self.timeout = timeout
        self.verbose = verbose
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of input text
        
        Args:
            text: Input text to analyze
            
        Returns:
            ISO 639-1 language code (e.g., 'en', 'fr', 'ar')
        """
        try:
            from langdetect import detect
            lang = detect(text)
            lang_map = {
                'fr': 'fr',
                'en': 'en',
                'ar': 'ar',
                'es': 'es'
            }
            return lang_map.get(lang, 'en')
        except Exception as e:
            if self.verbose:
                log.warning(f"Language detection failed: {e}. Defaulting to English.")
            return 'en'
    
    def build_prompt(
        self,
        user_query: str,
        db_results: List[Dict],
        language: Optional[str] = None
    ) -> str:
        """
        Build LLM prompt for natural language response generation
        
        Args:
            user_query: Original user question
            db_results: Database query results
            language: Language code (auto-detected if None)
            
        Returns:
            Complete prompt string
        """
        # Auto-detect language if not provided
        if language is None:
            language = self.detect_language(user_query)
        
        # Limit to max_results
        limited_results = db_results[:self.max_results]
        has_more = len(db_results) > self.max_results
        total_count = len(db_results)
        
        # Format database results
        if not limited_results or len(limited_results) == 0:
            results_text = "(no data found)"
        else:
            results_text = ""
            for idx, row in enumerate(limited_results, 1):
                parts = [f"{k}: {v}" for k, v in row.items() if v is not None]
                results_text += f"  - {' | '.join(parts)}\n"
            
            # Add note about limited results
            if has_more:
                results_text += f"\n  (Showing first {self.max_results} of {total_count} total results)\n"
        
        # Language-specific system prompts
        if language == 'fr':
            system_prompt = f"""Tu es un assistant virtuel qui répond aux questions sur les données.

RÈGLES STRICTES:
- Utilise UNIQUEMENT les données fournies
- Si l'information n'existe pas, réponds: "Désolé, aucune information disponible."
- Réponds en français clair et naturel
- Format texte brut, sans Markdown (pas de **, #, *, etc.)
- Le texte sera lu à haute voix - chaque phrase doit être courte et naturelle
- Si plusieurs résultats, utilise des virgules ou "et" pour séparer plutôt que des points
- Sois concis, professionnel et utile
- Si les résultats sont limités aux premiers {self.max_results}, MENTIONNE clairement "Voici les {self.max_results} premiers résultats sur {total_count} au total" dans ta réponse"""
        elif language == 'ar':
            system_prompt = f"""أنت مساعد افتراضي يجيب على الأسئلة حول البيانات.

القواعد الصارمة:
- استخدم البيانات المقدمة فقط
- إذا لم تكن المعلومات موجودة، أجب: "عذراً، لا توجد معلومات متاحة."
- الرد بالعربية الواضحة والطبيعية
- نص عادي، بدون Markdown
- سيتم قراءة النص بصوت عالٍ - اجعل الجمل قصيرة وطبيعية
- كن موجزاً ومحترفاً ومفيداً
- إذا كانت النتائج محدودة بأول {self.max_results}، اذكر بوضوح "هذه أول {self.max_results} نتائج من أصل {total_count}" في ردك"""
        else:  # English
            system_prompt = f"""You are a virtual assistant that answers questions about data.

STRICT RULES:
- Use ONLY the provided data
- If information doesn't exist, respond: "Sorry, no information available."
- Respond in clear, natural English
- Plain text format, no Markdown (no **, #, *, etc.)
- Text will be read aloud - keep sentences short and natural
- For multiple results, use commas or "and" instead of periods to separate
- Be concise, professional, and helpful
- If results are limited to first {self.max_results}, CLEARLY MENTION "Here are the first {self.max_results} results out of {total_count} total" in your response"""
        
        prompt = f"""{system_prompt}

USER QUESTION: {user_query}

DATABASE RESULTS:
{results_text}

Based on this data, provide a clear and helpful answer."""
        
        return prompt
    
    def generate_response(
        self,
        user_query: str,
        db_results: List[Dict],
        language: Optional[str] = None
    ) -> Dict:
        """
        Generate natural language response from database results
        
        Args:
            user_query: Original user question
            db_results: Database query results
            language: Language code (auto-detected if None)
            
        Returns:
            Dict with response, timing, and metadata
        """
        start_time = time.time()
        
        # Auto-detect language if not provided
        if language is None:
            language = self.detect_language(user_query)
        
        # Build prompt
        prompt = self.build_prompt(user_query, db_results, language)
        
        # Prepare API payload
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 512
            }
        }
        
        try:
            # Call Ollama API
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            elapsed = time.time() - start_time
            nl_response = result['response'].strip()
            
            if self.verbose:
                log.info(f"Generated response in {elapsed:.2f}s")
            
            return {
                "success": True,
                "response": nl_response,
                "language": language,
                "generation_time": elapsed,
                "prompt": prompt,
                "results_count": len(db_results),
                "results_shown": min(len(db_results), self.max_results)
            }
            
        except requests.exceptions.Timeout as e:
            elapsed = time.time() - start_time
            error_msg = f"Request timed out after {self.timeout}s"
            log.error(error_msg)
            
            return {
                "success": False,
                "response": "",
                "language": language,
                "generation_time": elapsed,
                "error": error_msg,
                "prompt": prompt
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e)
            log.error(f"Generation failed: {error_msg}")
            
            return {
                "success": False,
                "response": "",
                "language": language,
                "generation_time": elapsed,
                "error": error_msg,
                "prompt": prompt
            }
    
    def generate_error_response(
        self,
        user_query: str,
        error_message: str,
        language: Optional[str] = None
    ) -> Dict:
        """
        Generate a user-friendly error response
        
        Args:
            user_query: Original user question
            error_message: Technical error message
            language: Language code (auto-detected if None)
            
        Returns:
            Dict with error response
        """
        if language is None:
            language = self.detect_language(user_query)
        
        # Language-specific error messages
        error_responses = {
            'fr': "Désolé, je n'ai pas pu traiter votre demande. Veuillez réessayer.",
            'ar': "عذراً، لم أتمكن من معالجة طلبك. يرجى المحاولة مرة أخرى.",
            'es': "Lo siento, no pude procesar tu solicitud. Por favor, inténtalo de nuevo.",
            'en': "Sorry, I couldn't process your request. Please try again."
        }
        
        response = error_responses.get(language, error_responses['en'])
        
        return {
            "success": True,
            "response": response,
            "language": language,
            "generation_time": 0.0,
            "error_context": error_message
        }


def create_nl_generator(
    ollama_model: str = "llama3:8b",
    max_results: int = 10,
    timeout: int = 120,
    **kwargs
) -> NLResponseGenerator:
    """
    Create and initialize a Natural Language Response Generator
    
    Args:
        ollama_model: Ollama model name
        max_results: Maximum results to include in context
        timeout: Request timeout in seconds
        **kwargs: Additional arguments for NLResponseGenerator
        
    Returns:
        Initialized NLResponseGenerator instance
    """
    return NLResponseGenerator(
        ollama_model=ollama_model,
        max_results=max_results,
        timeout=timeout,
        **kwargs
    )