#nlp_processor.py
import openai

import jieba
import re
from collections import Counter
import logging
import requests
import json
from config import Config
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load Chinese stopwords (common words to exclude)
STOPWORDS_PATH = os.path.join(os.path.dirname(__file__), 'chinese_stopwords.txt')


def translate_to_english(chinese_text):
    """
    Translate Chinese text to English using ChatGPT API.
    
    Args:
        chinese_text (str): Chinese text to translate
        
    Returns:
        str: Translated English text
    """
    api_key = Config.OPENAI_API_KEY
    if not api_key or not chinese_text:
        logger.warning("No API key available or empty text. Cannot translate.")
        return None
    
    try:
        # Try to import openai safely
        try:
            import openai
        except ImportError:
            logger.warning("OpenAI module not installed. Cannot translate.")
            return None
            
        client = openai.OpenAI(api_key=api_key)
        
        # Create a prompt for translation
        prompt = f"""
        Translate the following Chinese text to English:
        
        {chinese_text}
        
        Return only the translated text.
        """
        
        # Log the prompt for debugging
        logger.info(f"PROMPT [translate_to_english]: {prompt[:100]}...")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional translator from Chinese to English."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        # Process the response
        result_text = response.choices[0].message.content.strip()
        logger.info(f"TRANSLATION [result]: {result_text[:100]}...")
        
        return result_text
    
    except Exception as e:
        logger.error(f"Error translating text: {str(e)}")
        return None

def translate_keywords(keywords, max_words=20):
    """
    Translate a list of Chinese keywords to English.
    
    Args:
        keywords (list): List of Chinese keywords or word frequency tuples
        max_words (int): Maximum number of words to translate
        
    Returns:
        dict: Dictionary mapping English words to their frequencies
    """
    api_key = Config.OPENAI_API_KEY
    if not api_key:
        logger.warning("No API key available. Cannot translate keywords.")
        return {}
    
    try:
        # Prepare the list of words to translate
        if isinstance(keywords, list) and len(keywords) > 0:
            if isinstance(keywords[0], tuple):
                # Word frequency list [(word, count), ...]
                words_to_translate = [word for word, count in keywords[:max_words]]
                counts = {word: count for word, count in keywords[:max_words]}
            else:
                # Simple list of words
                words_to_translate = keywords[:max_words]
                counts = {word: 10 for word in words_to_translate}  # Assign equal weight
        else:
            logger.warning(f"Invalid keywords format: {type(keywords)}")
            return {}
        
        if not words_to_translate:
            return {}
            
        # Translate keywords
        word_list = ", ".join(words_to_translate)
        
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        prompt = f"""
        Translate the following Chinese keywords to English.
        Return a JSON object with Chinese words as keys and English translations as values.
        
        Keywords: {word_list}
        
        Format:
        {{
          "中文词1": "English word 1",
          "中文词2": "English word 2",
          ...
        }}
        """
        
        logger.info(f"PROMPT [translate_keywords]: {prompt}")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a translator specialized in Chinese to English translation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        result_text = response.choices[0].message.content.strip()
        logger.info(f"RESPONSE [translate_keywords]: {result_text}")
        
        # Extract JSON from response
        import json
        import re
        
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            translations = json.loads(json_match.group(0))
            
            # Create dictionary with English words and original counts
            english_word_freq = {}
            for zh_word, en_word in translations.items():
                if zh_word in counts:
                    english_word_freq[en_word] = counts[zh_word]
            
            return english_word_freq
        else:
            logger.warning("Failed to parse translation JSON")
            return {}
    
    except Exception as e:
        logger.error(f"Error translating keywords: {str(e)}")
        return {}
        

def load_stopwords():
    """Load Chinese stopwords from file or use default list."""
    try:
        if os.path.exists(STOPWORDS_PATH):
            with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
                return set([line.strip() for line in f])
        else:
            # Default minimal stopwords if file not found
            return set(['的', '了', '和', '是', '在', '我', '有', '不', '这', '也', '就', '人', '都',
                      '一', '一个', '上', '中', '下', '而', '到', '地', '要'])
    except Exception as e:
        logger.error(f"Error loading stopwords: {str(e)}")
        return set(['的', '了', '和', '是', '在', '我', '有', '不'])

# Load stopwords
STOPWORDS = load_stopwords()

def process_chinese_text(text):
    # ... existing code for segmentation, filtering, etc.
    # Remove English header and extract the Chinese content:
    chinese_content = extract_chinese_content(text)

    # [Existing segmentation, stopword filtering, frequency count, etc.]
    segmented = jieba.cut(chinese_content.strip())
    words = [word for word in segmented if word not in STOPWORDS and len(word.strip()) > 1 and not re.match(r'^\d+$', word)]
    word_counts = Counter(words)
    word_freq = word_counts.most_common(100)

    dimensions = analyze_dimensions(words, chinese_content)
    keywords = extract_key_themes(chinese_content) or []

    # NEW: Extract emotional keywords using AI
    emotional_keywords = extract_emotional_keywords(chinese_content)

    return {
        'segmented_text': words,
        'word_frequency': word_freq,
        'dimensions': dimensions,
        'keywords': keywords,
        'emotional_keywords': emotional_keywords
    }


def extract_mood_keywords(text):
    """
    Extract English mood keywords that might appear at the beginning of an entry.
    
    Args:
        text (str): Full journal entry text
        
    Returns:
        list: Extracted mood keywords
    """
    # Skip empty text
    if not text or not isinstance(text, str):
        return []

    # Look for English text at the beginning followed by Chinese characters
    english_pattern = r'^([A-Za-z,\s]+)[\n\r]+'
    match = re.search(english_pattern, text)

    if match:
        # Extract the matched English text
        english_text = match.group(1).strip()

        # Split by commas and clean up
        keywords = [k.strip() for k in english_text.split(',') if k.strip()]

        # Remove "and more" or similar phrases
        filtered_keywords = [k for k in keywords if 'more' not in k.lower() and len(k) > 1]

        return filtered_keywords

    return []

def extract_chinese_content(text):
    """
    Extract the main Chinese content, removing English headers.
    
    Args:
        text (str): Full journal entry text
        
    Returns:
        str: Chinese content portion of the text
    """
    # Skip empty text
    if not text or not isinstance(text, str):
        return ""

    # Remove English lines at the beginning
    lines = text.strip().split('\n')
    chinese_lines = []
    found_chinese = False

    for line in lines:
        # Check if the line contains Chinese characters
        if re.search(r'[\u4e00-\u9fff]', line):
            found_chinese = True
            chinese_lines.append(line)
        elif found_chinese:
            # If we've already found Chinese text, include remaining lines
            chinese_lines.append(line)
        # Skip English lines before any Chinese is found

    return '\n'.join(chinese_lines)

def analyze_dimensions(words, full_text=None):
    """
    Analyze the text and categorize it into different self-care dimensions.
    Uses a simple keyword-based approach by default.
    With API access, this could be replaced with a more sophisticated ML approach.
    
    Args:
        words (list): List of segmented words
        full_text (str): Original text for context (used with API)
        
    Returns:
        dict: Scores for each dimension
    """
    # Initialize dimensions with zero scores
    dimensions = {dim: 0 for dim in Config.DIMENSIONS}

    # Use API-based analysis if available
    if Config.OPENAI_API_KEY and full_text:
        try:
            api_dimensions = analyze_dimensions_with_api(full_text)
            if api_dimensions:
                return api_dimensions
        except Exception as e:
            logger.error(f"API-based dimension analysis failed: {str(e)}")
            # Fall back to keyword-based method

    # Simple keyword-based analysis
    word_set = set(words)

    # Count keywords for each dimension
    for dim, keywords in Config.DIMENSION_KEYWORDS_ZH.items():
        # Count matches
        matches = sum(1 for keyword in keywords if keyword in word_set)

        # Also check for partial matches (e.g., if the keyword is part of a longer word)
        partial_matches = sum(1 for w in words for keyword in keywords
                             if keyword in w and keyword != w)

        # Combine scores with partial matches having less weight
        dimensions[dim] = matches + (partial_matches * 0.5)

    # Normalize to 0-10 scale
    max_score = max(dimensions.values()) if any(dimensions.values()) else 1
    for dim in dimensions:
        dimensions[dim] = round(min(10, (dimensions[dim] / max_score) * 10), 1)

    return dimensions

def analyze_dimensions_with_api(text):
    """
    Use the ChatGPT API to analyze dimensions in the text.
    
    Args:
        text (str): Full text content
        
    Returns:
        dict: Scores for each dimension or None if API call fails
    """
    api_key = Config.OPENAI_API_KEY
    if not api_key:
        return None

    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=Config.OPENAI_API_KEY)

        prompt = f"""
Analyze the following journal entry and rate it on a scale of 0-10 for each of these wellbeing dimensions:
- Physical: related to body, health, exercise, sleep, nutrition
- Psychological: related to mind, thoughts, mental health, cognition
- Emotional: related to feelings, emotions, mood
- Spiritual: related to meaning, purpose, meditation, values
- Relational: related to relationships, social connections, family, friends
- Professional: related to work, career, achievements, studies

Journal entry (in Chinese):
{text[:1000]}... [truncated for length]

Return only a JSON object with the dimension scores, like:
{{
    "physical": 5,
    "psychological": 7,
    "emotional": 8,
    "spiritual": 3,
    "relational": 6,
    "professional": 9
}}
"""
        logger.info(f"PROMPT [analyze_dimensions_with_api]: {prompt[:200]}")

        response = client.chat.completions.create(model="gpt-4o",  # or "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a data analyst who outputs strictly JSON."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        temperature=0.2)

        result_text = response.choices[0].message.content.strip()
        logger.info(f"RESPONSE [analyze_dimensions_with_api]: {result_text}")

        try:
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                dimensions = json.loads(json_str)
                # Validate dimensions
                for dim in Config.DIMENSIONS:
                    if dim not in dimensions:
                        dimensions[dim] = 0
                    else:
                        try:
                            dimensions[dim] = min(10, max(0, float(dimensions[dim])))
                        except (ValueError, TypeError):
                            dimensions[dim] = 0
                return dimensions
        except Exception as e:
            logger.error(f"Error parsing API response: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"API analysis error: {str(e)}")
        return None

def extract_key_themes(text):
    """
    Extract key themes from the journal entry using ChatGPT.
    
    Args:
        text (str): Text content to analyze
        
    Returns:
        list: Key themes/keywords extracted from the text
    """
    api_key = Config.OPENAI_API_KEY
    if not api_key or not text or len(text) < 10:
        return None

    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=Config.OPENAI_API_KEY)

        prompt = f"""
Extract 5-10 key themes or important words from this Chinese journal entry.
Focus on meaningful content words (nouns, verbs, adjectives) that represent the main topics.
Return only a comma-separated list of words in Chinese, with no extra text.
        
Journal entry:
{text[:1000]}
        
Key themes (in Chinese, comma-separated):
"""
        logger.info(f"PROMPT [extract_key_themes]: {prompt[:200]}")

        response = client.chat.completions.create(model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in extracting key themes from text. Please return only a comma-separated list of words."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.3)

        result_text = response.choices[0].message.content.strip()
        logger.info(f"RESPONSE [extract_key_themes]: {result_text}")

        keywords = [k.strip() for k in result_text.split(',') if k.strip()]
        return keywords

    except Exception as e:
        logger.error(f"Error extracting key themes: {str(e)}")
        return None


def generate_journal_insights(entries, use_gpt4o=False):
    """
    Generate overall insights from all journal entries using ChatGPT.
    
    Args:
        entries (list): List of journal entry dictionaries
        
    Returns:
        dict: Analysis results with insights and recommendations
    """
    api_key = Config.OPENAI_API_KEY
    if not api_key or not entries or len(entries) == 0:
        return {
            "summary": "Not enough data to generate insights.",
            "patterns": [],
            "recommendations": [],
            "wellbeing_assessment": "Insufficient data for assessment."
        }

    try:
        # Try to import openai safely
        try:
            from openai import OpenAI

            client = OpenAI()
        except ImportError:
            logger.warning("OpenAI module not installed. Cannot generate journal insights.")
            return {
                "summary": "OpenAI API not available for analysis.",
                "patterns": [],
                "recommendations": [],
                "wellbeing_assessment": "API not available for assessment."
            }

        client = openai.OpenAI(api_key=api_key)

        # Prepare data for analysis
        entry_summaries = []
        for entry in entries:
            date = entry.get('date', '')
            text_preview = entry.get('text', '')[:200]
            sentiment = entry.get('sentiment', {}).get('dominant', 'neutral')

            # Get top dimensions
            dimensions = entry.get('dimensions', {})
            top_dims = sorted(dimensions.items(), key=lambda x: x[1], reverse=True)[:2]
            top_dims_str = ', '.join([f"{dim}: {score}" for dim, score in top_dims])

            entry_summaries.append(f"Date: {date}\nSentiment: {sentiment}\nTop dimensions: {top_dims_str}\nPreview: {text_preview}")

        # Join with separator
        entries_text = "\n\n---\n\n".join(entry_summaries)

        # Create prompt for analysis
        prompt = f"""
        You are a wellbeing coach analyzing journal entries. Review these journal entry summaries and provide:
        
        1. A summary of overall patterns and trends in wellbeing
        2. 3-5 specific patterns noticed across entries
        3. 3-5 personalized recommendations based on the journal content
        4. A brief wellbeing assessment
        
        Journal entries:
        {entries_text}
        
        Format your response as JSON with these fields:
        {{
            "summary": "Overall analysis of the journal entries",
            "patterns": ["Pattern 1", "Pattern 2", "Pattern 3"],
            "recommendations": ["Recommendation 1", "Recommendation 2", "Recommendation 3"],
            "wellbeing_assessment": "Brief wellbeing assessment"
        }}
        """

        # Log the prompt (truncated for readability)
        logger.info(f"PROMPT [generate_journal_insights]: {prompt[:500]}... [truncated]")

        # Choose model based on the flag
        model_name = "gpt-4o" if use_gpt4o else "gpt-3.5-turbo"

        response = client.chat.completions.create(
            model=model_name,  # Use the selected model
            messages=[
                {"role": "system", "content": "You are a wellbeing coach analyzing journal entries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )

        # Process the response
        result_text = response.choices[0].message.content.strip()
        logger.info(f"RESPONSE [generate_journal_insights]: {result_text}")

        try:
            # Extract JSON from the response
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                insights = json.loads(json_str)

                # Validate basic structure
                if "summary" not in insights:
                    insights["summary"] = "Analysis completed."
                if "patterns" not in insights:
                    insights["patterns"] = []
                if "recommendations" not in insights:
                    insights["recommendations"] = []
                if "wellbeing_assessment" not in insights:
                    insights["wellbeing_assessment"] = "Assessment completed."

                return insights
        except Exception as e:
            logger.error(f"Error parsing API insights response: {str(e)}")

        # Fallback if parsing fails
        return {
            "summary": "Analysis completed but results could not be formatted properly.",
            "patterns": [],
            "recommendations": [],
            "wellbeing_assessment": "Unable to generate detailed assessment."
        }

    except Exception as e:
        logger.error(f"Error generating journal insights: {str(e)}")
        return {
            "summary": f"Error during analysis: {str(e)}",
            "patterns": [],
            "recommendations": [],
            "wellbeing_assessment": "Error during assessment."
        }


def extract_emotional_keywords(text):
    """
    Use ChatGPT API to extract 5-10 emotional keywords from the given Chinese text.
    """
    if not Config.OPENAI_API_KEY or not text or len(text) < 10:
        return []
    try:
        prompt = f"""
        请从以下中文日记内容中提取 **5 到 10 个** 能准确反映作者 **情绪状态** 的关键词。  

        - **关键词应包括：**  
          - **情感、心理状态**（如“焦虑”、“满足”）。  
          - **导致情绪变化的具体因素**（如“失败”、“争吵”）。  
        - **避免提取：**  
          - 时间相关词语（如“周一”、“早晨”, "晚安“，“睡觉”）。  
          - 日常活动（如“吃饭”、“起床”）。  
          - 常见环境描述（如“天气”、“路上”）。  
          - 普通人物称呼（如“朋友”、“老师”），但如果其与情绪强相关（如“被朋友背叛”），可以提取。  
          - 无明显情感指向的抽象词（如“事情”、“情况”）。  
          - “日记”
        - **仅返回** 以 **逗号** 分隔的关键词列表，**不包含** 任何额外文字或解释。

日记内容:
{text[:1000]}
        """
        response = client.chat.completions.create(model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一位中文情绪分析专家。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000)
        result_text = response.choices[0].message.content.strip()
        keywords = [word.strip() for word in result_text.split(',') if word.strip()]
        return keywords
    except Exception as e:
        logger.error(f"Error extracting emotional keywords: {str(e)}")
        return []

