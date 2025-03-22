#sentiment.py
import logging
import json
import os
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths for sentiment dictionaries
SENTIMENT_PATH = os.path.join(os.path.dirname(__file__), 'chinese_sentiment_dict.json')

def load_sentiment_dict():
    """Load the Chinese sentiment dictionary or use a minimal default."""
    try:
        if os.path.exists(SENTIMENT_PATH):
            with open(SENTIMENT_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Return a minimal dictionary if file doesn't exist
            return {
                'positive': ['开心', '快乐', '高兴', '满意', '喜欢', '爱', '成功', '好', '积极', '希望'],
                'negative': ['悲伤', '难过', '痛苦', '失望', '害怕', '担心', '焦虑', '生气', '愤怒', '压力'],
                'neutral': ['觉得', '认为', '想', '看', '知道', '感觉', '理解', '思考', '考虑', '思想']
            }
    except Exception as e:
        logger.error(f"Error loading sentiment dictionary: {str(e)}")
        return {'positive': [], 'negative': [], 'neutral': []}

# Load sentiment dictionaries
SENTIMENT_DICT = load_sentiment_dict()

def analyze_sentiment(words, api_enabled=True):
    """
    Analyze sentiment of Chinese text.
    Args:
        words (list): List of segmented Chinese words
        api_enabled (bool): Whether to try using ChatGPT API for sentiment analysis
    Returns:
        dict: Sentiment scores and dominant sentiment
    """
    # Join words into a string for API analysis.
    text = ' '.join(words)
    if api_enabled and Config.OPENAI_API_KEY and isinstance(words, list) and len(words) > 5:
        api_result = analyze_sentiment_with_api(text)
        if api_result:
            return api_result

    # Fall back to dictionary-based method if API fails or isn't available.
    return dictionary_based_sentiment(words)

def dictionary_based_sentiment(words):
    """
    Analyze sentiment using a dictionary-based approach.
    
    Args:
        words (list): List of segmented Chinese words
        
    Returns:
        dict: Sentiment scores and dominant sentiment
    """
    if not isinstance(words, list):
        logger.warning(f"Expected list of words, got {type(words)}")
        words = []

    # Initialize sentiment counters
    sentiment_counts = {
        'positive': 0,
        'negative': 0,
        'neutral': 0
    }

    # Count sentiment words
    word_set = set(words)
    for sentiment, keywords in SENTIMENT_DICT.items():
        sentiment_counts[sentiment] = sum(1 for keyword in keywords if keyword in word_set)

    # Calculate scores (percentage of each sentiment)
    total = sum(sentiment_counts.values()) or 1  # Avoid division by zero
    sentiment_scores = {
        'positive': round((sentiment_counts['positive'] / total) * 100, 1),
        'negative': round((sentiment_counts['negative'] / total) * 100, 1),
        'neutral': round((sentiment_counts['neutral'] / total) * 100, 1)
    }

    # Determine dominant sentiment
    dominant = max(sentiment_scores.items(), key=lambda x: x[1])[0]

    # If all scores are zero, set as neutral
    if sentiment_scores['positive'] == 0 and sentiment_scores['negative'] == 0:
        dominant = 'neutral'
        sentiment_scores['neutral'] = 100

    return {
        'scores': sentiment_scores,
        'dominant': dominant
    }

def analyze_sentiment_with_api(text):
    """
    Use the ChatGPT API to analyze the sentiment of the given Chinese text.
    Returns a dict with sentiment percentages for 'positive', 'neutral', and 'negative'
    that sum to 100, along with the dominant sentiment.
    """
    api_key = Config.OPENAI_API_KEY
    if not api_key or not text:
        return None
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        prompt = (
            "请分析以下中文文本的情感，并返回一个JSON对象，包含'positive'、'neutral'和'negative'三个键，其值分别为文本中正面、中性和负面情绪的百分比，并确保这三个数字之和为100。例如，如果文本主要表达正面情绪，则可能返回："
            '{"positive": 70.0, "neutral": 20.0, "negative": 10.0}。 \n'
            "文本内容：\n" + text
        )
        response = client.chat.completions.create(model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一位情感分析专家。请严格按照要求返回JSON格式数据，不要包含额外文字。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=3000)
        result_text = response.choices[0].message.content.strip()
        result_text = result_text.strip("`")  # Remove any stray backticks
        # Use the helper to remove markdown code fences if present.
        def extract_json_local(text):
            text = text.strip()
            if text.startswith("```"):
                lines = text.splitlines()
                if len(lines) >= 2:
                    text = "\n".join(lines[1:-1]).strip()
            return text
        result_text = extract_json_local(result_text)
        if not result_text:
            return None
        sentiment = json.loads(result_text)
        dominant = max(sentiment, key=sentiment.get)
        return {
            'scores': sentiment,
            'dominant': dominant
        }
    except Exception as e:
        logger.error(f"Error in analyze_sentiment_with_api: {str(e)}")
        return None


#def analyze_sentiment_with_api(text):
#    """
#    Use the ChatGPT API to analyze sentiment.
#    
#    Args:
#        text (str): Text to analyze
#        
#    Returns:
#        dict: Sentiment analysis results or None if API call fails
#    """
#    api_key = Config.OPENAI_API_KEY
#    if not api_key or not text:
#        return None
#    
#    try:
#        # Try to import openai safely
#        try:
#            import openai
#        except ImportError:
#            logger.warning("OpenAI module not installed. Using basic sentiment analysis instead.")
#            return None
#            
#        client = openai.OpenAI(api_key=api_key)
#        
#        # Create a prompt for sentiment analysis
#        prompt = f"""
#        Analyze the sentiment of the following Chinese text and categorize it as positive, negative, or neutral.
#        Also provide scores as percentages for each category (should sum to 100%).
#        
#        Text:
#        {text}
#        
#        Return only a JSON object like:
#        {{
#            "scores": {{
#                "positive": 70.5,
#                "negative": 10.2,
#                "neutral": 19.3
#            }},
#            "dominant": "positive"
#        }}
#        """
#        
#        response = client.completions.create(
#            model="gpt-3.5-turbo-instruct",
#            prompt=prompt,
#            max_tokens=150,
#            temperature=0.2
#        )
#        
#        # Parse the response
#        result_text = response.choices[0].text.strip()
#        try:
#            # Extract JSON from the response
#            json_start = result_text.find('{')
#            json_end = result_text.rfind('}') + 1
#            if json_start >= 0 and json_end > json_start:
#                json_str = result_text[json_start:json_end]
#                sentiment = json.loads(json_str)
#                
#                # Validate structure
#                if 'scores' in sentiment and 'dominant' in sentiment:
#                    if all(k in sentiment['scores'] for k in ['positive', 'negative', 'neutral']):
#                        return sentiment
#        except Exception as e:
#            logger.error(f"Error parsing API sentiment response: {str(e)}")
#        
#        return None
#    
#    except ImportError:
#        logger.warning("OpenAI module not installed. Using basic sentiment analysis instead.")
#        return None
#    except Exception as e:
#        logger.error(f"API sentiment analysis error: {str(e)}")
#        return None


