import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration settings for the application."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    # API keys
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    
    # File upload settings
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    ANALYSIS_FOLDER = os.path.join(os.getcwd(), 'analyses')
    ALLOWED_EXTENSIONS = {'pdf', 'txt', 'html', 'htm'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
    
    # NLP settings
    DEFAULT_LANGUAGE = 'zh'  # Chinese
    
    # Visualization settings
    WORD_CLOUD_MAX_WORDS = 100
    WORD_CLOUD_WIDTH = 800
    WORD_CLOUD_HEIGHT = 400
    
    # Self-care dimensions
    DIMENSIONS = [
        'physical',
        'psychological',
        'emotional',
        'spiritual',
        'relational',
        'professional'
    ]
    
    # Dimension keywords (English examples - will need Chinese equivalents)
    DIMENSION_KEYWORDS = {
        'physical': ['exercise', 'sleep', 'food', 'nutrition', 'health', 'body', 'rest', 'walk'],
        'psychological': ['mind', 'thoughts', 'thinking', 'mental', 'focus', 'concentration', 'clarity'],
        'emotional': ['feel', 'feeling', 'emotion', 'happy', 'sad', 'angry', 'anxious', 'joy'],
        'spiritual': ['meaning', 'purpose', 'meditation', 'prayer', 'belief', 'faith', 'soul'],
        'relational': ['friend', 'family', 'partner', 'relationship', 'conversation', 'social', 'people'],
        'professional': ['work', 'job', 'career', 'project', 'meeting', 'client', 'task', 'deadline']
    }
    
    # Chinese dimension keywords
    DIMENSION_KEYWORDS_ZH = {
        'physical': ['运动', '锻炼', '睡眠', '饮食', '营养', '健康', '身体', '休息', '步行', '疲劳'],
        'psychological': ['思考', '思想', '精神', '专注', '注意力', '清晰', '心理', '意识', '认知'],
        'emotional': ['感觉', '情绪', '开心', '悲伤', '愤怒', '焦虑', '喜悦', '恐惧', '担心', '压力'],
        'spiritual': ['意义', '目的', '冥想', '祈祷', '信仰', '灵魂', '内心', '和平', '平静'],
        'relational': ['朋友', '家人', '伴侣', '关系', '交流', '社交', '人际', '沟通', '亲密'],
        'professional': ['工作', '职业', '事业', '项目', '会议', '客户', '任务', '截止日期', '学习', '进步']
    }
