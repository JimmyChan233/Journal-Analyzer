# AI-Powered Journal Analysis Tool

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An advanced tool for analyzing Chinese/English journal entries, providing insights into wellbeing dimensions, emotional patterns, and generating visualizations with AI-powered analysis. This application helps users gain deeper understanding of their journaling practice through natural language processing and machine learning.

## Features

### Multi-Format Journal Processing
- **PDF Support**: Extract text with advanced OCR for scanned documents
- **TXT Support**: Process plain text with automatic encoding detection
- **HTML Support**: Parse journal entries from HTML files (naming: YYYY-MM-DD_title.html)
- **Multi-Entry Detection**: Automatically split journal files with multiple dated entries

### Chinese Text Analysis
- **Word Segmentation**: Process Chinese text using Jieba
- **Sentiment Analysis**: Identify positive, neutral, and negative emotional tones
- **Dimension Categorization**: Analyze content across six wellbeing dimensions:
  - Physical
  - Psychological
  - Emotional
  - Spiritual
  - Relational
  - Professional
- **Keyword Extraction**: Identify important themes and topics

### Translation & Visualization
- **Chinese-to-English Translation**: Convert keywords and themes for broader accessibility
- **Dual-Language Word Clouds**: Generate both Chinese and English visualizations
- **Summary Word Clouds**: Create combined visualizations that update with each new entry
- **Interactive Charts**: Track sentiment and dimension changes over time

### AI-Powered Insights
- **GPT-4o Integration**: Advanced analysis capabilities with OpenAI's latest models
- **Pattern Detection**: Identify trends across journal entries
- **Personalized Recommendations**: Receive tailored wellbeing suggestions
- **Comprehensive Assessment**: Get detailed analysis of journaling patterns

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Tesseract OCR (for PDF scanning capabilities)

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/journal-analyzer.git
cd journal-analyzer
```

### Step 2: Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set up environment variables
Create a `.env` file in the project root:
```
SECRET_KEY=your_random_secret_key_here
DEBUG=True
OPENAI_API_KEY=your_openai_api_key
```

You can generate a random secret key with:
```python
import secrets
print(secrets.token_hex(16))
```

### Step 5: Install Tesseract OCR (for PDF OCR capabilities)

**For macOS:**
```bash
brew install tesseract
brew install tesseract-lang  # For language support
```

**For Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-chi-sim  # For Chinese support
```

**For Windows:**
1. Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
2. Add Tesseract to your PATH environment variable

## Usage

### Starting the Application
```bash
python app.py
```
Access the application at: http://127.0.0.1:5000

### Uploading Journal Files
1. Navigate to the "Upload" page
2. Select one or more files (PDF, TXT, or HTML)
3. Supported formats:
   - Single-entry files (one journal per file)
   - Multi-entry files (entries separated by dates like "Monday, February 3, 2025")
   - HTML files named like "2025-02-03_第一天写日记.html"

### Viewing Analysis
1. After uploading, you'll be redirected to the dashboard
2. Navigate through different tabs:
   - **Overview**: Summary of sentiment, dimensions, and word frequency
   - **AI Insights**: GPT-4o powered analysis and recommendations
   - **Sentiment**: Detailed emotional analysis over time
   - **Dimensions**: Self-care dimension breakdown
   - **Trends**: Patterns and correlations
   - **Journal Entries**: List of processed entries

### Word Clouds
The application generates two types of word clouds:
- **Chinese Word Cloud**: Shows the original Chinese keywords
- **English Word Cloud**: Shows translated keywords for easier sharing
- **Summary Visualization**: Updates automatically as you add new entries

## Project Structure
```
journal_analyzer/
│
├── app.py               # Main Flask application
├── config.py            # Configuration settings
├── requirements.txt     # Project dependencies
│
├── static/              # Static files
│   ├── css/             # CSS files
│   │   └── styles.css   # Custom styles
│   ├── js/              # JavaScript files
│   │   └── dashboard.js # Dashboard interactivity
│   └── visualizations/  # Generated visualizations
│
├── templates/           # HTML templates
│   ├── index.html       # Main page
│   ├── dashboard.html   # Analysis dashboard
│   └── upload.html      # File upload form
│
├── utils/               # Utility modules
│   ├── __init__.py      # Package initialization
│   ├── pdf_extractor.py # File text extraction (PDF/TXT/HTML)
│   ├── nlp_processor.py # NLP processing and translation
│   ├── sentiment.py     # Sentiment analysis
│   └── visualizer.py    # Data visualization
│
├── uploads/             # Directory for uploaded files
└── analyses/            # Stored analysis results
```

## Key Configuration Options

### config.py
```python
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
    'physical', 'psychological', 'emotional', 
    'spiritual', 'relational', 'professional'
]

# Dimension keywords (both English and Chinese)
DIMENSION_KEYWORDS = {...}
DIMENSION_KEYWORDS_ZH = {...}
```

## Dependencies

### Core Requirements
```
flask==2.3.3
pdfplumber==0.10.2
plotly==5.18.0
pandas==2.1.2
numpy==1.26.1
jieba==0.42.1
requests==2.31.0
python-dotenv==1.0.0
wordcloud==1.9.2
matplotlib==3.8.1
openai==1.3.7
langdetect==1.0.9
chardet==5.2.0
```

### PDF & OCR Requirements
```
pytesseract==0.3.10
pdf2image==1.16.3
pillow==10.2.0
```

### HTML Processing
```
beautifulsoup4==4.12.2
```

## API Integration

### OpenAI Configuration
The application uses the OpenAI API for:
1. Enhanced sentiment analysis
2. Dimension categorization
3. Chinese to English translation
4. Comprehensive journal insights

To enable these features:
1. Obtain an API key from OpenAI
2. Add it to your `.env` file as `OPENAI_API_KEY=your_key_here`
3. The application will automatically use GPT-4o when available

### GPT-4o Benefits
- More nuanced understanding of context
- Better translation quality
- More insightful pattern recognition
- More personalized recommendations

## Troubleshooting

### Common Issues

#### PDF Text Extraction Problems
- **Issue**: OCR not working properly
- **Solution**: Ensure Tesseract is properly installed and in your PATH
- **Verification**: Run `tesseract --version` in your terminal

#### Missing Word Clouds
- **Issue**: Word clouds not appearing
- **Solution 1**: Check if the `static/visualizations` directory exists and is writable
- **Solution 2**: Verify that your text contains sufficient content for analysis

#### OpenAI API Errors
- **Issue**: Analysis features not working
- **Solution 1**: Check your API key is valid and has sufficient credits
- **Solution 2**: Verify internet connectivity
- **Solution 3**: The app will fall back to basic analysis if API is unavailable

#### Encoding Issues with Chinese Text
- **Issue**: Chinese characters appearing as gibberish
- **Solution**: Ensure your files are saved with UTF-8 encoding

## Future Improvements

### Planned Enhancements
- User accounts and authentication
- Export functionality for reports
- Additional language support
- Improved mobile interface
- Integration with journaling apps
- Advanced correlation analysis

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin new-feature`
5. Submit a pull request

### Coding Standards
- Follow PEP 8 guidelines
- Add docstrings to all functions
- Include unit tests for new features

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Jieba for Chinese word segmentation
- OpenAI for advanced language processing
- Tesseract for OCR capabilities
- Flask team for the web framework
- All contributors and supporters of the project

---

For questions or support, please open an issue on the repository.
