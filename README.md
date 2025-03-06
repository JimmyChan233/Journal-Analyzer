# AI-Powered Journal Analysis Tool

An advanced tool for analyzing Chinese journal entries, extracting insights, and visualizing wellbeing trends.

## Overview

This application is designed to help you analyze your personal journal entries written in Chinese. It categorizes content into different self-care dimensions, performs sentiment analysis, identifies common themes through word frequency analysis, and visualizes trends over time.

## Features

- **Text Extraction**: Extract text from PDF or TXT journal entries
- **Chinese NLP Processing**: Segment Chinese text, remove stopwords, and analyze content
- **Sentiment Analysis**: Analyze emotional tone (positive, neutral, negative)
- **Self-Care Dimensions Categorization**: Categorize content into physical, psychological, emotional, spiritual, relational, and professional dimensions
- **Word Frequency Analysis**: Identify common themes and topics
- **Trend Analysis**: Track patterns over time and discover correlations
- **Interactive Visualizations**: View insights through interactive charts and dashboards

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/journal-analyzer.git
   cd journal-analyzer
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your configuration:
   ```
   SECRET_KEY=your_secret_key_here
   DEBUG=True
   OPENAI_API_KEY=your_openai_api_key  # Optional, enhances analysis if provided
   ```

5. Run the application:
   ```bash
   python app.py
   ```

6. Open your browser and navigate to `http://127.0.0.1:5000`

## Usage

1. **Upload Journal Files**: 
   - Navigate to the upload page
   - Select one or more PDF or TXT files containing your Chinese journal entries
   - The system supports both:
     - Single entry per file (named with dates like journal_2023-10-15.pdf)
     - Multiple entries in one file (where each entry starts with a date like "Monday, February 3, 2025")

2. **View Analysis Dashboard**:
   - After processing, you'll be redirected to the dashboard
   - Explore different tabs for various insights:
     - **Overview**: Summary of sentiment, dimensions, and word frequency
     - **Sentiment**: Detailed emotional analysis over time
     - **Dimensions**: Self-care dimension breakdown and balance
     - **Trends**: Patterns and correlations
     - **Journal Entries**: List of processed entries with previews

3. **Interpret Results**:
   - **Sentiment Analysis**: Track emotional patterns
   - **Dimension Radar**: See which wellbeing areas receive most attention
   - **Dimension Correlations**: Discover relationships between different aspects of wellbeing
   - **Word Clouds**: Identify frequently discussed themes

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
├── uploads/             # Uploaded journal PDFs
├── analyses/            # Stored analysis results
│
├── utils/               # Utility modules
│   ├── __init__.py      
│   ├── pdf_extractor.py # PDF text extraction
│   ├── nlp_processor.py # NLP processing
│   ├── sentiment.py     # Sentiment analysis
│   └── visualizer.py    # Data visualization
```

## Customization

### Dimension Keywords

You can customize the keywords used to categorize dimensions by editing the `config.py` file:

```python
DIMENSION_KEYWORDS_ZH = {
    'physical': ['运动', '锻炼', '睡眠', ...],
    'psychological': ['思考', '思想', '精神', ...],
    # ...
}
```

### Sentiment Dictionary

To improve sentiment analysis, you can enhance the Chinese sentiment dictionary in `utils/chinese_sentiment_dict.json`.

## Using OpenAI API (Optional)

For enhanced analysis, the tool can integrate with OpenAI's API:

1. Get an API key from OpenAI
2. Add it to your `.env` file as `OPENAI_API_KEY=your_key_here`
3. The tool will automatically use the API for more accurate sentiment and dimension analysis

## Requirements

- Python 3.8+
- Flask
- pdfplumber
- jieba
- plotly
- pandas
- numpy
- wordcloud
- matplotlib
- requests
- python-dotenv
- openai (optional)

## License

MIT License

## Acknowledgements

This project was created as a creative project focusing on wellbeing analysis and self-care management.
