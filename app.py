import os
import json
import logging
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import utility modules
from utils.pdf_extractor import extract_text_from_file, split_journal_into_entries
from utils.nlp_processor import (
    process_chinese_text,
    analyze_dimensions,
    generate_journal_insights,
    translate_keywords
)
from utils.sentiment import analyze_sentiment
from utils.visualizer import (
    generate_word_cloud,
    generate_sentiment_chart,
    generate_dimension_chart,
    generate_trend_analysis
)

# Import configuration
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Increase max content length (100 MB)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ANALYSIS_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle file uploads."""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        files = request.files.getlist('file')
        
        if not files or files[0].filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        file_paths = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                file_paths.append(filepath)
        
        if file_paths:
            # Process the uploaded files
            results = process_files(file_paths)
            
            # Save analysis results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join(app.config['ANALYSIS_FOLDER'], f'analysis_{timestamp}.json')
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            
            return redirect(url_for('dashboard', analysis_id=f'analysis_{timestamp}.json'))
    
    return render_template('upload.html')

@app.route('/dashboard')
@app.route('/dashboard/<analysis_id>')
def dashboard(analysis_id=None):
    """Display the analysis dashboard."""
    logger.info(f"Accessing dashboard with analysis_id: {analysis_id}")
    
    if analysis_id:
        result_path = os.path.join(app.config['ANALYSIS_FOLDER'], analysis_id)
        if os.path.exists(result_path):
            with open(result_path, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
            logger.info(f"Loaded analysis from {result_path}")
            return render_template('dashboard.html', analysis=analysis)
    
    # List available analyses
    analyses = []
    if os.path.exists(app.config['ANALYSIS_FOLDER']):
        for filename in os.listdir(app.config['ANALYSIS_FOLDER']):
            if filename.startswith('analysis_') and filename.endswith('.json'):
                analyses.append(filename)
        logger.info(f"Found {len(analyses)} existing analyses")
    
    return render_template('dashboard.html', analyses=analyses)

@app.route('/api/analysis', methods=['GET'])
def get_analyses():
    """API endpoint to get list of analyses."""
    analyses = []
    for filename in os.listdir(app.config['ANALYSIS_FOLDER']):
        if filename.startswith('analysis_') and filename.endswith('.json'):
            timestamp = filename.replace('analysis_', '').replace('.json', '')
            date_obj = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            formatted_date = date_obj.strftime("%Y-%m-%d %H:%M:%S")
            analyses.append({
                'id': filename,
                'date': formatted_date
            })
    return jsonify(analyses)

@app.route('/api/analysis/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    """API endpoint to get a specific analysis."""
    result_path = os.path.join(app.config['ANALYSIS_FOLDER'], analysis_id)
    if os.path.exists(result_path):
        with open(result_path, 'r', encoding='utf-8') as f:
            analysis = json.load(f)
        return jsonify(analysis)
    return jsonify({'error': 'Analysis not found'}), 404

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_files(file_paths):
    """Process uploaded files and generate analysis."""
    results = {
        'entries': [],
        'overall': {
            'sentiment': {
                'positive': 0,
                'neutral': 0,
                'negative': 0
            },
            'dimensions': {
                'physical': 0,
                'psychological': 0,
                'emotional': 0,
                'spiritual': 0,
                'relational': 0,
                'professional': 0
            },
            'word_frequency': {}
        }
    }
    
    # Store all keywords across entries for summary word cloud
    all_chinese_keywords = []
    all_english_keywords = {}
    
    for filepath in file_paths:
        # Extract text from file (PDF, TXT, or HTML)
        text = extract_text_from_file(filepath)
        if not text:
            continue
        
        logger.info(f"Successfully extracted text from {filepath}: {len(text)} characters")
        logger.info(f"Text preview: {text[:100]}...")
        
        # Split the text into individual entries
        entries = split_journal_into_entries(text)
        logger.info(f"Split into {len(entries)} entries")
        
        # Process each entry
        for entry_data in entries:
            entry_text = entry_data['text']
            entry_date = entry_data['date']
            
            # Process Chinese text
            processed_text = process_chinese_text(entry_text)
            
            # Analyze sentiment
            sentiment = analyze_sentiment(processed_text['segmented_text'])
            
            # Get filename for reference
            filename = os.path.basename(filepath)
            entry_id = f"{filename}_{entry_date}"
            
            # Collect keywords for summary word cloud
            word_cloud_data = processed_text.get('keywords', []) if processed_text.get('keywords') else processed_text['word_frequency']
            
            # Add to all keywords collections
            if isinstance(word_cloud_data, list):
                if len(word_cloud_data) > 0 and isinstance(word_cloud_data[0], tuple):
                    # Word frequency list
                    all_chinese_keywords.extend(word_cloud_data)
                else:
                    # Simple keywords list
                    all_chinese_keywords.extend([(word, 10) for word in word_cloud_data])
            
            # Translate keywords to English for this entry
            english_word_freq = translate_keywords(word_cloud_data)
            logger.info(f"Translated {len(english_word_freq)} keywords to English for entry {entry_date}")
            
            # Add to all English keywords
            for word, count in english_word_freq.items():
                if word in all_english_keywords:
                    all_english_keywords[word] += count
                else:
                    all_english_keywords[word] = count
            
            # Create entry without individual word clouds for now
            entry = {
                'date': entry_date,
                'filepath': filepath,
                'filename': filename,
                'text': entry_text[:200] + '...' if len(entry_text) > 200 else entry_text,  # Preview
                'sentiment': sentiment,
                'dimensions': processed_text['dimensions'],
                'word_frequency': processed_text['word_frequency'][:20],  # Top 20 words
                'keywords': processed_text.get('keywords', [])
            }
            
            results['entries'].append(entry)
            
            # Update overall stats
            results['overall']['sentiment'][sentiment['dominant']] += 1
            
            for dim, score in processed_text['dimensions'].items():
                results['overall']['dimensions'][dim] += score
            
            # Combine word frequencies
            for word, count in processed_text['word_frequency']:
                if word in results['overall']['word_frequency']:
                    results['overall']['word_frequency'][word] += count
                else:
                    results['overall']['word_frequency'][word] = count
    
    # Sort entries by date
    results['entries'].sort(key=lambda x: x['date'])
    
    # Now generate summary word clouds for all entries
    if all_chinese_keywords:
        # Combine duplicate words in Chinese keywords
        chinese_word_dict = {}
        for word, count in all_chinese_keywords:
            if word in chinese_word_dict:
                chinese_word_dict[word] += count
            else:
                chinese_word_dict[word] = count
        
        # Generate summary word clouds
        summary_chinese_cloud = generate_word_cloud(chinese_word_dict, "summary_chinese")
        summary_english_cloud = generate_word_cloud(all_english_keywords, "summary_english", is_english=True)
        
        # Add the summary word clouds to the results
        results['summary_word_clouds'] = {
            'chinese': summary_chinese_cloud,
            'english': summary_english_cloud
        }
        
        # Also add the summary word clouds to each entry for display consistency
        for entry in results['entries']:
            entry['word_cloud_path'] = summary_english_cloud
            entry['chinese_word_cloud_path'] = summary_chinese_cloud
    
    # Generate overall visualizations
    results['visualizations'] = {
        'sentiment_chart': generate_sentiment_chart(results['entries']),
        'dimension_chart': generate_dimension_chart(results['entries']),
        'trend_analysis': generate_trend_analysis(results['entries'])
    }
    
    # Generate AI insights using GPT-4o if available
    if results['entries']:
        results['insights'] = generate_journal_insights(results['entries'], use_gpt4o=True)
    else:
        results['insights'] = {
            "summary": "No journal entries to analyze.",
            "patterns": [],
            "recommendations": [],
            "wellbeing_assessment": "Please upload journal entries for analysis."
        }
    
    return results
    
    # Sort entries by date
    results['entries'].sort(key=lambda x: x['date'])
    
    # Generate overall visualizations
    results['visualizations'] = {
        'sentiment_chart': generate_sentiment_chart(results['entries']),
        'dimension_chart': generate_dimension_chart(results['entries']),
        'trend_analysis': generate_trend_analysis(results['entries'])
    }
    
    return results

def extract_date_from_filename(filename):
    """Extract date from filename or return current date."""
    # Try to extract date from filename (format: YYYY-MM-DD or similar)
    # This is a simple implementation - you may need to customize based on your naming convention
    import re
    date_match = re.search(r'(\d{4}[-_]\d{2}[-_]\d{2})', filename)
    if date_match:
        date_str = date_match.group(1).replace('_', '-')
        return date_str
    
    # If no date found, use file creation time
    return datetime.now().strftime("%Y-%m-%d")

if __name__ == '__main__':
    app.run(debug=True)
