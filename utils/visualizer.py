import os
import json
import numpy as np
import matplotlib

# Set non-interactive backend before importing pyplot
matplotlib.use('Agg')  # Use the Agg backend which doesn't require a GUI
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from openai import OpenAI

client = OpenAI()
from config import Config

from openai import OpenAI

client = OpenAI(api_key=Config.OPENAI_API_KEY)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create visualizations directory if it doesn't exist
VISUALIZATION_PATH = os.path.join(os.getcwd(), 'static', 'visualizations')
os.makedirs(VISUALIZATION_PATH, exist_ok=True)

def extract_json(text):
    """
    If the returned text is wrapped in markdown code fences, remove them.
    """
    text = text.strip()
    # If text starts with triple backticks, remove them
    if text.startswith("```"):
        # Optionally, remove a language identifier like "```json"
        lines = text.splitlines()
        # Remove first and last line
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1]).strip()
    return text


def generate_word_cloud(word_data, filename_base, is_english=False):
    """
    Generate a word cloud image from word frequencies or keywords.
    
    Args:
        word_data (list/dict): List of (word, count) tuples or keywords list or dictionary.
        filename_base (str): Base filename for the output image.
        is_english (bool): Whether the word data is in English.
        
    Returns:
        str: Path to the generated word cloud image.
    """
    try:
        logger.info(f"Generating {'English' if is_english else 'Chinese'} word cloud for {filename_base}")

        # Convert data into a word frequency dictionary
        word_dict = {}
        if isinstance(word_data, list) and len(word_data) > 0:
            if all(isinstance(item, str) for item in word_data):
                logger.info(f"Using keyword list of {len(word_data)} items for word cloud")
                # Convert keywords list to word dictionary with equal weights
                word_dict = {word: 20 for word in word_data}
            elif all(isinstance(item, tuple) and len(item) == 2 for item in word_data):
                logger.info(f"Using word frequency list of {len(word_data)} items for word cloud")
                word_dict = {word: count for word, count in word_data}
        elif isinstance(word_data, dict) and len(word_data) > 0:
            logger.info(f"Using word dictionary with {len(word_data)} items for word cloud")
            word_dict = word_data
        else:
            logger.warning(f"Invalid data type for word cloud: {type(word_data)}, falling back to default content")
            if is_english:
                word_dict = {"Journal": 10, "Writing": 9, "Thoughts": 8, "Mood": 7, "Life": 6, "Reflection": 5}
            else:
                word_dict = {"日记": 10, "写作": 9, "感想": 8, "心情": 7, "生活": 6, "思考": 5}

        # Generate a unique filename
        safe_filename = ''.join(c if c.isalnum() else '_' for c in str(filename_base))
        output_filename = f"wordcloud_{safe_filename}.png"
        output_path = os.path.join(VISUALIZATION_PATH, output_filename)

        logger.info(f"Word cloud will be saved to: {output_path}")

        # Choose font based on language
        font_path = None
        if is_english:
            logger.info("Using default font for English word cloud")
        else:
            # Use your custom Chinese font if available
            custom_font = os.path.join('static', 'fonts', 'NotoSansSC-VariableFont_wght.ttf')
            if os.path.exists(custom_font):
                font_path = custom_font
                logger.info(f"Using custom Chinese font: {font_path}")
            else:
                # Fallback: attempt to auto-detect a Chinese font from system fonts
                try:
                    import matplotlib.font_manager as fm
                    system_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
                    chinese_fonts = [f for f in system_fonts if any(name in f.lower() for name in ['chinese', 'cjk', 'noto', 'ming', 'song', 'simhei', 'simsun', 'simkai'])]
                    if chinese_fonts:
                        font_path = chinese_fonts[0]
                        logger.info(f"Using detected Chinese font: {font_path}")
                    else:
                        logger.warning("No suitable Chinese font found, using default font")
                except Exception as e:
                    logger.error(f"Error finding system fonts: {str(e)}")

        # Create the word cloud with explicit parameters
        wordcloud_params = {
            'width': 500,
            'height': 300,
            'background_color': 'white',
            'max_words': 100,
            'collocations': False,    # Don't include collocations to avoid duplicates
            'min_font_size': 10,
            'max_font_size': 150,
            'random_state': 42        # For reproducibility
        }

        # Add font path if needed for Chinese
        if font_path and not is_english:
            wordcloud_params['font_path'] = font_path

        # Generate the word cloud
        wordcloud = WordCloud(**wordcloud_params).generate_from_frequencies(word_dict)

        # Save the image with explicit parameters
        plt.figure(figsize=(6, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(output_path, format='png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Word cloud successfully saved to: {output_path}")

        # Return the relative path from the static directory
        return os.path.join('visualizations', output_filename)

    except Exception as e:
        logger.error(f"Error generating word cloud: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return ""



def analyze_sentiment_trends_api(entries):
    """
    Use ChatGPT API to analyze sentiment trends from journal entries.
    Returns a JSON object with arrays for dates and sentiment scores.
    Expected JSON format:
    {
      "dates": ["YYYY-MM-DD", ...],
      "positive": [70.5, 68.0, ...],
      "neutral": [20.0, 22.0, ...],
      "negative": [9.5, 10.0, ...]
    }
    """
    if not Config.OPENAI_API_KEY:
        return None

    # Build a summary from each entry's sentiment
    summaries = []
    for entry in entries:
        date = entry.get('date', '')
        sentiment = entry.get('sentiment', {}).get('scores', {})
        pos = sentiment.get('positive', 0)
        neu = sentiment.get('neutral', 0)
        neg = sentiment.get('negative', 0)
        summaries.append(f"Date: {date}, Positive: {pos}, Neutral: {neu}, Negative: {neg}")
    summary_text = "\n".join(summaries)

    prompt = f"""
You are an expert data analyst. Given the following journal entry sentiment summaries, please analyze the sentiment trends over time.
Provide a JSON object with the following format:
{{
   "dates": [list of dates in YYYY-MM-DD format],
   "positive": [list of positive sentiment percentages corresponding to each date],
   "neutral": [list of neutral sentiment percentages corresponding to each date],
   "negative": [list of negative sentiment percentages corresponding to each date]
}}
Use the data below:
{summary_text}

ONLY output the JSON object exactly as specified, with no additional text.
    """
    logger.info(f"Sentiment trends prompt: {prompt[:200]}... [truncated]")

    try:
        response = client.chat.completions.create(model="gpt-4o",
        messages=[
          {"role": "system", "content": "You are a data analyst proficient in generating JSON output from text data."},
          {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=3000)
        result_text = response.choices[0].message.content.strip()
        if not result_text:
            logger.error("The API returned an empty response for sentiment trend analysis.")
            return None
        result_text = extract_json(result_text)
        try:
            result_json = json.loads(result_text)
            return result_json
        except Exception as parse_err:
            logger.error(f"Error parsing JSON from sentiment trend analysis response: {parse_err}. Raw response: {result_text}")
            return None
    except Exception as e:
        logger.error(f"Error in API sentiment trend analysis: {str(e)}")
        return None


def generate_sentiment_chart(entries):
    """
    Generate a sentiment analysis chart using API-based analysis.
    """
    api_data = None
    if Config.OPENAI_API_KEY:
        api_data = analyze_sentiment_trends_api(entries)
    if api_data:
         dates = api_data.get("dates", [])
         positive_scores = api_data.get("positive", [])
         neutral_scores = api_data.get("neutral", [])
         negative_scores = api_data.get("negative", [])

         fig = go.Figure()
         fig.add_trace(go.Scatter(
             x=dates,
             y=positive_scores,
             mode='lines+markers',
             name='Positive',
             line=dict(color='green', width=2),
             marker=dict(size=8)
         ))
         fig.add_trace(go.Scatter(
             x=dates,
             y=negative_scores,
             mode='lines+markers',
             name='Negative',
             line=dict(color='red', width=2),
             marker=dict(size=8)
         ))
         fig.add_trace(go.Scatter(
             x=dates,
             y=neutral_scores,
             mode='lines+markers',
             name='Neutral',
             line=dict(color='blue', width=2),
             marker=dict(size=8)
         ))
         fig.update_layout(
             title="Sentiment Analysis Over Time (API Based)",
             xaxis_title="Date",
             yaxis_title="Sentiment Score (%)",
             yaxis=dict(range=[0, 100]),
             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
             margin=dict(l=10, r=10, t=30, b=10),
         )
         return json.loads(fig.to_json())
    else:
         logger.warning("API sentiment trend analysis failed; no data available.")
         return {}

def generate_dimension_chart(entries):
    """
    Generate a radar chart showing the average scores for each dimension.
    
    Args:
        entries (list): List of journal entry dictionaries
        
    Returns:
        dict: Plotly figure data for the dimension radar chart
    """
    try:
        if not entries:
            return {}

        # Calculate average dimension scores
        dimensions = Config.DIMENSIONS
        dimension_data = {dim: [] for dim in dimensions}

        for entry in entries:
            for dim in dimensions:
                if dim in entry['dimensions']:
                    dimension_data[dim].append(entry['dimensions'][dim])

        # Calculate averages
        avg_dimensions = {dim: sum(scores)/len(scores) if scores else 0
                         for dim, scores in dimension_data.items()}

        # Prepare data for radar chart
        r_values = [avg_dimensions[dim] for dim in dimensions]
        theta = dimensions

        # Create radar chart
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=theta,
            fill='toself',
            name='Average Dimension Scores',
            line_color='rgb(31, 119, 180)',
            fillcolor='rgba(31, 119, 180, 0.3)'
        ))

        # Update layout
        fig.update_layout(
            title="Self-Care Dimensions Analysis",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=False
        )

        # Convert to JSON for JavaScript
        return json.loads(fig.to_json())

    except Exception as e:
        logger.error(f"Error generating dimension chart: {str(e)}")
        return {}

def analyze_trend_api(entries):
    """
    Use ChatGPT API to analyze overall trends and correlations from journal entries.
    Expected JSON format:
    {{
      "dates": ["YYYY-MM-DD", ...],
      "dimension_trends": {{
           "physical": [score1, score2, ...],
           "psychological": [score1, score2, ...],
           "emotional": [score1, score2, ...],
           "spiritual": [score1, score2, ...],
           "relational": [score1, score2, ...],
           "professional": [score1, score2, ...]
      }},
      "correlation_matrix": [
          [1, 0.5, ...],
          [0.5, 1, ...],
          ...
      ]
    }}
    """
    if not Config.OPENAI_API_KEY:
        return None

    summaries = []
    for entry in entries:
        date = entry.get('date', '')
        dims = entry.get('dimensions', {})
        dims_str = ", ".join([f"{k}: {v}" for k, v in dims.items()])
        summaries.append(f"Date: {date}, {dims_str}")
    summary_text = "\n".join(summaries)

    prompt = f"""
You are an expert data analyst. Given the following journal entry dimension summaries, please analyze the trends and correlations among the self-care dimensions over time.
Provide a JSON object with the following keys:
- "dates": a list of dates in YYYY-MM-DD format.
- "dimension_trends": a dictionary where each key is a dimension (physical, psychological, emotional, spiritual, relational, professional) and its value is a list of scores corresponding to each date in the order provided.
- "correlation_matrix": a 2D list representing the correlation matrix between the dimensions in the order: physical, psychological, emotional, spiritual, relational, professional.

Use the data below:
{summary_text}

ONLY output the JSON object exactly as specified, with no additional text.
    """
    logger.info(f"Trend analysis prompt: {prompt[:200]}... [truncated]")

    try:
        response = client.chat.completions.create(model="gpt-4o",
        messages=[
          {"role": "system", "content": "You are a data analyst who outputs strictly JSON."},
          {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=3000)
        result_text = response.choices[0].message.content.strip()
        if not result_text:
            logger.error("The API returned an empty response for trend analysis.")
            return None
        result_text = extract_json(result_text)
        try:
            result_json = json.loads(result_text)
            return result_json
        except Exception as parse_err:
            logger.error(f"Error parsing JSON from trend analysis response: {parse_err}. Raw response: {result_text}")
            return None
    except Exception as e:
        logger.error(f"Error in API trend analysis: {str(e)}")
        return None

def generate_trend_analysis(entries):
    """
    Generate trend analysis visualizations using API-based analysis.
    """
    api_data = None
    if Config.OPENAI_API_KEY:
        api_data = analyze_trend_api(entries)
    if api_data:
         dates = api_data.get("dates", [])
         dimension_trends = api_data.get("dimension_trends", {})
         correlation_matrix = api_data.get("correlation_matrix", [])

         fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Dimension Trends Over Time", "Dimension Correlations"),
            specs=[[{"type": "scatter"}], [{"type": "heatmap"}]],
            vertical_spacing=0.2,
            row_heights=[0.6, 0.4]
         )
         for dim, values in dimension_trends.items():
             fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=values,
                    mode='lines+markers',
                    name=dim.capitalize(),
                    line=dict(width=2),
                    marker=dict(size=6)
                ),
                row=1, col=1
             )
         if correlation_matrix and isinstance(correlation_matrix, list):
             fig.add_trace(
                go.Heatmap(
                    z=correlation_matrix,
                    x=[d.capitalize() for d in ["physical", "psychological", "emotional", "spiritual", "relational", "professional"]],
                    y=[d.capitalize() for d in ["physical", "psychological", "emotional", "spiritual", "relational", "professional"]],
                    colorscale='RdBu',
                    zmid=0,
                    colorbar=dict(title="Correlation")
                ),
                row=2, col=1
             )
         fig.update_layout(
             title="Journal Entry Trends and Correlations (API Based)",
             height=800,
             legend=dict(orientation="h", yanchor="bottom", y=0.6, xanchor="center", x=0.5),
             margin=dict(l=10, r=10, t=50, b=10),
         )
         fig.update_xaxes(title_text="Date", row=1, col=1)
         fig.update_yaxes(title_text="Score", range=[0, 10], row=1, col=1)
         return json.loads(fig.to_json())
    else:
         logger.warning("API trend analysis failed; no data available.")
         return {}

def generate_dimension_details_api(entries):
    """
    Use ChatGPT API to generate detailed explanations for each self-care dimension.
    Returns a dict mapping each dimension to its detailed explanation.
    """
    if not Config.OPENAI_API_KEY or not entries:
        return {}

    try:
        # Build a summary of dimensions from the entries
        summary_lines = []
        for entry in entries:
            date = entry.get("date", "")
            dims = entry.get("dimensions", {})
            dims_str = ", ".join([f"{k}: {v}" for k, v in dims.items()])
            summary_lines.append(f"Date: {date}, {dims_str}")
        summary_text = "\n".join(summary_lines)

        prompt = f"""
You are a wellness expert focused on holistic health. Based on the following journal entry dimension summaries, please generate detailed explanations for each self-care dimension (physical, psychological, emotional, spiritual, relational, professional). For each dimension, describe what a high or low score might indicate and provide actionable recommendations.
Please return the result in JSON format exactly as specified below:
{{
  "physical": "Detailed explanation...",
  "psychological": "Detailed explanation...",
  "emotional": "Detailed explanation...",
  "spiritual": "Detailed explanation...",
  "relational": "Detailed explanation...",
  "professional": "Detailed explanation..."
}}

Data：
{summary_text}

ONLY output the JSON object exactly as specified, with no additional text.
        """
        logger.info(f"Dimension details prompt: {prompt[:200]}... [truncated]")

        response = client.chat.completions.create(model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a health consultant adept at summarizing and providing actionable recommendations. Please return only JSON formatted data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=3000)
        result_text = response.choices[0].message.content.strip()
        result_text = extract_json(result_text)
        try:
            details = json.loads(result_text)
            return details
        except Exception as parse_err:
            logger.error(f"Error parsing JSON from dimension details response: {parse_err}. Raw response: {result_text}")
            return {}
    except Exception as e:
        logger.error(f"Error generating dimension details: {str(e)}")
        return {}

