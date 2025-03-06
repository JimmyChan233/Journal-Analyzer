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
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create visualizations directory if it doesn't exist
VISUALIZATION_PATH = os.path.join(os.getcwd(), 'static', 'visualizations')
os.makedirs(VISUALIZATION_PATH, exist_ok=True)

def generate_word_cloud(word_data, filename_base, is_english=False):
    """
    Generate a word cloud image from word frequencies or keywords.
    
    Args:
        word_data (list/dict): List of (word, count) tuples or keywords list or dictionary
        filename_base (str): Base filename for the output image
        is_english (bool): Whether the word data is in English
        
    Returns:
        str: Path to the generated word cloud image
    """
    try:
        logger.info(f"Generating {'English' if is_english else 'Chinese'} word cloud for {filename_base}")
        
        # Convert data into a word frequency dictionary
        word_dict = {}
        
        # Check if we received keywords list
        if isinstance(word_data, list) and len(word_data) > 0:
            if all(isinstance(item, str) for item in word_data):
                logger.info(f"Using keyword list of {len(word_data)} items for word cloud")
                # Convert keywords list to word dictionary with equal weights
                word_dict = {word: 20 for word in word_data}
            elif all(isinstance(item, tuple) and len(item) == 2 for item in word_data):
                logger.info(f"Using word frequency list of {len(word_data)} items for word cloud")
                # Convert list of tuples to dictionary
                word_dict = {word: count for word, count in word_data}
        # Check if we already have a dictionary
        elif isinstance(word_data, dict) and len(word_data) > 0:
            logger.info(f"Using word dictionary with {len(word_data)} items for word cloud")
            word_dict = word_data
        else:
            logger.warning(f"Invalid data type for word cloud: {type(word_data)}, falling back to default content")
            # Fallback to default content
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
            # Use default font for English
            logger.info("Using default font for English word cloud")
        else:
            # Find a Chinese font
            try:
                # Try to find a suitable Chinese font
                import matplotlib.font_manager as fm
                system_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
                chinese_fonts = [f for f in system_fonts if any(name in f.lower()
                                for name in ['chinese', 'cjk', 'noto', 'ming', 'song', 'simhei', 'simsun', 'simkai'])]
                
                if chinese_fonts:
                    font_path = chinese_fonts[0]
                    logger.info(f"Using Chinese font: {font_path}")
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

def generate_sentiment_chart(entries):
    """
    Generate a sentiment analysis chart for journal entries over time.
    
    Args:
        entries (list): List of journal entry dictionaries
        
    Returns:
        dict: Plotly figure data for the sentiment chart
    """
    try:
        if not entries:
            return {}
        
        # Extract dates and sentiment scores
        dates = [entry['date'] for entry in entries]
        positive_scores = [entry['sentiment']['scores']['positive'] for entry in entries]
        negative_scores = [entry['sentiment']['scores']['negative'] for entry in entries]
        neutral_scores = [entry['sentiment']['scores']['neutral'] for entry in entries]
        
        # Create figure
        fig = go.Figure()
        
        # Add sentiment score lines
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
        
        # Add dominant sentiment as colored background
        for i, entry in enumerate(entries):
            dominant = entry['sentiment']['dominant']
            color = 'rgba(0, 128, 0, 0.2)' if dominant == 'positive' else \
                    'rgba(255, 0, 0, 0.2)' if dominant == 'negative' else \
                    'rgba(0, 0, 255, 0.2)'
            
            if i < len(entries) - 1:
                x0, x1 = dates[i], dates[i+1]
            else:
                # For the last point, extend a bit
                from datetime import datetime, timedelta
                try:
                    last_date = datetime.strptime(dates[i], '%Y-%m-%d')
                    next_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
                    x0, x1 = dates[i], next_date
                except:
                    x0, x1 = dates[i], dates[i]
            
            fig.add_shape(
                type="rect",
                xref="x",
                yref="paper",
                x0=x0,
                y0=0,
                x1=x1,
                y1=1,
                fillcolor=color,
                opacity=0.3,
                layer="below",
                line_width=0,
            )
        
        # Update layout
        fig.update_layout(
            title="Sentiment Analysis Over Time",
            xaxis_title="Date",
            yaxis_title="Sentiment Score (%)",
            yaxis=dict(range=[0, 100]),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=10, r=10, t=30, b=10),
        )
        
        # Convert to JSON for JavaScript
        return json.loads(fig.to_json())
    
    except Exception as e:
        logger.error(f"Error generating sentiment chart: {str(e)}")
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

def generate_trend_analysis(entries):
    """
    Generate trend analysis visualizations showing patterns and correlations.
    
    Args:
        entries (list): List of journal entry dictionaries
        
    Returns:
        dict: Plotly figure data for trend analysis
    """
    try:
        if not entries or len(entries) < 3:
            return {}
        
        # Create subplots: dimension trends and correlation heatmap
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Dimension Trends Over Time", "Dimension Correlations"),
            specs=[[{"type": "scatter"}], [{"type": "heatmap"}]],
            vertical_spacing=0.2,
            row_heights=[0.6, 0.4]
        )
        
        # Extract dates and dimension scores
        dates = [entry['date'] for entry in entries]
        dimensions = Config.DIMENSIONS
        
        # Add dimension trend lines
        for dim in dimensions:
            values = [entry['dimensions'].get(dim, 0) for entry in entries]
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
        
        # Calculate correlation matrix
        dim_values = {dim: [] for dim in dimensions}
        for entry in entries:
            for dim in dimensions:
                dim_values[dim].append(entry['dimensions'].get(dim, 0))
        
        # Convert to numpy arrays
        dim_arrays = {dim: np.array(values) for dim, values in dim_values.items()}
        
        # Calculate correlation matrix
        corr_matrix = np.zeros((len(dimensions), len(dimensions)))
        for i, dim1 in enumerate(dimensions):
            for j, dim2 in enumerate(dimensions):
                # Calculate correlation if there's variance
                if np.std(dim_arrays[dim1]) > 0 and np.std(dim_arrays[dim2]) > 0:
                    corr_matrix[i, j] = np.corrcoef(dim_arrays[dim1], dim_arrays[dim2])[0, 1]
                else:
                    corr_matrix[i, j] = 0
        
        # Add correlation heatmap
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix,
                x=[dim.capitalize() for dim in dimensions],
                y=[dim.capitalize() for dim in dimensions],
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="Correlation")
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title="Journal Entry Trends and Correlations",
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.6,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        
        # Update first subplot layout
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Score", range=[0, 10], row=1, col=1)
        
        # Convert to JSON for JavaScript
        return json.loads(fig.to_json())
    
    except Exception as e:
        logger.error(f"Error generating trend analysis: {str(e)}")
        return {}
