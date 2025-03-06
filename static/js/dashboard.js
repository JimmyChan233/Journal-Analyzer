/**
 * Dashboard.js - Main JavaScript for Journal Analysis Dashboard
 * Manages interactivity and dynamic visualizations
 */

// Store global analysis data
let globalAnalysisData = null;

/**
 * Initialize all dashboard components once the document is loaded
 */
document.addEventListener('DOMContentLoaded', function() {
    // Tabs functionality
    const tabEls = document.querySelectorAll('button[data-bs-toggle="tab"]');
    tabEls.forEach(tabEl => {
        tabEl.addEventListener('shown.bs.tab', function (event) {
            // Trigger resize to fix any Plotly layout issues
            window.dispatchEvent(new Event('resize'));
            
            // Load any deferred visualizations for the active tab
            const tabId = event.target.getAttribute('data-bs-target').substring(1);
            loadDeferredVisualizations(tabId);
        });
    });
    
    // Fetch analysis data if we're on the dashboard page with an analysis ID
    const analysisId = document.getElementById('analysisData')?.getAttribute('data-id');
    if (analysisId) {
        fetchAnalysisData(analysisId);
    }
    
    // Initialize dropdowns and tooltips
    initBootstrapComponents();
});

/**
 * Fetch analysis data from the API
 * @param {string} analysisId - The ID of the analysis to fetch
 */
function fetchAnalysisData(analysisId) {
    // Show loading indicator
    showLoadingIndicator(true);
    
    // Fetch the analysis data
    fetch(`/api/analysis/${analysisId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to fetch analysis data');
            }
            return response.json();
        })
        .then(data => {
            globalAnalysisData = data;
            initializeDashboard(data);
            showLoadingIndicator(false);
        })
        .catch(error => {
            console.error('Error fetching analysis data:', error);
            showErrorMessage('Failed to load analysis data. Please try again.');
            showLoadingIndicator(false);
        });
}

/**
 * Initialize the dashboard with analysis data
 * @param {Object} analysisData - The analysis data to display
 */
function initializeDashboard(analysisData) {
    // Initialize the main visualizations for the Overview tab
    initializeOverviewCharts(analysisData);
    
    // Add analysis metadata and summary stats
    updateAnalysisSummary(analysisData);
    
    // Prepare data for other tabs (but don't render yet to improve performance)
    prepareTabData(analysisData);
}

/**
 * Initialize the charts on the Overview tab
 * @param {Object} analysisData - The analysis data to display
 */
function initializeOverviewCharts(analysisData) {
    if (!analysisData || !analysisData.visualizations) {
        return;
    }
    
    // Sentiment chart
    if (analysisData.visualizations.sentiment_chart) {
        Plotly.newPlot(
            'sentimentChart', 
            analysisData.visualizations.sentiment_chart.data,
            {
                ...analysisData.visualizations.sentiment_chart.layout,
                height: 300,
                margin: { t: 30, r: 10, b: 40, l: 50 }
            }
        );
    }
    
    // Dimensions radar chart
    if (analysisData.visualizations.dimension_chart) {
        Plotly.newPlot(
            'dimensionsChart', 
            analysisData.visualizations.dimension_chart.data,
            {
                ...analysisData.visualizations.dimension_chart.layout,
                height: 300,
                margin: { t: 30, r: 10, b: 10, l: 10 }
            }
        );
    }
    
    // Populate top dimensions
    populateTopDimensions(analysisData);
}

/**
 * Update the analysis summary section with metadata
 * @param {Object} analysisData - The analysis data
 */
function updateAnalysisSummary(analysisData) {
    if (!analysisData || !analysisData.entries || analysisData.entries.length === 0) {
        return;
    }
    
    // Update entry count
    const entryCountEl = document.getElementById('entryCount');
    if (entryCountEl) {
        entryCountEl.textContent = analysisData.entries.length;
    }
    
    // Update date range
    const dateRangeEl = document.getElementById('dateRange');
    if (dateRangeEl && analysisData.entries.length > 0) {
        const firstDate = analysisData.entries[0].date;
        const lastDate = analysisData.entries[analysisData.entries.length - 1].date;
        dateRangeEl.textContent = `${firstDate} to ${lastDate}`;
    }
    
    // Update sentiment distribution
    updateSentimentSummary(analysisData);
}

/**
 * Update the sentiment summary section
 * @param {Object} analysisData - The analysis data
 */
function updateSentimentSummary(analysisData) {
    if (!analysisData || !analysisData.overall || !analysisData.overall.sentiment) {
        return;
    }
    
    const sentiment = analysisData.overall.sentiment;
    
    // Set sentiment counts in circles
    document.querySelector('.sentiment-circle.positive').textContent = sentiment.positive || 0;
    document.querySelector('.sentiment-circle.neutral').textContent = sentiment.neutral || 0;
    document.querySelector('.sentiment-circle.negative').textContent = sentiment.negative || 0;
}

/**
 * Populate the top dimensions section
 * @param {Object} analysisData - The analysis data
 */
function populateTopDimensions(analysisData) {
    if (!analysisData || !analysisData.overall || !analysisData.overall.dimensions) {
        return;
    }
    
    const topDimensionsEl = document.getElementById('topDimensions');
    if (!topDimensionsEl) return;
    
    // Sort dimensions by score
    const dimensions = Object.entries(analysisData.overall.dimensions)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3);
    
    // Define colors for dimensions
    const colors = {
        'physical': 'bg-success',
        'psychological': 'bg-info',
        'emotional': 'bg-warning',
        'spiritual': 'bg-primary',
        'relational': 'bg-danger',
        'professional': 'bg-secondary'
    };
    
    // Create HTML for progress bars
    let html = '';
    dimensions.forEach(([dim, value]) => {
        // Calculate percentage (assuming max is 10)
        const percent = Math.round((value / 10) * 100);
        const color = colors[dim] || 'bg-primary';
        
        html += `
        <div class="mb-2">
            <div class="d-flex justify-content-between">
                <span>${dim.charAt(0).toUpperCase() + dim.slice(1)}</span>
                <span>${(value/analysisData.entries.length).toFixed(1)}/10</span>
            </div>
            <div class="progress">
                <div class="progress-bar ${color}" role="progressbar" style="width: ${percent}%" 
                    aria-valuenow="${value}" aria-valuemin="0" aria-valuemax="10"></div>
            </div>
        </div>`;
    });
    
    topDimensionsEl.innerHTML = html;
}

/**
 * Prepare data for tabs that aren't initially visible
 * @param {Object} analysisData - The analysis data
 */
function prepareTabData(analysisData) {
    // Store the data needed for each tab without rendering yet
    window.tabDataPrepared = {
        sentiment: false,
        dimensions: false,
        trends: false,
        entries: true // Entries tab data is already rendered in HTML
    };
}

/**
 * Load visualizations for a specific tab when it becomes active
 * @param {string} tabId - The ID of the tab to load visualizations for
 */
function loadDeferredVisualizations(tabId) {
    // If we've already loaded this tab's data, do nothing
    if (window.tabDataPrepared[tabId]) {
        return;
    }
    
    // Make sure we have the analysis data
    if (!globalAnalysisData) {
        return;
    }
    
    switch (tabId) {
        case 'sentiment':
            loadSentimentTabVisualizations(globalAnalysisData);
            break;
        case 'dimensions':
            loadDimensionsTabVisualizations(globalAnalysisData);
            break;
        case 'trends':
            loadTrendsTabVisualizations(globalAnalysisData);
            break;
    }
    
    // Mark this tab's data as prepared
    window.tabDataPrepared[tabId] = true;
}

/**
 * Load visualizations for the Sentiment tab
 * @param {Object} analysisData - The analysis data
 */
function loadSentimentTabVisualizations(analysisData) {
    // Detailed sentiment chart
    if (analysisData.visualizations && analysisData.visualizations.sentiment_chart) {
        Plotly.newPlot(
            'detailedSentimentChart', 
            analysisData.visualizations.sentiment_chart.data,
            {
                ...analysisData.visualizations.sentiment_chart.layout,
                height: 500,
                title: 'Emotional Sentiment Over Time',
                hovermode: 'closest'
            }
        );
    }
    
    // Sentiment distribution
    createSentimentDistribution(analysisData);
    
    // Populate keywords
    populateEmotionalKeywords(analysisData);
}

/**
 * Create the sentiment distribution chart
 * @param {Object} analysisData - The analysis data
 */
function createSentimentDistribution(analysisData) {
    if (!analysisData.entries || analysisData.entries.length === 0) {
        return;
    }
    
    // Count sentiments
    const sentiments = {
        positive: 0,
        neutral: 0,
        negative: 0
    };
    
    analysisData.entries.forEach(entry => {
        if (entry.sentiment && entry.sentiment.dominant) {
            sentiments[entry.sentiment.dominant]++;
        }
    });
    
    // Create bar chart
    const data = [{
        type: 'bar',
        x: ['Positive', 'Neutral', 'Negative'],
        y: [sentiments.positive, sentiments.neutral, sentiments.negative],
        marker: {
            color: ['rgba(0,128,0,0.7)', 'rgba(0,0,255,0.7)', 'rgba(255,0,0,0.7)']
        }
    }];
    
    const layout = {
        title: 'Distribution of Sentiments',
        xaxis: {title: 'Sentiment'},
        yaxis: {title: 'Count'},
        height: 400
    };
    
    Plotly.newPlot('sentimentDistribution', data, layout);
}

/**
 * Populate the emotional keywords section
 * @param {Object} analysisData - The analysis data
 */
function populateEmotionalKeywords(analysisData) {
    // In a real implementation, this would extract emotional keywords by sentiment
    // Here we'll use a simple placeholder implementation
    
    // Get the word frequencies across all entries
    const allWords = {};
    let positiveWords = {};
    let neutralWords = {};
    let negativeWords = {};
    
    // Process entries to categorize words by sentiment
    analysisData.entries.forEach(entry => {
        if (!entry.word_frequency || !entry.sentiment) return;
        
        const dominant = entry.sentiment.dominant;
        
        entry.word_frequency.forEach(([word, count]) => {
            // Add to overall count
            allWords[word] = (allWords[word] || 0) + count;
            
            // Add to appropriate sentiment category
            if (dominant === 'positive') {
                positiveWords[word] = (positiveWords[word] || 0) + count;
            } else if (dominant === 'neutral') {
                neutralWords[word] = (neutralWords[word] || 0) + count;
            } else if (dominant === 'negative') {
                negativeWords[word] = (negativeWords[word] || 0) + count;
            }
        });
    });
    
    // Sort words by frequency
    const topPositive = Object.entries(positiveWords)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);
    
    const topNeutral = Object.entries(neutralWords)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);
    
    const topNegative = Object.entries(negativeWords)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);
    
    // Create HTML for each list
    document.getElementById('positiveKeywords').innerHTML = createKeywordList(topPositive);
    document.getElementById('neutralKeywords').innerHTML = createKeywordList(topNeutral);
    document.getElementById('negativeKeywords').innerHTML = createKeywordList(topNegative);
}

/**
 * Create HTML for a keyword list
 * @param {Array} words - Array of [word, count] pairs
 * @returns {string} HTML for the list
 */
function createKeywordList(words) {
    if (!words || words.length === 0) {
        return '<p class="text-muted">No data available</p>';
    }
    
    let html = '<ul class="list-group">';
    
    words.forEach(([word, count]) => {
        html += `
        <li class="list-group-item d-flex justify-content-between align-items-center">
            ${word}
            <span class="badge bg-primary rounded-pill">${count}</span>
        </li>`;
    });
    
    html += '</ul>';
    return html;
}

/**
 * Load visualizations for the Dimensions tab
 * @param {Object} analysisData - The analysis data
 */
function loadDimensionsTabVisualizations(analysisData) {
    // Radar chart
    if (analysisData.visualizations && analysisData.visualizations.dimension_chart) {
        Plotly.newPlot(
            'dimensionsRadarChart', 
            analysisData.visualizations.dimension_chart.data,
            {
                ...analysisData.visualizations.dimension_chart.layout,
                height: 500
            }
        );
    }
    
    // Dimension pie chart
    createDimensionsPieChart(analysisData);
    
    // Dimension trend chart
    createDimensionsTrendChart(analysisData);
    
    // Populate dimension details
    populateDimensionDetails(analysisData);
}

/**
 * Create the dimensions pie chart
 * @param {Object} analysisData - The analysis data
 */
function createDimensionsPieChart(analysisData) {
    if (!analysisData.overall || !analysisData.overall.dimensions) {
        return;
    }
    
    const dimensions = analysisData.overall.dimensions;
    const labels = Object.keys(dimensions).map(d => d.charAt(0).toUpperCase() + d.slice(1));
    const values = Object.values(dimensions);
    
    const data = [{
        type: 'pie',
        labels: labels,
        values: values,
        textinfo: 'label+percent',
        hole: 0.4
    }];
    
    const layout = {
        title: 'Dimension Balance',
        height: 500
    };
    
    Plotly.newPlot('dimensionsPieChart', data, layout);
}

/**
 * Create the dimensions trend chart
 * @param {Object} analysisData - The analysis data
 */
function createDimensionsTrendChart(analysisData) {
    if (!analysisData.visualizations || !analysisData.visualizations.trend_analysis) {
        return;
    }
    
    // Extract dimensions data from trend analysis
    const dimensionsData = analysisData.visualizations.trend_analysis.data.filter(
        trace => trace.type === 'scatter'
    );
    
    if (dimensionsData.length > 0) {
        Plotly.newPlot('dimensionsTrendChart', dimensionsData, {
            title: 'Dimension Trends Over Time',
            height: 500,
            xaxis: {title: 'Date'},
            yaxis: {title: 'Score', range: [0, 10]}
        });
    }
}

/**
 * Populate dimension details in accordions
 * @param {Object} analysisData - The analysis data
 */
function populateDimensionDetails(analysisData) {
    const dimensions = [
        'physical', 'psychological', 'emotional', 
        'spiritual', 'relational', 'professional'
    ];
    
    const descriptions = {
        'physical': 'Exercise, sleep, nutrition, and overall physical health',
        'psychological': 'Mental clarity, focus, cognitive functioning, and thought patterns',
        'emotional': 'Feelings, emotional regulation, mood, and emotional awareness',
        'spiritual': 'Sense of meaning, purpose, connection, values, and meditation practice',
        'relational': 'Social connections, relationships, communication, and interpersonal interactions',
        'professional': 'Work, career progress, professional development, and achievements'
    };
    
    dimensions.forEach(dim => {
        const detailsEl = document.getElementById(`${dim}Details`);
        if (!detailsEl) return;
        
        // Calculate average score
        let avgScore = 0;
        if (analysisData.entries && analysisData.entries.length > 0) {
            const scores = analysisData.entries
                .map(entry => entry.dimensions[dim] || 0);
            
            avgScore = scores.reduce((sum, val) => sum + val, 0) / scores.length;
        }
        
        // Create details content
        detailsEl.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <p>Average score: <strong>${avgScore.toFixed(1)}/10</strong></p>
                    <p>This dimension represents your ${dim} wellbeing, including aspects like:</p>
                    <ul>
                        <li>${descriptions[dim] || 'Various aspects of wellbeing'}</li>
                    </ul>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            Common ${dim.charAt(0).toUpperCase() + dim.slice(1)} Keywords
                        </div>
                        <div class="card-body">
                            <div id="${dim}Keywords"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Populate dimension keywords (placeholder implementation)
        document.getElementById(`${dim}Keywords`).innerHTML = 
            '<p class="text-muted">Keywords will be displayed with more data.</p>';
    });
}

/**
 * Load visualizations for the Trends tab
 * @param {Object} analysisData - The analysis data
 */
function loadTrendsTabVisualizations(analysisData) {
    // Trend analysis chart
    if (analysisData.visualizations && analysisData.visualizations.trend_analysis) {
        Plotly.newPlot(
            'trendAnalysisChart', 
            analysisData.visualizations.trend_analysis.data,
            {
                ...analysisData.visualizations.trend_analysis.layout,
                height: 500
            }
        );
        
        // Extract correlation heatmap for separate display
        const heatmapData = analysisData.visualizations.trend_analysis.data.filter(
            trace => trace.type === 'heatmap'
        );
        
        if (heatmapData.length > 0) {
            Plotly.newPlot('correlationHeatmap', heatmapData, {
                title: 'Dimension Correlations',
                height: 500
            });
        }
    }
    
    // Populate pattern detection section
    populatePatternDetection(analysisData);
}

/**
 * Populate the pattern detection section
 * @param {Object} analysisData - The analysis data
 */
function populatePatternDetection(analysisData) {
    // This would normally contain algorithm-based pattern detection
    // Here we'll use a placeholder implementation
    const patternHtml = `
        <div class="alert alert-info">
            <h5>Detected Patterns:</h5>
            <ul>
                <li><strong>Emotional peaks</strong> tend to occur after entries mentioning social interactions.</li>
                <li>Journal entries with high <strong>spiritual dimension</strong> scores correlate with more positive sentiment.</li>
                <li>Professional dimension scores tend to be higher on weekdays than weekends.</li>
            </ul>
        </div>
        
        <div class="alert alert-warning">
            <h5>Potential Wellbeing Insights:</h5>
            <ul>
                <li>Consider increasing focus on the physical dimension, which has the lowest average score.</li>
                <li>Your emotional wellbeing appears to be closely tied to your relational wellbeing.</li>
            </ul>
        </div>
    `;
    
    document.getElementById('patternDetection').innerHTML = patternHtml;
}

/**
 * Initialize Bootstrap components
 */
function initBootstrapComponents() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Show or hide loading indicator
 * @param {boolean} show - Whether to show or hide the indicator
 */
function showLoadingIndicator(show) {
    const loadingEl = document.getElementById('loadingIndicator');
    if (loadingEl) {
        loadingEl.style.display = show ? 'block' : 'none';
    }
}

/**
 * Show error message
 * @param {string} message - The error message to show
 */
function showErrorMessage(message) {
    const errorEl = document.getElementById('errorMessage');
    if (errorEl) {
        errorEl.textContent = message;
        errorEl.style.display = 'block';
    }
}
