#pdf_extractor.py

def extract_text_from_html(html_path):
    """
    Extract text from an HTML file.
    
    Args:
        html_path (str): Path to the HTML file
        
    Returns:
        str: Extracted text from the HTML file
    """
    try:
        logger.info(f"Extracting text from HTML: {html_path}")
        
        # First, detect the encoding
        with open(html_path, 'rb') as f:
            rawdata = f.read()
            result = chardet.detect(rawdata)
            encoding = result['encoding'] or 'utf-8'
            logger.info(f"Detected encoding: {encoding} with confidence {result['confidence']}")
        
        # Read the file with the detected encoding
        with open(html_path, 'r', encoding=encoding, errors='replace') as f:
            html_content = f.read()
        
        # Parse the HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        
        # Get all text without HTML tags
        text = soup.get_text()
        
        # Clean up the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        logger.info(f"Successfully extracted {len(text)} characters from HTML file")
        
        # Get date from filename if available
        filename = os.path.basename(html_path)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if date_match:
            date_str = date_match.group(1)
            logger.info(f"Extracted date from filename: {date_str}")
        
        return text
    
    except Exception as e:
        logger.error(f"Error extracting text from HTML {html_path}: {str(e)}")
        return ""
        
import os
import re
import pdfplumber
import logging
from datetime import datetime
from langdetect import detect
import chardet
import tempfile

# Add OCR imports
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# Add HTML processing imports
from bs4 import BeautifulSoup
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
import re
from bs4 import BeautifulSoup

def extract_text_from_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.pdf', '.txt']:
        # Your existing logic for PDFs and TXT files
        return extract_text_from_non_html(filepath)
    elif ext in ['.html', '.htm']:
        with open(filepath, encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            # Remove common distraction elements
            for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
                tag.decompose()
            # If your journal entries are wrapped in a specific container (for example, an <article> tag or a <div> with a specific class), extract that:
            main_content = soup.find('article')
            if not main_content:
                # Optionally, if your sample HTML journal entry is structured with a custom container, use that:
                main_content = soup.find('div', class_='journal-entry')
            # Fallback to the <body> tag if no specific container is found
            if not main_content:
                main_content = soup.body
            # Extract text and clean extra whitespace
            if main_content:
                text = main_content.get_text(separator='\n')
            else:
                text = soup.get_text(separator='\n')
            text = re.sub(r'\n+', '\n', text).strip()
            return text
    return ""


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file with enhanced extraction including OCR.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        logger.info(f"Extracting text from PDF: {pdf_path}")
        text_content = ""
        
        # First try regular text extraction
        with pdfplumber.open(pdf_path) as pdf:
            logger.info(f"PDF opened successfully. Pages: {len(pdf.pages)}")
            
            for i, page in enumerate(pdf.pages):
                # Try different extraction methods for better results
                page_text = page.extract_text(
                    x_tolerance=3,  # More tolerant horizontal text grouping
                    y_tolerance=3,  # More tolerant vertical text grouping
                    layout=True,    # Use layout analysis
                    keep_blank_chars=False,
                    use_text_flow=True,
                    horizontal_ltr=True,
                ) or ""
                
                # If still no text, try with different settings
                if not page_text.strip():
                    page_text = page.extract_text(
                        x_tolerance=5,
                        y_tolerance=5,
                        layout=False,
                        keep_blank_chars=True,
                    ) or ""
                
                logger.info(f"Page {i+1}: Extracted {len(page_text)} characters via pdfplumber")
                text_content += page_text + "\n\n"
        
        # If regular extraction didn't yield much text, try OCR
        if len(text_content.strip()) < 100:
            logger.info("Limited text found. Trying OCR extraction...")
            ocr_text = extract_text_with_ocr(pdf_path)
            if ocr_text:
                text_content = ocr_text
        
        # Clean up the text
        text_content = clean_extracted_text(text_content)
        
        # Verify that we've extracted Chinese text
        if text_content.strip():
            try:
                lang = detect(text_content)
                if lang != 'zh-cn' and lang != 'zh-tw' and lang != 'zh':
                    logger.warning(f"Detected language is {lang}, not Chinese. Document may not be in Chinese.")
            except Exception as e:
                logger.warning(f"Language detection failed: {str(e)}")
        
        return text_content.strip()
    
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        # If regular extraction fails, fall back to OCR
        try:
            logger.info("Attempting OCR after failed regular extraction...")
            return extract_text_with_ocr(pdf_path)
        except Exception as ocr_error:
            logger.error(f"OCR extraction also failed: {str(ocr_error)}")
            return ""

def extract_text_with_ocr(pdf_path):
    """
    Extract text from PDF using OCR (Optical Character Recognition).
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF using OCR
    """
    try:
        logger.info(f"Starting OCR process for {pdf_path}")
        
        # Create a temporary directory to store images
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Created temporary directory for OCR images: {temp_dir}")
            
            # Convert PDF to images
            logger.info("Converting PDF to images...")
            images = convert_from_path(
                pdf_path,
                dpi=300,
                output_folder=temp_dir,
                fmt='jpg',
                output_file=f"page"
            )
            
            logger.info(f"Converted {len(images)} pages to images, now performing OCR...")
            
            # Perform OCR on each image
            text_content = ""
            for i, image in enumerate(images):
                # Optimize image for OCR if needed
                image = image.convert('L')  # Convert to grayscale
                
                # Perform OCR with pytesseract
                # Use Chinese and English language data to ensure both can be recognized
                ocr_result = pytesseract.image_to_string(
                    image,
                    lang='chi_sim+eng',
                    config='--psm 3 --oem 3'
                )
                
                logger.info(f"Extracted {len(ocr_result)} characters from page {i+1} using OCR")
                text_content += ocr_result + "\n\n"
            
            return clean_extracted_text(text_content)
    
    except Exception as e:
        logger.error(f"Error in OCR extraction: {str(e)}")
        return ""

def extract_text_from_txt(txt_path):
    """
    Extract text from a TXT file with encoding detection.
    
    Args:
        txt_path (str): Path to the TXT file
        
    Returns:
        str: Extracted text from the TXT file
    """
    try:
        logger.info(f"Extracting text from TXT: {txt_path}")
        
        # First, detect the encoding
        with open(txt_path, 'rb') as f:
            rawdata = f.read()
            result = chardet.detect(rawdata)
            encoding = result['encoding'] or 'utf-8'
            logger.info(f"Detected encoding: {encoding} with confidence {result['confidence']}")
        
        # Now read the file with the detected encoding
        with open(txt_path, 'r', encoding=encoding, errors='replace') as f:
            text_content = f.read()
        
        # Clean up the text
        text_content = clean_extracted_text(text_content)
        
        logger.info(f"Successfully extracted {len(text_content)} characters from TXT file")
        return text_content.strip()
    
    except Exception as e:
        logger.error(f"Error extracting text from TXT {txt_path}: {str(e)}")
        return ""

def clean_extracted_text(text):
    """
    Clean up extracted text by fixing common issues.
    
    Args:
        text (str): Raw extracted text
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Replace repeated newlines with a single newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Fix broken words (words split by newlines)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    
    # Remove weird control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    return text

def split_journal_into_entries(full_text):
    """
    Split a journal text into individual entries based on date patterns.
    
    Args:
        full_text (str): The full journal text
        
    Returns:
        list: List of dictionaries with 'date' and 'text' for each entry
    """
    # Pattern to match dates like "Monday, February 3, 2025" or similar variations
    date_pattern = r'(Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday),\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}'
    
    # Also match Chinese date patterns like "2023年10月15日" (with or without weekday)
    chinese_date_pattern = r'((星期[一二三四五六日]|周[一二三四五六日])?\s*\d{4}年\d{1,2}月\d{1,2}日)'
    
    # Also match ISO date format YYYY-MM-DD with optional day of week
    iso_date_pattern = r'(\d{4}-\d{2}-\d{2})'
    
    # Combine patterns
    combined_pattern = f"{date_pattern}|{chinese_date_pattern}|{iso_date_pattern}"
    
    # Find all matches of the date pattern
    matches = list(re.finditer(combined_pattern, full_text, re.IGNORECASE))
    
    entries = []
    
    # If no matches found, treat the whole text as a single entry
    if not matches:
        logger.warning("No date patterns found. Treating the whole document as a single entry.")
        return [{'date': extract_date_from_text(full_text) or datetime.now().strftime("%Y-%m-%d"), 'text': full_text}]
    
    # Process each entry
    for i, match in enumerate(matches):
        start_idx = match.start()
        
        # Extract the date string
        date_str = match.group(0)
        parsed_date = parse_date_string(date_str)
        
        # Determine the end of this entry (start of next entry or end of text)
        if i < len(matches) - 1:
            end_idx = matches[i + 1].start()
        else:
            end_idx = len(full_text)
        
        # Extract the entry text
        entry_text = full_text[start_idx:end_idx].strip()
        
        entries.append({
            'date': parsed_date,
            'text': entry_text
        })
    
    return entries

def parse_date_string(date_str):
    """
    Parse different date formats to YYYY-MM-DD format.
    
    Args:
        date_str (str): Date string
        
    Returns:
        str: Date in YYYY-MM-DD format
    """
    try:
        # Western format: "Monday, February 3, 2025"
        if re.match(r'(Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday),\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}', date_str, re.IGNORECASE):
            try:
                return datetime.strptime(date_str, "%A, %B %d, %Y").strftime("%Y-%m-%d")
            except:
                # Try without weekday
                match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(\d{4})', date_str, re.IGNORECASE)
                if match:
                    month, day, year = match.groups()
                    return datetime.strptime(f"{month} {day} {year}", "%B %d %Y").strftime("%Y-%m-%d")
        
        # Chinese format: "2023年10月15日" or "星期一 2023年10月15日"
        if "年" in date_str and "月" in date_str and "日" in date_str:
            match = re.search(r'\d{4}年\d{1,2}月\d{1,2}日', date_str)
            if match:
                date_part = match.group(0)
                year = re.search(r'(\d{4})年', date_part).group(1)
                month = re.search(r'年(\d{1,2})月', date_part).group(1)
                day = re.search(r'月(\d{1,2})日', date_part).group(1)
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        
        # ISO format: "2023-10-15"
        if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
            return date_str
    
    except Exception as e:
        logger.error(f"Error parsing date '{date_str}': {str(e)}")
    
    # Default to current date if parsing fails
    return datetime.now().strftime("%Y-%m-%d")

def extract_date_from_text(text):
    """
    Try to extract date information from text content.
    
    Args:
        text (str): Text content
        
    Returns:
        str: Extracted date or empty string
    """
    if not text:
        return ""
    
    # Look for various date formats
    patterns = [
        r'(\d{4})年(\d{1,2})月(\d{1,2})日',  # Chinese format: 2023年10月15日
        r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})',  # ISO format: 2023-10-15 or 2023/10/15
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s+(\d{4})'  # English format: October 15, 2023
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            if pattern == patterns[0]:  # Chinese format
                year, month, day = match.groups()
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            elif pattern == patterns[1]:  # ISO format
                year, month, day = match.groups()
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            elif pattern == patterns[2]:  # English format
                month, day, year = match.groups()
                month_num = {"January": "01", "February": "02", "March": "03", "April": "04",
                             "May": "05", "June": "06", "July": "07", "August": "08",
                             "September": "09", "October": "10", "November": "11", "December": "12"}[month]
                return f"{year}-{month_num}-{day.zfill(2)}"
    
    return ""
