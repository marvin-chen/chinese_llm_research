"""
Preprocess Ground Truth CSV
Cleans text and prepares it for LLM annotation
"""

import pandas as pd
import re
import emoji

def preprocess_weibo_text(text):
    if text is None or (isinstance(text, float) and str(text) == "nan"):
        return ""

    text = str(text)

    # Weibo bracket tokens like [偷乐]
    text = re.sub(r'\[[^\]]+\]', '', text)

    # Remove all emoji (handles keycaps etc.)
    text = emoji.replace_emoji(text, replace="")

    # Keep hashtag content
    text = re.sub(r'#([^#]+)#', r'\1', text)

    # Remove mentions and URLs
    text = re.sub(r'@[^\s]+', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)

    # Normalize whitespace
    text = re.sub(r'[\s\u3000]+', ' ', text)
    return text.strip()

def analyze_keyword_presence(df, keyword='孝'):
    """Extract keyword compounds from text"""
    df_analysis = df.copy()
    
    df_analysis['contains_keyword'] = df_analysis['cleaned_text'].str.contains(keyword, na=False)
    
    # Extract compounds around the keyword
    pattern = f'[\u4e00-\u9fff]{{0,1}}{keyword}[\u4e00-\u9fff]{{0,2}}'
    df_analysis['keyword_compounds'] = df_analysis['cleaned_text'].apply(
        lambda x: ', '.join(set(re.findall(pattern, str(x)))) if pd.notna(x) else ''
    )
    
    df_analysis['keyword_count'] = df_analysis['cleaned_text'].str.count(keyword)
    return df_analysis

def preprocess_ground_truth(input_file, output_file, keyword='孝'):
    """
    Preprocess the ground truth CSV file
    
    Expected input columns:
    - Column 0: original text (微博内容)
    - Column 1: manual_relevant (TRUE/FALSE)
    - Column 2: manual_sentiment (score)
    - Column 3: manual_reasoning (理由)
    """
    print(f"Loading ground truth from {input_file}...")
    
    # Read CSV - adjust column names based on your actual file
    df = pd.read_csv(input_file, encoding='utf-8')
    
    # Detect column names (your file may not have headers)
    if df.columns[0].startswith('Unnamed') or len(df.columns) == 4:
        # No header row detected
        df.columns = ['original_text', 'manual_relevant', 'manual_sentiment', 'manual_reasoning']
    
    print(f"Loaded {len(df)} ground truth posts")
    
    # Clean the text
    print("Cleaning text...")
    df['cleaned_text'] = df['original_text'].apply(preprocess_weibo_text)
    df['text_length'] = df['cleaned_text'].str.len()
    
    # Analyze keyword presence
    print(f"Analyzing keyword '{keyword}'...")
    df = analyze_keyword_presence(df, keyword)
    
    # Convert manual_relevant to boolean
    df['manual_relevant'] = df['manual_relevant'].map({
        'TRUE': True, 'True': True, 'true': True, True: True,
        'FALSE': False, 'False': False, 'false': False, False: False
    })
    
    # Ensure sentiment is numeric
    df['manual_sentiment'] = pd.to_numeric(df['manual_sentiment'], errors='coerce')
    
    # Remove any completely empty rows
    df = df[df['cleaned_text'].str.len() > 0].copy()
    
    print(f"\nSaving to {output_file}...")
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print("\n" + "="*80)
    print("GROUND TRUTH PREPROCESSING COMPLETE")
    print("="*80)
    print(f"Total posts: {len(df)}")
    print(f"Posts with keyword: {df['contains_keyword'].sum()}")
    print(f"Manually marked relevant: {df['manual_relevant'].sum()}")
    print(f"Average text length: {df['text_length'].mean():.1f} characters")
    
    # Show sentiment distribution
    print("\nManual Sentiment Distribution:")
    sentiment_counts = df['manual_sentiment'].value_counts().sort_index()
    for sentiment, count in sentiment_counts.items():
        label = {-2: "Strong Negative", -1: "Mild Negative", 
                 0: "Neutral/Irrelevant", 1: "Mild Positive", 
                 2: "Strong Positive"}.get(sentiment, "Unknown")
        print(f"  {int(sentiment):+2d} ({label}): {count}")
    
    return df

if __name__ == "__main__":
    # Update these paths to match your files
    INPUT_FILE = "ground_truth.csv"
    OUTPUT_FILE = "ground_truth_preprocessed.csv"
    KEYWORD = "孝"
    
    df_processed = preprocess_ground_truth(INPUT_FILE, OUTPUT_FILE, KEYWORD)
    
    print(f"\nPreprocessed ground truth saved to: {OUTPUT_FILE}")
    print("Next step: Run analyze_ground_truth.py to get LLM predictions")
