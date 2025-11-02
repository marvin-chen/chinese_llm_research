"""
Weibo Data Preprocessing Script
"""

import pandas as pd
import re
from datetime import datetime

def preprocess_weibo_text(text):
    """
    Clean Weibo text by removing emojis, mentions, and other artifacts
    while preserving Chinese characters.

    Args:
        text (str): Raw Weibo text
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # 1. Remove Weibo-specific emoji markers (e.g., [偷乐], [打call])
    text = re.sub(r'\[[^\]]+\]', '', text)

    # 2. Remove Unicode emoji characters (excluding CJK ranges)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U0001F900-\U0001F9FF"  # supplemental symbols
        u"\U0001FA00-\U0001FA6F"  # chess symbols
        u"\U00002600-\U000026FF"  # miscellaneous symbols
        u"\U00002700-\U000027BF"  # dingbats
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)

    # 3. Remove hashtags but keep content
    text = re.sub(r'#([^#]+)#', r'\1', text)

    # 4. Remove @ mentions
    text = re.sub(r'@[^\s]+', '', text)

    # 5. Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)

    # 6. Remove decorative symbols
    text = re.sub(r'[↓△※☆…]+', '', text)

    # 7. Normalize whitespace (including full-width spaces)
    text = re.sub(r'[\s\u3000]+', ' ', text)

    # 8. Strip leading/trailing whitespace
    text = text.strip()

    return text

def analyze_keyword_presence(df, keyword='孝'):
    """
    Analyze the presence and context of a keyword in the dataset.

    Args:
        df (pd.DataFrame): DataFrame with 'full_text' column
        keyword (str): Chinese character to search for

    Returns:
        pd.DataFrame: DataFrame with analysis columns added
    """
    df_analysis = df.copy()

    # Check if keyword is present
    df_analysis['contains_keyword'] = df_analysis['full_text'].str.contains(keyword, na=False)

    # Extract compounds containing the keyword
    # Pattern: 0-1 character before + keyword + 0-2 characters after
    pattern = f'[\u4e00-\u9fff]{{0,1}}{keyword}[\u4e00-\u9fff]{{0,2}}'
    df_analysis['keyword_compounds'] = df_analysis['full_text'].apply(
        lambda x: ', '.join(set(re.findall(pattern, str(x)))) if pd.notna(x) else ''
    )

    # Count keyword occurrences
    df_analysis['keyword_count'] = df_analysis['full_text'].str.count(keyword)

    return df_analysis

def filter_false_positives(df, keyword='孝', exclusion_list=None):
    """
    Filter out false positives where keyword appears in names, brands, etc.

    Args:
        df (pd.DataFrame): DataFrame with keyword analysis
        keyword (str): The keyword being analyzed
        exclusion_list (list): List of terms to exclude (e.g., names)

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    if exclusion_list is None:
        # Default exclusion list for 孝
        exclusion_list = [
            '天野喜孝',  # Artist name
            '林孝埈',    # Athlete name
            '喜孝',      # Part of 天野喜孝
        ]

    df_filtered = df.copy()

    # Create exclusion mask
    exclusion_mask = pd.Series([False] * len(df_filtered), index=df_filtered.index)

    for term in exclusion_list:
        exclusion_mask |= df_filtered['full_text'].str.contains(term, na=False)

    # Apply filter
    df_filtered['is_false_positive'] = exclusion_mask
    df_filtered = df_filtered[~exclusion_mask].copy()

    return df_filtered

def preprocess_weibo_dataset(input_file, output_file, keyword='孝'):
    """
    Complete preprocessing pipeline for Weibo dataset.

    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save processed CSV
        keyword (str): Keyword to analyze

    Returns:
        pd.DataFrame: Processed DataFrame
    """
    print("Loading data...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} posts")

    print("\nPreprocessing text...")
    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'])

    # Clean text content
    df['cleaned_content'] = df['weibo_content'].apply(preprocess_weibo_text)
    df['cleaned_r_content'] = df['r_weibo_content'].apply(preprocess_weibo_text)

    # Combine original and repost content
    df['full_text'] = df['cleaned_content'] + ' ' + df['cleaned_r_content']
    df['full_text'] = df['full_text'].str.strip()

    # Remove exact duplicates
    df['content_hash'] = df['full_text'].apply(lambda x: hash(x))
    df = df.drop_duplicates(subset=['content_hash']).copy()
    print(f"After deduplication: {len(df)} posts")

    # Filter out empty posts
    df = df[df['full_text'].str.len() > 0].copy()
    print(f"After removing empty posts: {len(df)} posts")

    # Add metadata
    df['text_length'] = df['full_text'].str.len()
    df['has_repost'] = ~df['r_weibo_content'].isna()

    print(f"\nAnalyzing keyword '{keyword}'...")
    df = analyze_keyword_presence(df, keyword)
    print(f"Posts containing '{keyword}': {df['contains_keyword'].sum()}")

    print(f"\nFiltering false positives...")
    df_relevant = df[df['contains_keyword']].copy()
    df_relevant = filter_false_positives(df_relevant, keyword)
    print(f"Posts after filtering: {len(df_relevant)}")

    # Get all non-keyword posts for comparison
    df_no_keyword = df[~df['contains_keyword']].copy()

    # Combine back
    df_final = pd.concat([df_relevant, df_no_keyword], ignore_index=True)
    df_final = df_final.sort_values('time', ascending=False).reset_index(drop=True)

    print(f"\nSaving to {output_file}...")
    df_final.to_csv(output_file, index=False, encoding='utf-8')
    print("Done!")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total posts: {len(df_final)}")
    print(f"Posts with keyword: {df_final['contains_keyword'].sum()}")
    print(f"Posts with reposts: {df_final['has_repost'].sum()}")
    print(f"Average text length: {df_final['text_length'].mean():.1f} characters")
    print(f"Date range: {df_final['time'].min()} to {df_final['time'].max()}")

    # Show common compounds
    relevant_posts = df_final[df_final['contains_keyword'] & ~df_final['is_false_positive']]
    if len(relevant_posts) > 0:
        print(f"\nMost common compounds containing '{keyword}':")
        all_compounds = relevant_posts['keyword_compounds'].str.split(', ').explode()
        print(all_compounds.value_counts().head(10))

    return df_final

if __name__ == "__main__":
    # Example usage
    input_file = "weibo_raw_data.csv"
    output_file = "weibo_preprocessed.csv"
    keyword = "孝"  # filial piety

    df_processed = preprocess_weibo_dataset(input_file, output_file, keyword)

    # Save posts with keyword to separate file for annotation
    df_keyword = df_processed[
        df_processed['contains_keyword'] & 
        ~df_processed['is_false_positive']
    ].copy()

    if len(df_keyword) > 0:
        annotation_file = f"weibo_for_annotation_{keyword}.csv"
        df_keyword.to_csv(annotation_file, index=False, encoding='utf-8')
        print(f"\nPosts for annotation saved to: {annotation_file}")
        print(f"Total posts to annotate: {len(df_keyword)}")
