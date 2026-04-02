"""
Batch Preprocessing Script for All Xiao Dataset TSV Files
Processes all .tsv files in xiao-2016-2019 and xiao-2020-2023 directories
Adds year, month, source_file columns
Applies text cleaning and saves as single CSV file
"""

import pandas as pd
import re
import os
import emoji
import glob
from datetime import datetime

def preprocess_weibo_text(text):
    """Normalize a single Weibo post text (same logic as preprocess.py)"""
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

def extract_date_from_filename(filename):
    """Extract year and month from filename like '2016-01-01.tsv'"""
    basename = os.path.basename(filename)
    date_str = os.path.splitext(basename)[0]  # Remove .tsv extension
    try:
        year, month, day = date_str.split('-')
        return int(year), int(month)
    except:
        return None, None

def process_single_tsv(file_path):
    """Process a single TSV file and return DataFrame with additional columns"""
    print(f"Processing: {os.path.basename(file_path)}")
    
    try:
        # Read the TSV file
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        
        # Extract year and month from filename
        year, month = extract_date_from_filename(file_path)
        
        # Add metadata columns
        df['year'] = year
        df['month'] = month
        df['source_file'] = os.path.basename(file_path)
        
        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Clean text content
        df['cleaned_content'] = df['weibo_content'].apply(preprocess_weibo_text)
        df['cleaned_r_content'] = df['r_weibo_content'].apply(preprocess_weibo_text)
        
        # Combine into cleaned_text (similar to full_text in preprocess.py)
        df['cleaned_text'] = df['cleaned_content'] + ' ' + df['cleaned_r_content']
        df['cleaned_text'] = df['cleaned_text'].str.strip()
        
        # Remove rows with empty cleaned text
        df = df[df['cleaned_text'].str.len() > 0].copy()
        
        print(f"  → Loaded {len(df)} posts from {os.path.basename(file_path)}")
        return df
        
    except Exception as e:
        print(f"  → Error processing {file_path}: {str(e)}")
        return None

def batch_process_xiao_dataset():
    """Process all TSV files in both new directories and combine them"""
    print("="*80)
    print("BATCH PREPROCESSING NEW DATASET")
    print("="*80)
    
    # Get all TSV files from both directories
    pattern1 = "new-2016-2019/*.tsv"
    pattern2 = "new-2020-2025/*.tsv"
    
    tsv_files = glob.glob(pattern1) + glob.glob(pattern2)
    tsv_files.sort()  # Sort chronologically
    
    print(f"Found {len(tsv_files)} TSV files to process:")
    for f in tsv_files:
        print(f"  - {os.path.basename(f)}")
    
    if not tsv_files:
        print("No TSV files found! Check directory paths.")
        return
    
    print("\\nStarting batch processing...")
    
    # Process each file and collect DataFrames
    all_dataframes = []
    total_posts = 0
    successful_files = 0
    
    for file_path in tsv_files:
        df = process_single_tsv(file_path)
        if df is not None:
            all_dataframes.append(df)
            total_posts += len(df)
            successful_files += 1
    
    if not all_dataframes:
        print("No data frames to combine!")
        return
    
    print("\\n" + "="*80)
    print("COMBINING DATA")
    print("="*80)
    
    # Concatenate all DataFrames
    print("Combining all DataFrames...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Simple deduplication based on cleaned_text hash
    print("Removing duplicates...")
    combined_df['content_hash'] = combined_df['cleaned_text'].apply(lambda x: hash(x))
    before_dedup = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['content_hash']).copy()
    after_dedup = len(combined_df)
    
    # Sort by time
    combined_df = combined_df.sort_values('time').reset_index(drop=True)
    
    # Add useful metadata
    combined_df['text_length'] = combined_df['cleaned_text'].str.len()
    combined_df['has_repost'] = ~combined_df['r_weibo_content'].isna()
    
    # Check for 孝 keyword
    combined_df['contains_xiao'] = combined_df['cleaned_text'].str.contains('孝', na=False)
    
    print("\\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save the final dataset
    output_file = "weibo_xiao_new_cleaned.csv"
    print(f"Saving to {output_file}...")
    
    # Select columns to save (keeping essential ones)
    columns_to_save = [
        'weibo_id', 'time', 'year', 'month', 'source_file',
        'weibo_content', 'r_weibo_content', 
        'cleaned_text', 'text_length', 'has_repost',
        'user_id', 'contains_xiao'
    ]
    
    # Only include columns that exist
    available_columns = [col for col in columns_to_save if col in combined_df.columns]
    final_df = combined_df[available_columns].copy()
    
    final_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print("\\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Files processed successfully: {successful_files}/{len(tsv_files)}")
    print(f"Total posts before deduplication: {before_dedup:,}")
    print(f"Total posts after deduplication: {after_dedup:,}")
    print(f"Posts containing '孝': {final_df['contains_xiao'].sum():,}")
    print(f"Date range: {final_df['time'].min()} to {final_df['time'].max()}")
    print(f"Average text length: {final_df['text_length'].mean():.1f} characters")
    print(f"Posts with reposts: {final_df['has_repost'].sum():,}")
    
    print(f"\\nYear distribution:")
    print(final_df['year'].value_counts().sort_index())
    
    print(f"\\nOutput saved as: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    
    return final_df

if __name__ == "__main__":
    df = batch_process_xiao_dataset()
    print("\\nBatch preprocessing completed!")