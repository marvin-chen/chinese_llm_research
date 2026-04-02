#!/usr/bin/env python3
"""
Batch Preprocess TSV Files from "new data" folders
Processes all TSV files organized by topic folders (女主内, 忠, 愚公移山, 男主外)
Creates separate CSV files for each topic folder with deduplication
"""

import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime

# Import preprocessing functions from existing script
sys.path.append('scripts')
from preprocess import preprocess_weibo_text

def process_topic_folder(topic_folder, output_folder='data'):
    """
    Process all TSV files from a single topic folder
    
    Args:
        topic_folder: Path object to topic folder
        output_folder: Folder to save output CSV
    
    Returns:
        DataFrame with processed posts
    """
    
    topic_name = topic_folder.name
    print(f"\n{'='*80}")
    print(f"Processing topic: {topic_name}")
    print(f"{'='*80}")
    
    # Find all TSV files in this topic folder
    tsv_files = list(topic_folder.glob("*.tsv"))
    print(f"Found {len(tsv_files)} TSV files")
    
    if len(tsv_files) == 0:
        print(f"No TSV files found in {topic_name}")
        return None
    
    # Process all TSV files for this topic
    all_dataframes = []
    total_posts = 0
    
    for i, tsv_file in enumerate(tsv_files, 1):
        try:
            # Load TSV file
            df = pd.read_csv(tsv_file, sep='\t', encoding='utf-8')
            
            # Add source file
            df['source_file'] = tsv_file.name
            
            # Parse time
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            
            # Rename columns to standard format
            if 'weibo_id' in df.columns:
                df = df.rename(columns={'weibo_id': 'post_id'})
            if 'weibo_content' in df.columns:
                df = df.rename(columns={'weibo_content': 'text'})
            if 'r_weibo_content' in df.columns:
                df = df.rename(columns={'r_weibo_content': 'repost_text'})
            
            all_dataframes.append(df)
            total_posts += len(df)
            
            if i % 20 == 0:
                print(f"  Processed {i}/{len(tsv_files)} files ({total_posts:,} posts)...")
        
        except Exception as e:
            print(f"  WARNING: Error processing {tsv_file.name}: {e}")
            continue
    
    if len(all_dataframes) == 0:
        print(f"No data loaded for {topic_name}")
        return None
    
    # Combine all files for this topic
    print(f"\nCombining {len(all_dataframes)} files...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Total posts: {len(combined_df):,}")
    
    # Clean text
    print(f"Cleaning text...")
    combined_df['cleaned_text'] = combined_df['text'].apply(preprocess_weibo_text)
    
    # Clean repost text if it exists
    if 'repost_text' in combined_df.columns:
        combined_df['cleaned_repost'] = combined_df['repost_text'].apply(preprocess_weibo_text)
        combined_df['full_text'] = combined_df['cleaned_text'] + ' ' + combined_df['cleaned_repost']
    else:
        combined_df['full_text'] = combined_df['cleaned_text']
    
    combined_df['full_text'] = combined_df['full_text'].str.strip()
    
    # Remove empty posts
    original_count = len(combined_df)
    combined_df = combined_df[combined_df['full_text'].str.len() > 0].copy()
    print(f"After removing empty posts: {len(combined_df):,} (removed {original_count - len(combined_df):,})")
    
    # Deduplicate by post_id
    print(f"Deduplicating by post_id...")
    original_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['post_id'], keep='first')
    print(f"  Removed {original_count - len(combined_df):,} duplicate post_ids")
    
    # Deduplicate by content hash
    print(f"Deduplicating by content...")
    combined_df['content_hash'] = combined_df['full_text'].apply(lambda x: hash(x))
    original_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['content_hash'], keep='first')
    print(f"  Removed {original_count - len(combined_df):,} duplicate contents")
    print(f"  Final count: {len(combined_df):,}")
    
    # Add text length
    combined_df['text_length'] = combined_df['full_text'].str.len()
    
    # Add year and month from time
    if 'time' in combined_df.columns:
        combined_df['year'] = combined_df['time'].dt.year
        combined_df['month'] = combined_df['time'].dt.month
    
    # Sort by time
    combined_df = combined_df.sort_values('time', ascending=True).reset_index(drop=True)
    
    # Save to file
    output_file = os.path.join(output_folder, f'{topic_name}_cleaned.csv')
    os.makedirs(output_folder, exist_ok=True)
    print(f"\nSaving to {output_file}...")
    combined_df.to_csv(output_file, index=False, encoding='utf-8')
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY: {topic_name}")
    print(f"{'='*80}")
    print(f"Total posts: {len(combined_df):,}")
    print(f"Average text length: {combined_df['text_length'].mean():.1f} characters")
    
    if 'time' in combined_df.columns:
        print(f"Date range: {combined_df['time'].min()} to {combined_df['time'].max()}")
    
    print(f"Saved to: {output_file}")
    
    return combined_df

def process_all_topics(base_folder='new data', output_folder='data'):
    """
    Process all topic folders and create separate CSV files for each
    
    Args:
        base_folder: Folder containing topic subfolders
        output_folder: Folder to save output CSVs
    
    Returns:
        Dictionary mapping topic names to DataFrames
    """
    
    print("="*80)
    print("BATCH TSV PREPROCESSING - SEPARATE FILES BY TOPIC")
    print("="*80)
    
    # Find all topic folders
    base_path = Path(base_folder)
    if not base_path.exists():
        print(f"ERROR: Base folder '{base_folder}' not found!")
        return None
    
    topic_folders = [f for f in base_path.iterdir() if f.is_dir() and not f.name.startswith('.')]
    
    print(f"\nFound {len(topic_folders)} topic folders:")
    for folder in topic_folders:
        print(f"  - {folder.name}")
    
    # Process each topic folder separately
    results = {}
    
    for topic_folder in topic_folders:
        df = process_topic_folder(topic_folder, output_folder)
        if df is not None:
            results[topic_folder.name] = df
    
    # Overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    
    total_posts = sum(len(df) for df in results.values())
    print(f"Topics processed: {len(results)}")
    print(f"Total posts across all topics: {total_posts:,}")
    
    print(f"\nBreakdown by topic:")
    for topic_name, df in results.items():
        print(f"  {topic_name}: {len(df):,} posts")
    
    print("\n" + "="*80)
    print("✓ All topics processed successfully!")
    print("="*80)
    print(f"\nOutput files saved to '{output_folder}/' directory:")
    for topic_name in results.keys():
        print(f"  - {topic_name}_cleaned.csv")
    
    return results

if __name__ == '__main__':
    # Process all topics, creating separate CSV for each
    results = process_all_topics(
        base_folder='new data',
        output_folder='data'
    )
