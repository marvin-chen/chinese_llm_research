#!/usr/bin/env python3
"""
Create targeted sample dataset for reasoning extraction
Includes:
1. All positive posts (sentiment +1, +2) in Family Conflict bucket
2. Diverse sample from each sentiment×bucket combination
"""

import pandas as pd
import os
from datetime import datetime

def normalize_bucket_names(df):
    """Normalize bucket names to English"""
    bucket_mapping = {
        '日常实践': 'Daily Practice',
        '责任义务': 'Obligation', 
        '家庭冲突': 'Family Conflict',
        '理论探讨': 'Theory/Critique',
        '婚恋择偶': 'Marriage/Dating',
    }
    df['qwen_bucket'] = df['qwen_bucket'].replace(bucket_mapping)
    return df

def create_targeted_sample(results_file='results/qwen_analysis_results.csv',
                          output_file='data/targeted_sample_for_reasoning.csv',
                          sample_per_combination=5):
    """
    Create targeted sample dataset
    
    Args:
        results_file: Path to full analysis results
        output_file: Path to save targeted sample
        sample_per_combination: Number of posts to sample per sentiment×bucket combination
    """
    
    print("="*60)
    print("CREATING TARGETED SAMPLE FOR REASONING EXTRACTION")
    print("="*60)
    
    # Load results
    if not os.path.exists(results_file):
        print(f"ERROR: Results file not found: {results_file}")
        return None
    
    print(f"\nLoading results from: {results_file}")
    df = pd.read_csv(results_file)
    
    # Normalize bucket names
    df = normalize_bucket_names(df)
    
    print(f"Total posts in results: {len(df):,}")
    
    # Filter to only posts with valid sentiment and bucket
    valid_df = df[(df['qwen_sentiment'].notna()) & 
                  (df['qwen_bucket'].notna()) & 
                  (df['qwen_bucket'] != '')].copy()
    
    print(f"Posts with valid sentiment and bucket: {len(valid_df):,}")
    
    # 1. Get ALL positive posts in Family Conflict
    family_conflict_positive = valid_df[
        (valid_df['qwen_bucket'] == 'Family Conflict') & 
        (valid_df['qwen_sentiment'] > 0)
    ].copy()
    
    print(f"\n{'='*60}")
    print("FAMILY CONFLICT POSITIVE POSTS:")
    print(f"{'='*60}")
    print(f"Sentiment +1: {len(family_conflict_positive[family_conflict_positive['qwen_sentiment'] == 1]):,}")
    print(f"Sentiment +2: {len(family_conflict_positive[family_conflict_positive['qwen_sentiment'] == 2]):,}")
    print(f"TOTAL: {len(family_conflict_positive):,}")
    
    # 2. Get diverse sample from each sentiment×bucket combination
    print(f"\n{'='*60}")
    print(f"DIVERSE SAMPLING ({sample_per_combination} per combination):")
    print(f"{'='*60}")
    
    sampled_posts = []
    
    # Get all unique sentiment×bucket combinations
    for sentiment in sorted(valid_df['qwen_sentiment'].unique()):
        for bucket in sorted(valid_df['qwen_bucket'].unique()):
            subset = valid_df[(valid_df['qwen_sentiment'] == sentiment) & 
                            (valid_df['qwen_bucket'] == bucket)]
            
            if len(subset) > 0:
                # Sample up to N posts from this combination
                n_sample = min(sample_per_combination, len(subset))
                sample = subset.sample(n=n_sample, random_state=42)
                sampled_posts.append(sample)
                
                print(f"  Sentiment {sentiment:+.0f} × {bucket}: {len(subset):,} total, sampled {n_sample}")
    
    # Combine all sampled posts
    diverse_sample = pd.concat(sampled_posts, ignore_index=True)
    
    print(f"\nDiverse sample size: {len(diverse_sample):,}")
    
    # 3. Combine Family Conflict positive + diverse sample (removing duplicates)
    print(f"\n{'='*60}")
    print("COMBINING DATASETS:")
    print(f"{'='*60}")
    
    # Merge and remove duplicates based on post_id
    combined = pd.concat([family_conflict_positive, diverse_sample], ignore_index=True)
    combined = combined.drop_duplicates(subset='post_id', keep='first')
    
    print(f"Family Conflict positive: {len(family_conflict_positive):,}")
    print(f"Diverse sample: {len(diverse_sample):,}")
    print(f"Combined (after deduplication): {len(combined):,}")
    
    # Sort by sentiment and bucket for easier review
    combined = combined.sort_values(['qwen_sentiment', 'qwen_bucket', 'qwen_confidence'], 
                                   ascending=[True, True, False])
    
    # Add a reasoning column if it doesn't exist (will be populated by reasoning extraction script)
    if 'qwen_reasoning' not in combined.columns:
        combined['qwen_reasoning'] = ''
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"SAVED TO: {output_file}")
    print(f"Total posts: {len(combined):,}")
    print(f"{'='*60}")
    
    # Show breakdown by sentiment×bucket
    print("\nFinal Sample Breakdown:")
    crosstab = pd.crosstab(combined['qwen_sentiment'], combined['qwen_bucket'], margins=True)
    print(crosstab)
    
    return combined

if __name__ == '__main__':
    df = create_targeted_sample(
        results_file='results/qwen_analysis_results.csv',
        output_file='data/targeted_sample_for_reasoning.csv',
        sample_per_combination=5  # Sample 5 posts per sentiment×bucket combination
    )
    
    if df is not None:
        print(f"\n✓ Targeted sample created successfully!")
        print(f"\nNext step: Run extract_reasoning.py to get LLM explanations for these posts")
