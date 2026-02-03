#!/usr/bin/env python3
"""
Sentiment Analysis Visualization Dashboard
Analyzes and visualizes results from qwen_analysis_results.csv
Works with any number of analyzed posts
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

def normalize_bucket_names(df):
    """Normalize bucket names by removing Chinese translations in parentheses"""
    
    df = df.copy()
    
    # Create mapping for bucket name normalization
    bucket_mapping = {
        'Care(赡养照护)': 'Care',
        'Obligation(责任义务)': 'Obligation', 
        'Conflict(家庭冲突)': 'Conflict',
        'Critique/Abstract(理论探讨)': 'Critique/Abstract',
        'Reciprocity(情感互惠)': 'Reciprocity'
    }
    
    # Apply normalization
    df['qwen_bucket'] = df['qwen_bucket'].replace(bucket_mapping)
    
    return df

def load_and_prepare_data():
    """Load and prepare the analysis results"""
    
    results_file = 'results/qwen_analysis_results.csv'
    
    if not os.path.exists(results_file):
        print(f"ERROR: Results file '{results_file}' not found!")
        print("Run batch_analyze.py first to generate results.")
        return None
    
    print(f"Loading results from: {results_file}")
    df = pd.read_csv(results_file)
    
    # Filter to only processed posts
    processed_df = df[df['qwen_processed_at'].notna()].copy()
    
    # Normalize bucket names
    print(f"\nBucket normalization:")
    original_buckets = processed_df[processed_df['qwen_relevant'] == True]['qwen_bucket'].value_counts()
    processed_df = normalize_bucket_names(processed_df)
    normalized_buckets = processed_df[processed_df['qwen_relevant'] == True]['qwen_bucket'].value_counts()
    
    print(f"Original unique buckets: {len(original_buckets)}")
    print(f"Normalized unique buckets: {len(normalized_buckets)}")
    
    print(f"Total posts in file: {len(df)}")
    print(f"Successfully processed: {len(processed_df)}")
    
    if len(processed_df) == 0:
        print("No processed posts found!")
        return None
    
    return processed_df

def create_sentiment_distribution(df, save_path='results/'):
    """Create sentiment score distribution chart"""
    
    plt.figure(figsize=(10, 6))
    
    # Count sentiment scores
    sentiment_counts = df['qwen_sentiment'].value_counts().sort_index()
    
    # Create labels with descriptions
    sentiment_labels = {
        -2: 'Strongly Negative (-2)',
        -1: 'Slightly Negative (-1)', 
        0: 'Neutral/Irrelevant (0)',
        1: 'Slightly Positive (+1)',
        2: 'Strongly Positive (+2)'
    }
    
    # Plot bar chart
    bars = plt.bar(sentiment_counts.index, sentiment_counts.values, 
                   color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd'])
    
    plt.title(f'Sentiment Distribution (n={len(df)} posts)', fontsize=16, fontweight='bold')
    plt.xlabel('Sentiment Score', fontsize=12)
    plt.ylabel('Number of Posts', fontsize=12)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    # Add percentage labels
    total = len(df)
    max_count = sentiment_counts.values.max()
    for i, (score, count) in enumerate(sentiment_counts.items()):
        percentage = (count / total) * 100
        plt.text(score, count + max_count * 0.05,
                f'{percentage:.1f}%',
                ha='center', va='bottom', fontsize=10, style='italic')
    
    plt.xticks(list(sentiment_labels.keys()), 
               [sentiment_labels[k] for k in sorted(sentiment_labels.keys())])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'{save_path}sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return sentiment_counts

def create_bucket_distribution(df, save_path='results/'):
    """Create bucket distribution chart"""
    
    # Only look at relevant posts for bucket analysis
    relevant_df = df[df['qwen_relevant'] == True]
    
    if len(relevant_df) == 0:
        print("No relevant posts found for bucket analysis")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bucket counts
    bucket_counts = relevant_df['qwen_bucket'].value_counts()
    
    # Bar chart
    bars = ax1.bar(range(len(bucket_counts)), bucket_counts.values, 
                   color=sns.color_palette("husl", len(bucket_counts)))
    ax1.set_title(f'Context Bucket Distribution\n(Relevant posts only: n={len(relevant_df)})', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Context Buckets', fontsize=12)
    ax1.set_ylabel('Number of Posts', fontsize=12)
    ax1.set_xticks(range(len(bucket_counts)))
    ax1.set_xticklabels(bucket_counts.index, rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    colors = sns.color_palette("husl", len(bucket_counts))
    wedges, texts, autotexts = ax2.pie(bucket_counts.values, labels=bucket_counts.index, 
                                      autopct='%1.1f%%', startangle=90, colors=colors)
    ax2.set_title('Bucket Distribution\n(Proportions)', fontsize=14, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}bucket_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return bucket_counts

def analyze_confidence_scores(df, save_path='results/', low_confidence_threshold=70):
    """Analyze confidence scores and identify posts needing manual review"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Confidence distribution histogram
    ax1.hist(df['qwen_confidence'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.axvline(low_confidence_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Low confidence threshold ({low_confidence_threshold}%)')
    ax1.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Confidence Score (%)', fontsize=12)
    ax1.set_ylabel('Number of Posts', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Confidence vs Sentiment scatter
    scatter = ax2.scatter(df['qwen_sentiment'], df['qwen_confidence'], 
                         alpha=0.6, c=df['qwen_confidence'], cmap='viridis')
    ax2.axhline(low_confidence_threshold, color='red', linestyle='--', linewidth=2)
    ax2.set_title('Confidence vs Sentiment', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sentiment Score', fontsize=12)
    ax2.set_ylabel('Confidence Score (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Confidence %')
    
    # 3. Confidence by bucket (for relevant posts)
    relevant_df = df[df['qwen_relevant'] == True]
    if len(relevant_df) > 0:
        bucket_confidence = relevant_df.groupby('qwen_bucket')['qwen_confidence'].agg(['mean', 'std', 'count'])
        bucket_confidence = bucket_confidence.sort_values('mean')
        
        bars = ax3.bar(range(len(bucket_confidence)), bucket_confidence['mean'], 
                      yerr=bucket_confidence['std'], capsize=5, alpha=0.7, color='lightcoral')
        ax3.axhline(low_confidence_threshold, color='red', linestyle='--', linewidth=2)
        ax3.set_title('Average Confidence by Bucket', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Context Buckets', fontsize=12)
        ax3.set_ylabel('Average Confidence (%)', fontsize=12)
        ax3.set_xticks(range(len(bucket_confidence)))
        ax3.set_xticklabels(bucket_confidence.index, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, bucket_confidence['count'])):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'n={count}',
                    ha='center', va='bottom', fontsize=10)
    
    # 4. Low confidence posts by sentiment
    low_conf_df = df[df['qwen_confidence'] < low_confidence_threshold]
    if len(low_conf_df) > 0:
        low_conf_sentiment = low_conf_df['qwen_sentiment'].value_counts().sort_index()
        ax4.bar(low_conf_sentiment.index, low_conf_sentiment.values, 
               color='orange', alpha=0.7)
        ax4.set_title(f'Low Confidence Posts by Sentiment\n(n={len(low_conf_df)} posts need review)', 
                     fontsize=14, fontweight='bold')
        ax4.set_xlabel('Sentiment Score', fontsize=12)
        ax4.set_ylabel('Number of Low Confidence Posts', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (score, count) in enumerate(low_conf_sentiment.items()):
            ax4.text(score, count, f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return low_conf_df

def create_sentiment_bucket_heatmap(df, save_path='results/'):
    """Create heatmap showing sentiment distribution across buckets"""
    
    relevant_df = df[df['qwen_relevant'] == True]
    if len(relevant_df) == 0:
        print("No relevant posts for heatmap")
        return
    
    # Create crosstab
    sentiment_bucket_crosstab = pd.crosstab(relevant_df['qwen_bucket'], 
                                           relevant_df['qwen_sentiment'], 
                                           margins=True)
    
    # Remove margins for heatmap, keep for reference
    heatmap_data = sentiment_bucket_crosstab.iloc[:-1, :-1]  # Remove 'All' row and column
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Blues', 
                cbar_kws={'label': 'Number of Posts'})
    plt.title('Sentiment Distribution Across Context Buckets\n(Relevant Posts Only)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Sentiment Score', fontsize=12)
    plt.ylabel('Context Bucket', fontsize=12)
    
    # Add sentiment labels
    sentiment_labels = {-2: 'Strong Neg', -1: 'Slight Neg', 0: 'Neutral', 
                       1: 'Slight Pos', 2: 'Strong Pos'}
    
    plt.tight_layout()
    plt.savefig(f'{save_path}sentiment_bucket_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return sentiment_bucket_crosstab

def export_low_confidence_posts(df, save_path='results/', threshold=70):
    """Export low confidence posts for manual annotation"""
    
    low_conf_df = df[df['qwen_confidence'] < threshold].copy()
    
    if len(low_conf_df) == 0:
        print(f"No posts found with confidence < {threshold}%")
        return
    
    # Select relevant columns for manual review
    review_cols = ['post_id', 'text', 'qwen_relevant', 'qwen_sentiment', 
                   'qwen_bucket', 'qwen_confidence', 'qwen_reasoning']
    
    review_df = low_conf_df[review_cols].copy()
    review_df = review_df.sort_values('qwen_confidence')  # Sort by lowest confidence first
    
    # Add manual review columns
    review_df['manual_relevant'] = ''
    review_df['manual_sentiment'] = ''  
    review_df['manual_bucket'] = ''
    review_df['manual_notes'] = ''
    
    # Export to CSV
    output_file = f'{save_path}low_confidence_posts_for_review.csv'
    review_df.to_csv(output_file, index=False)
    
    print(f"\nLow confidence posts exported to: {output_file}")
    print(f"Posts needing manual review: {len(review_df)}")
    print(f"Confidence range: {review_df['qwen_confidence'].min():.1f}% - {review_df['qwen_confidence'].max():.1f}%")
    
    return review_df

def generate_summary_report(df, save_path='results/'):
    """Generate comprehensive summary report"""
    
    total_posts = len(df)
    relevant_posts = len(df[df['qwen_relevant'] == True])
    
    report = f"""
SENTIMENT ANALYSIS SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

DATASET OVERVIEW:
- Total posts analyzed: {total_posts:,}
- Relevant posts (mention filial piety): {relevant_posts:,} ({relevant_posts/total_posts*100:.1f}%)
- Irrelevant posts: {total_posts-relevant_posts:,} ({(total_posts-relevant_posts)/total_posts*100:.1f}%)

SENTIMENT DISTRIBUTION:
"""
    
    sentiment_counts = df['qwen_sentiment'].value_counts().sort_index()
    sentiment_labels = {
        -2: 'Strongly Negative', -1: 'Slightly Negative', 0: 'Neutral/Irrelevant',
        1: 'Slightly Positive', 2: 'Strongly Positive'
    }
    
    for score, count in sentiment_counts.items():
        percentage = (count / total_posts) * 100
        label = sentiment_labels.get(score, f'Score {score}')
        report += f"- {label} ({score}): {count:,} posts ({percentage:.1f}%)\n"
    
    if relevant_posts > 0:
        report += f"\nCONTEXT BUCKET DISTRIBUTION (Relevant posts only):\n"
        bucket_counts = df[df['qwen_relevant'] == True]['qwen_bucket'].value_counts()
        for bucket, count in bucket_counts.items():
            percentage = (count / relevant_posts) * 100
            report += f"- {bucket}: {count:,} posts ({percentage:.1f}%)\n"
    
    # Confidence analysis
    report += f"\nCONFIDENCE ANALYSIS:\n"
    report += f"- Average confidence: {df['qwen_confidence'].mean():.1f}%\n"
    report += f"- Median confidence: {df['qwen_confidence'].median():.1f}%\n"
    report += f"- Min confidence: {df['qwen_confidence'].min():.1f}%\n"
    report += f"- Max confidence: {df['qwen_confidence'].max():.1f}%\n"
    
    low_conf_count = len(df[df['qwen_confidence'] < 70])
    report += f"- Posts with confidence < 70%: {low_conf_count:,} ({low_conf_count/total_posts*100:.1f}%)\n"
    
    # Save report
    with open(f'{save_path}analysis_summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    
    return report

def main():
    """Main visualization pipeline"""
    
    print("SENTIMENT ANALYSIS VISUALIZATION DASHBOARD")
    print("=" * 50)
    
    # Create results directory if not exists
    os.makedirs('results', exist_ok=True)
    
    # Load data
    df = load_and_prepare_data()
    if df is None:
        return
    
    print(f"\nGenerating visualizations...")
    
    # Create all visualizations
    print("1. Creating sentiment distribution chart...")
    sentiment_counts = create_sentiment_distribution(df)
    
    print("2. Creating bucket distribution charts...")  
    bucket_counts = create_bucket_distribution(df)
    
    print("3. Analyzing confidence scores...")
    low_conf_df = analyze_confidence_scores(df)
    
    print("4. Creating sentiment-bucket heatmap...")
    crosstab = create_sentiment_bucket_heatmap(df)
    
    print("5. Exporting low confidence posts for manual review...")
    review_df = export_low_confidence_posts(df)
    
    print("6. Generating summary report...")
    report = generate_summary_report(df)
    
    print(f"\nAll visualizations saved to 'results/' directory!")
    print(f"Check the following files:")
    print(f"- sentiment_distribution.png")
    print(f"- bucket_distribution.png") 
    print(f"- confidence_analysis.png")
    print(f"- sentiment_bucket_heatmap.png")
    print(f"- low_confidence_posts_for_review.csv")
    print(f"- analysis_summary_report.txt")

if __name__ == '__main__':
    main()