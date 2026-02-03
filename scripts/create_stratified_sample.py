"""
Time-Stratified Random Sampling for Weibo Dataset
Creates equal-per-month sample from weibo_xiao_cleaned.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def create_time_stratified_sample(input_file, k=100, seed=42, 
                                start_date='2016-01-01', end_date='2023-11-30'):
    """
    Create time-stratified random sample with equal posts per month
    
    Args:
        input_file: Path to weibo_xiao_cleaned.csv
        k: Target number of posts per month
        seed: Random seed for reproducibility
        start_date: Start of study period (YYYY-MM-DD)
        end_date: End of study period (YYYY-MM-DD)
    
    Returns:
        sample_df: Sampled dataframe
    """
    
    print("="*80)
    print("TIME-STRATIFIED SAMPLING FOR WEIBO DATASET")
    print("="*80)
    print(f"Input file: {input_file}")
    print(f"Target posts per month: {k}")
    print(f"Study period: {start_date} to {end_date}")
    print(f"Random seed: {seed}")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Step 1: Load and filter data
    print("\n1. Loading and filtering data...")
    df = pd.read_csv(input_file, encoding='utf-8')
    print(f"   Original dataset size: {len(df):,} posts")
    
    # Parse timestamp
    df['time'] = pd.to_datetime(df['time'])
    
    # Filter to study period
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Filter by date range
    df_filtered = df[(df['time'] >= start_dt) & (df['time'] <= end_dt)].copy()
    print(f"   After date filtering: {len(df_filtered):,} posts")
    
    # Ensure we have the required columns
    # Map to expected column names if they don't exist
    if 'weibo_id' not in df_filtered.columns and 'post_id' not in df_filtered.columns:
        df_filtered['post_id'] = df_filtered['weibo_id']  # Use weibo_id as post_id
    elif 'post_id' not in df_filtered.columns:
        df_filtered['post_id'] = df_filtered['weibo_id']
    
    if 'text' not in df_filtered.columns:
        df_filtered['text'] = df_filtered['cleaned_text']  # Use cleaned_text as text
    
    # Step 2: Create time strata
    print("\n2. Creating time strata...")
    
    # Add year and month columns (may already exist from preprocessing)
    if 'year' not in df_filtered.columns:
        df_filtered['year'] = df_filtered['time'].dt.year
    if 'month' not in df_filtered.columns:
        df_filtered['month'] = df_filtered['time'].dt.month
    
    # Get all unique (year, month) pairs
    strata = df_filtered.groupby(['year', 'month']).size().reset_index()
    strata.columns = ['year', 'month', 'count']
    strata = strata.sort_values(['year', 'month'])
    
    print(f"   Found {len(strata)} unique (year, month) strata:")
    print("   Year-Month: Count")
    for _, row in strata.iterrows():
        print(f"   {row['year']:4d}-{row['month']:02d}: {row['count']:,}")
    
    total_posts_available = strata['count'].sum()
    print(f"\n   Total posts in study period: {total_posts_available:,}")
    
    # Step 3: Sample within each month
    print(f"\n3. Sampling {k} posts per month...")
    
    sampled_dfs = []
    sampling_summary = []
    
    for _, stratum in strata.iterrows():
        year, month, N_h = stratum['year'], stratum['month'], stratum['count']
        
        # Get posts for this month
        df_h = df_filtered[(df_filtered['year'] == year) & (df_filtered['month'] == month)].copy()
        
        # Determine sample size for this month
        if N_h >= k:
            sample_size = k
            df_sampled = df_h.sample(n=k, random_state=seed, replace=False)
        else:
            sample_size = N_h
            df_sampled = df_h.copy()  # Take all posts
        
        sampled_dfs.append(df_sampled)
        sampling_summary.append({
            'year': year,
            'month': month,
            'available': N_h,
            'sampled': sample_size,
            'sampling_rate': sample_size / N_h if N_h > 0 else 0
        })
        
        print(f"   {year}-{month:02d}: {sample_size:3d} / {N_h:,} posts "
              f"({100 * sample_size / N_h:.1f}%)")
    
    # Step 4: Combine all samples
    print("\n4. Combining samples...")
    sample_df = pd.concat(sampled_dfs, ignore_index=True)
    
    # Sort by timestamp
    sample_df = sample_df.sort_values('time').reset_index(drop=True)
    
    # Step 5: Generate summary statistics
    print("\n" + "="*80)
    print("SAMPLING SUMMARY")
    print("="*80)
    
    summary_df = pd.DataFrame(sampling_summary)
    
    total_sampled = summary_df['sampled'].sum()
    total_available = summary_df['available'].sum()
    months_with_full_sample = (summary_df['sampled'] == k).sum()
    months_with_partial_sample = (summary_df['sampled'] < k).sum()
    
    print(f"Total months in study period: {len(summary_df)}")
    print(f"Months with full sample ({k} posts): {months_with_full_sample}")
    print(f"Months with partial sample (<{k} posts): {months_with_partial_sample}")
    print(f"")
    print(f"Target sample size (if all months had {k}+ posts): {len(summary_df) * k:,}")
    print(f"Actual sample size: {total_sampled:,}")
    print(f"Overall sampling rate: {100 * total_sampled / total_available:.2f}%")
    print(f"")
    print(f"Sample date range: {sample_df['time'].min()} to {sample_df['time'].max()}")
    
    # Show months with insufficient posts
    insufficient_months = summary_df[summary_df['sampled'] < k]
    if len(insufficient_months) > 0:
        print(f"\nMonths with fewer than {k} posts:")
        for _, row in insufficient_months.iterrows():
            print(f"  {row['year']:4d}-{row['month']:02d}: {row['available']:3d} posts available")
    
    return sample_df, summary_df

def save_sample_results(sample_df, output_prefix="weibo_xiao_sample_equal_per_month"):
    """
    Save sampling results to CSV files
    
    Args:
        sample_df: Sampled dataframe
        output_prefix: Prefix for output files
    """
    
    print(f"\n5. Saving results...")
    
    # Define columns to save in main file
    essential_columns = ['post_id', 'time', 'year', 'month', 'text']
    
    # Add any other useful columns that exist
    additional_columns = ['weibo_id', 'user_id', 'text_length', 'has_repost', 'contains_xiao', 'source_file']
    
    columns_to_save = []
    for col in essential_columns:
        if col in sample_df.columns:
            columns_to_save.append(col)
    
    for col in additional_columns:
        if col in sample_df.columns and col not in columns_to_save:
            columns_to_save.append(col)
    
    # Save main sample file
    main_output = f"{output_prefix}.csv"
    sample_df[columns_to_save].to_csv(main_output, index=False, encoding='utf-8')
    print(f"   Main sample saved to: {main_output}")
    
    # Save post IDs only file
    ids_output = f"{output_prefix}_ids.csv"
    sample_df[['post_id']].to_csv(ids_output, index=False, encoding='utf-8')
    print(f"   Post IDs saved to: {ids_output}")
    
    # File sizes
    main_size = os.path.getsize(main_output) / (1024 * 1024)  # MB
    ids_size = os.path.getsize(ids_output) / 1024  # KB
    
    print(f"   Main file size: {main_size:.1f} MB")
    print(f"   IDs file size: {ids_size:.1f} KB")
    
    return main_output, ids_output

def main():
    """Main execution function"""
    
    # Configuration
    input_file = "weibo_xiao_cleaned.csv"
    k = 100  # Target posts per month
    seed = 42
    start_date = '2016-01-01'
    end_date = '2023-11-30'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print("Please make sure weibo_xiao_cleaned.csv exists in the current directory.")
        return
    
    # Create stratified sample
    sample_df, summary_df = create_time_stratified_sample(
        input_file=input_file,
        k=k,
        seed=seed,
        start_date=start_date,
        end_date=end_date
    )
    
    # Save results
    main_file, ids_file = save_sample_results(sample_df)
    
    print("\n" + "="*80)
    print("SAMPLING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Sample files created:")
    print(f"  - {main_file}")
    print(f"  - {ids_file}")
    print("")
    print("To reproduce this exact sample, run with the same parameters:")
    print(f"  k = {k}")
    print(f"  seed = {seed}")
    print(f"  date range = {start_date} to {end_date}")
    
    return sample_df, summary_df

if __name__ == "__main__":
    sample_df, summary_df = main()