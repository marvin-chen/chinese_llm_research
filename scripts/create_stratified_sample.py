"""
Time-Stratified Random Sampling for Weibo Dataset
Creates equal-per-month sample from weibo_xiao_cleaned.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def create_time_stratified_sample(input_file='data/weibo_xiao_cleaned.csv', 
                                k=2000, seed=42, 
                                start_date='2016-01-01', end_date='2023-12-31'):
    """
    Create time-stratified random sample with equal posts per month
    
    Args:
        input_file: Path to weibo_xiao_cleaned.csv
        k: Target posts per month (2000 for ~192k total over 96 months: 2016-2023)
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
    
    # Set random seed
    np.random.seed(seed)
    
    # Step 1: Load and filter data
    print("\n1. Loading dataset...")
    df = pd.read_csv(input_file, encoding='utf-8')
    print(f"   Original dataset size: {len(df):,} posts")
    
    # Parse timestamp
    df['time'] = pd.to_datetime(df['time'])
    
    # Filter to study period
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df_filtered = df[(df['time'] >= start_dt) & (df['time'] <= end_dt)].copy()
    print(f"   After date filtering: {len(df_filtered):,} posts")
    
    # Ensure we have required columns
    if 'post_id' not in df_filtered.columns and 'weibo_id' in df_filtered.columns:
        df_filtered['post_id'] = df_filtered['weibo_id']
    
    if 'text' not in df_filtered.columns and 'cleaned_text' in df_filtered.columns:
        df_filtered['text'] = df_filtered['cleaned_text']
    
    # Add year/month columns if needed
    if 'year' not in df_filtered.columns:
        df_filtered['year'] = df_filtered['time'].dt.year
    if 'month' not in df_filtered.columns:
        df_filtered['month'] = df_filtered['time'].dt.month
    
    # Step 2: Sample from each month
    print("\n2. Sampling from each month...")
    
    # Get all unique (year, month) pairs
    strata = df_filtered.groupby(['year', 'month']).size().reset_index()
    strata.columns = ['year', 'month', 'count']
    strata = strata.sort_values(['year', 'month'])
    
    total_available = strata['count'].sum()
    print(f"   Found {len(strata)} months with total {total_available:,} posts")
    
    sampled_dfs = []
    sampling_summary = []
    
    for _, stratum in strata.iterrows():
        year, month, N_h = stratum['year'], stratum['month'], stratum['count']
        
        # Get posts for this month
        df_month = df_filtered[(df_filtered['year'] == year) & (df_filtered['month'] == month)].copy()
        
        # Determine sample size for this month
        if N_h >= k:
            sample_size = k
            df_sampled = df_month.sample(n=k, random_state=seed, replace=False)
        else:
            sample_size = N_h
            df_sampled = df_month.copy()
        
        sampled_dfs.append(df_sampled)
        sampling_summary.append({
            'year': year,
            'month': month,
            'available': N_h,
            'sampled': sample_size,
            'sampling_rate': sample_size / N_h if N_h > 0 else 0
        })
        
        print(f"   {year}-{month:02d}: {sample_size:4d} / {N_h:,} posts "
              f"({100 * sample_size / N_h:.1f}%)")
    
    # Step 3: Combine all samples
    print(f"\n3. Combining samples from {len(sampled_dfs)} months...")
    
    if len(sampled_dfs) == 0:
        print("ERROR: No data was sampled!")
        return None, None
    
    sample_df = pd.concat(sampled_dfs, ignore_index=True)
    sample_df = sample_df.sort_values('time').reset_index(drop=True)
    
    # Step 4: Generate summary statistics
    print("\n" + "="*80)
    print("SAMPLING SUMMARY")
    print("="*80)
    
    summary_df = pd.DataFrame(sampling_summary)
    total_sampled = summary_df['sampled'].sum()
    months_with_full_sample = (summary_df['sampled'] == k).sum()
    months_with_partial_sample = (summary_df['sampled'] < k).sum()
    
    print(f"Total months in study period: {len(summary_df)}")
    print(f"Months with full sample ({k} posts): {months_with_full_sample}")
    print(f"Months with partial sample (<{k} posts): {months_with_partial_sample}")
    print(f"")
    print(f"Posts available in study period: {total_available:,}")
    print(f"Target sample size: {len(summary_df) * k:,}")
    print(f"Actual sample size: {total_sampled:,}")
    print(f"Overall sampling rate: {100 * total_sampled / total_available:.2f}%")
    print(f"Sample date range: {sample_df['time'].min()} to {sample_df['time'].max()}")
    
    # Show months with insufficient posts
    insufficient_months = summary_df[summary_df['sampled'] < k]
    if len(insufficient_months) > 0:
        print(f"\nMonths with fewer than {k} posts:")
        for _, row in insufficient_months.iterrows():
            print(f"  {int(row['year'])}-{int(row['month']):02d}: {int(row['available']):,} posts")
    
    return sample_df, summary_df

def save_sample_results(sample_df, output_prefix="data/weibo_xiao_sample_equal_per_month"):
    """Save sampling results to CSV files"""
    
    print(f"\n5. Saving results...")
    os.makedirs("data", exist_ok=True)
    
    # Define columns to save
    essential_columns = ['post_id', 'time', 'year', 'month', 'text']
    additional_columns = ['user_id', 'text_length', 'r_weibo_content']
    
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
    
    # Save post IDs only
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
    """Main execution"""
    
    input_file = 'data/weibo_xiao_cleaned.csv'
    k = 2000  # Target posts per month (192k total over 96 months: 2016-2023)
    seed = 42
    start_date = '2016-01-01'
    end_date = '2023-12-31'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print("Please make sure weibo_xiao_cleaned.csv exists in the data directory.")
        return
    
    # Create stratified sample
    sample_df, summary_df = create_time_stratified_sample(
        input_file=input_file,
        k=k,
        seed=seed,
        start_date=start_date,
        end_date=end_date
    )
    
    if sample_df is None:
        print("Sampling failed!")
        return
    
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
    print(f"  input_file = {input_file}")
    print(f"  k = {k}")
    print(f"  seed = {seed}")
    print(f"  date range = {start_date} to {end_date}")
    print("")
    print("Next step: Run batch_analyze.py on this new sample")
    
    return sample_df, summary_df

if __name__ == "__main__":
    sample_df, summary_df = main()
