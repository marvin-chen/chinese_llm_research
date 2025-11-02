"""
Weibo Data Preprocessing Script
Supports TSV input, CSV output
Prepared for manual and LLM-based exclusion filtering
"""

import pandas as pd
import re
import os
from datetime import datetime

def preprocess_weibo_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'\[[^\]]+\]', '', text)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA00-\U0001FA6F"
        u"\U00002600-\U000026FF"
        u"\U00002700-\U000027BF"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    text = re.sub(r'#([^#]+)#', r'\1', text)
    text = re.sub(r'@[^\s]+', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[↓△※☆…]+', '', text)
    text = re.sub(r'[\s\u3000]+', ' ', text)
    return text.strip()

def analyze_keyword_presence(df, keyword='孝'):
    df_analysis = df.copy()
    df_analysis['contains_keyword'] = df_analysis['full_text'].str.contains(keyword, na=False)
    pattern = f'[\u4e00-\u9fff]{{0,1}}{keyword}[\u4e00-\u9fff]{{0,2}}'
    df_analysis['keyword_compounds'] = df_analysis['full_text'].apply(
        lambda x: ', '.join(set(re.findall(pattern, str(x)))) if pd.notna(x) else ''
    )
    df_analysis['keyword_count'] = df_analysis['full_text'].str.count(keyword)
    return df_analysis

def filter_false_positives(df, keyword='孝', exclusion_list=None, llm_exclusion_results=None):
    df_filtered = df.copy()
    exclusion_mask = pd.Series([False] * len(df_filtered), index=df_filtered.index)
    if exclusion_list is None:
        exclusion_list = ['天野喜孝', '林孝埈', '喜孝']
    for term in exclusion_list:
        exclusion_mask |= df_filtered['full_text'].str.contains(term, na=False)
    if llm_exclusion_results is not None:
        exclusion_mask |= llm_exclusion_results
    df_filtered['is_false_positive'] = exclusion_mask.fillna(False).astype(bool)
    df_filtered = df_filtered[~df_filtered['is_false_positive'].fillna(False).astype(bool)].copy()
    return df_filtered

def preprocess_weibo_dataset(input_file, output_file, keyword='孝',
                             exclusion_list=None,
                             llm_exclusion_results=None,
                             input_sep='\t'):
    print("Loading data...")
    df = pd.read_csv(input_file, sep=input_sep, encoding='utf-8')
    print(f"Loaded {len(df)} posts")
    print("\nPreprocessing text...")
    df['time'] = pd.to_datetime(df['time'])
    df['cleaned_content'] = df['weibo_content'].apply(preprocess_weibo_text)
    df['cleaned_r_content'] = df['r_weibo_content'].apply(preprocess_weibo_text)
    df['full_text'] = df['cleaned_content'] + ' ' + df['cleaned_r_content']
    df['full_text'] = df['full_text'].str.strip()
    df['content_hash'] = df['full_text'].apply(lambda x: hash(x))
    df = df.drop_duplicates(subset=['content_hash']).copy()
    print(f"After deduplication: {len(df)} posts")
    df = df[df['full_text'].str.len() > 0].copy()
    print(f"After removing empty posts: {len(df)} posts")
    df['text_length'] = df['full_text'].str.len()
    df['has_repost'] = ~df['r_weibo_content'].isna()
    print(f"\nAnalyzing keyword '{keyword}'...")
    df = analyze_keyword_presence(df, keyword)
    print(f"Posts containing '{keyword}': {df['contains_keyword'].sum()}")
    print(f"\nFiltering false positives...")
    df_relevant = df[df['contains_keyword']].copy()
    df_relevant = filter_false_positives(df_relevant, keyword, exclusion_list, llm_exclusion_results)
    print(f"Posts after filtering: {len(df_relevant)}")
    df_no_keyword = df[~df['contains_keyword']].copy()
    df_final = pd.concat([df_relevant, df_no_keyword], ignore_index=True)
    df_final = df_final.sort_values('time', ascending=False).reset_index(drop=True)
    print(f"\nSaving to {output_file}...")
    df_final.to_csv(output_file, index=False, encoding='utf-8')
    print("Done!")
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total posts: {len(df_final)}")
    print(f"Posts with keyword: {df_final['contains_keyword'].sum()}")
    print(f"Posts with reposts: {df_final['has_repost'].sum()}")
    print(f"Average text length: {df_final['text_length'].mean():.1f} characters")
    print(f"Date range: {df_final['time'].min()} to {df_final['time'].max()}")
    relevant_posts = df_final[
        df_final['contains_keyword'] & (~df_final['is_false_positive'].fillna(False).astype(bool))
    ]
    if len(relevant_posts) > 0:
        print(f"\nMost common compounds containing '{keyword}':")
        all_compounds = relevant_posts['keyword_compounds'].str.split(', ').explode()
        print(all_compounds.value_counts().head(10))
    return df_final

def date_string_from_path(filepath):
    # Extract date from filename, e.g., "2016-01-01.tsv" -> "2016-01-01"
    base = os.path.basename(filepath)
    date_part = os.path.splitext(base)[0]
    return date_part

if __name__ == "__main__":
    input_file = "xiao-2016-2019/2016-01-01.tsv"    # Your input TSV path
    keyword = "孝"
    exclusion_list = None
    llm_exclusion_results = None
    input_sep = '\t'

    date_str = date_string_from_path(input_file)
    output_file = f"preprocessed_{date_str}.csv"

    df_processed = preprocess_weibo_dataset(
        input_file,
        output_file,
        keyword,
        exclusion_list=exclusion_list,
        llm_exclusion_results=llm_exclusion_results,
        input_sep=input_sep
    )
    df_keyword = df_processed[
        df_processed['contains_keyword'] & 
        (~df_processed['is_false_positive'].fillna(False).astype(bool))
    ].copy()
    if len(df_keyword) > 0:
        annotation_file = f"weibo_for_annotation_{keyword}_{date_str}.csv"
        df_keyword.to_csv(annotation_file, index=False, encoding='utf-8')
        print(f"\nPosts for annotation saved to: {annotation_file}")
        print(f"Total posts to annotate: {len(df_keyword)}")
