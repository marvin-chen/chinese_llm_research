"""
Weibo Data Preprocessing Script
Supports TSV input, CSV output
Prepared for manual and LLM-based exclusion filtering
"""

import pandas as pd
import re
import os
import time 

def preprocess_weibo_text(text):
    """Normalize a single Weibo post text.

    Steps performed:
    - Handle NaN values by returning an empty string.
    - Remove bracketed tokens like emojis/expressions in square brackets: "[笑]".
    - Remove a wide range of Unicode emojis using a compiled regex.
    - Unwrap hashtags of the form #tag# -> tag.
    - Remove @mentions, URLs, some decorative symbols, and normalize whitespace.

    Args:
        text: raw post text (may be NaN or non-string).

    Returns:
        Cleaned string with surrounding whitespace trimmed.

    Note: This function intentionally preserves Chinese characters and most punctuation
    while removing common noise observed in Weibo posts.
    """
    if pd.isna(text):
        return ""
    text = str(text)

    # Remove things like "[哈哈]" which often denote reactions/stickers
    text = re.sub(r'\[[^\]]+\]', '', text)

    # Emoji ranges: covers many pictographs and symbol blocks. Removing these
    # reduces noise for keyword matching and deduplication.
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA00-\U0001FA6F"
        u"\U00002600-\U000026FF"
        u"\U00002700-\U000027BF"
        "+]", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)

    # Convert hashtags like #孝道# -> 孝道 (remove surrounding # characters)
    text = re.sub(r'#([^#]+)#', r'\1', text)

    # Remove @mentions (usernames), URLs, and a few decorative symbols
    text = re.sub(r'@[^\s]+', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'[↓△※☆…]+', '', text)

    # Normalize spaces including full-width ideographic space (\u3000)
    text = re.sub(r'[\s\u3000]+', ' ', text)
    return text.strip()

def analyze_keyword_presence(df, keyword='孝'):
    """Analyze presence of a keyword in a DataFrame's `full_text` column.

    Adds three columns to a copy of the input DataFrame:
    - contains_keyword: boolean whether `keyword` appears anywhere in `full_text`.
    - keyword_compounds: short list of matched compounds around the keyword (1 char before, up to 2 after).
    - keyword_count: number of occurrences of the keyword in `full_text`.

    Args:
        df: DataFrame containing a `full_text` column (strings).
        keyword: single Chinese character or substring to search for.

    Returns:
        A new DataFrame with the added analysis columns.
    """
    df_analysis = df.copy()
    # Fast containment check (vectorized)
    df_analysis['contains_keyword'] = df_analysis['full_text'].str.contains(keyword, na=False)

    # Pattern captures short local contexts around the keyword to help surface common compounds.
    # Example for keyword='孝': matches '孝顺', '老孝子', etc. Limits to CJK range to avoid extraneous matches.
    pattern = f'[\u4e00-\u9fff]{{0,1}}{keyword}[\u4e00-\u9fff]{{0,2}}'
    df_analysis['keyword_compounds'] = df_analysis['full_text'].apply(
        lambda x: ', '.join(set(re.findall(pattern, str(x)))) if pd.notna(x) else ''
    )

    # Count occurrences of the keyword for simple frequency stats
    df_analysis['keyword_count'] = df_analysis['full_text'].str.count(keyword)
    return df_analysis

def filter_false_positives(df, keyword='孝', exclusion_list=None, llm_exclusion_results=None):
    """Filter out known false positives from a DataFrame of posts.

    The function marks posts as false positives if they contain any term from
    `exclusion_list` (e.g., proper names that happen to contain the character)
    or if an external boolean mask `llm_exclusion_results` flags them.

    Args:
        df: DataFrame with `full_text` column.
        keyword: (unused directly) kept for API symmetry.
        exclusion_list: list of substrings that indicate false positives.
        llm_exclusion_results: optional pandas Series (boolean mask) aligned with df index.

    Returns:
        A filtered DataFrame with `is_false_positive` column set, and only non-false-positive rows returned.
    """
    df_filtered = df.copy()
    # Start with an all-False mask
    exclusion_mask = pd.Series([False] * len(df_filtered), index=df_filtered.index)
    if exclusion_list is None:
        # Default exclusion examples: names or terms that include the character but are not relevant
        exclusion_list = ['天野喜孝', '林孝埈', '喜孝']
    for term in exclusion_list:
        # Mark rows that contain any exclusion term
        exclusion_mask |= df_filtered['full_text'].str.contains(term, na=False)
    if llm_exclusion_results is not None:
        # Allow an externally computed mask (for example, human or LLM labeling)
        exclusion_mask |= llm_exclusion_results

    df_filtered['is_false_positive'] = exclusion_mask.fillna(False).astype(bool)
    # Return only rows that are NOT false positives
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
    # Clean original and reposted content columns. The script expects the input
    # TSV to have at least the columns: `time`, `weibo_content`, `r_weibo_content`.
    df['cleaned_content'] = df['weibo_content'].apply(preprocess_weibo_text)
    df['cleaned_r_content'] = df['r_weibo_content'].apply(preprocess_weibo_text)

    # Combine original and repost text into a single searchable `full_text` field.
    df['full_text'] = df['cleaned_content'] + ' ' + df['cleaned_r_content']
    df['full_text'] = df['full_text'].str.strip()

    # Simple deduplication based on the (Python) hash of the normalized text.
    # Note: Python's hash is salted per process; this is sufficient for intra-run dedupe
    # but not stable across processes. If stable IDs are required, consider hashlib.md5.
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
