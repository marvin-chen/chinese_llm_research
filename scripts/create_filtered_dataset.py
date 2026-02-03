#!/usr/bin/env python3
"""
Create a filtered dataset with only contextually relevant posts for filial piety analysis
"""

import pandas as pd
import os
from tqdm import tqdm

def is_contextually_relevant(text):
    """Check if post is contextually discussing filial piety, not just containing the character"""
    if not isinstance(text, str) or '孝' not in text:
        return False
    
    # Keywords that indicate actual filial piety discussion
    filial_keywords = [
        '孝顺', '孝敬', '孝心', '孝道', '行孝', '尽孝', '孝子', '孝女', '孝义',
        '不孝', '孝行', '孝悌', '孝慈', '孝思', '孝养', '孝服', '孝治',
        '父母', '爸妈', '爹娘', '双亲', '家长', '长辈', '老人', '赡养',
        '照顾', '陪伴', '回家', '探亲', '养老', '侍奉', '报恩', '感恩'
    ]
    
    # Check for contextual relevance
    text_lower = text.lower()
    has_filial_context = any(keyword in text for keyword in filial_keywords)
    
    # If it just contains '孝' but no other context, likely irrelevant
    if not has_filial_context:
        # Check if it's likely a name, place, or irrelevant usage
        irrelevant_patterns = [
            '孝感',  # city name
            '孝陵',  # tomb name
            '孝庄',  # historical figure
            '孝文',  # historical figure
            '孝武',  # historical figure
            '孝康',  # name
            '孝义',  # when used as place name
            '二十四孝',  # when used as title only
        ]
        
        # If contains irrelevant patterns and no filial context, likely irrelevant
        if any(pattern in text for pattern in irrelevant_patterns):
            return False
    
    return has_filial_context or ('孝' in text and len(text) > 20)  # Give benefit of doubt for longer posts

def create_filtered_dataset():
    """Create filtered dataset with only contextually relevant posts"""
    input_file = "../data/weibo_xiao_sample_equal_per_month.csv"
    output_file = "../data/weibo_xiao_sample_relevant.csv"
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file '{input_file}' not found!")
        return
    
    print(f"Loading dataset: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} posts")
    
    print("Applying contextual relevance filtering...")
    
    # Apply filtering with progress bar
    relevant_mask = []
    for text in tqdm(df['text'], desc="Filtering posts"):
        relevant_mask.append(is_contextually_relevant(text))
    
    # Filter the dataset
    filtered_df = df[relevant_mask].copy().reset_index(drop=True)
    
    print(f"Filtered results:")
    print(f"   Original posts: {len(df)}")
    print(f"   Relevant posts: {len(filtered_df)}")
    print(f"   Filtered out: {len(df) - len(filtered_df)}")
    print(f"   Retention rate: {100 * len(filtered_df) / len(df):.1f}%")
    
    # Save filtered dataset
    filtered_df.to_csv(output_file, index=False)
    print(f"💾 Saved filtered dataset to: {output_file}")
    
    # Show some examples
    print(f"\nSample relevant posts:")
    for i, row in filtered_df.head(3).iterrows():
        text_preview = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
        print(f"   {i+1}. {text_preview}")
    
    return output_file

if __name__ == "__main__":
    create_filtered_dataset()