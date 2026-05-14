#!/usr/bin/env python3
"""
Extract reasoning from LLM for targeted sample posts
Asks the model to explain WHY it assigned specific sentiment and bucket classifications
"""

import pandas as pd
import subprocess
import json
import time
import os
from datetime import datetime
from tqdm import tqdm

REASONING_PROMPT_TEMPLATE = """你是一个中文社交媒体内容分析专家。请分析以下微博帖子关于"孝道"(filial piety)的情感和语境分类。

帖子内容：
{text}

这个帖子之前被分类为：
- 情感分数: {sentiment} (范围从-2到+2，-2=强烈负面，-1=轻微负面，0=中立/无关，+1=轻微正面，+2=强烈正面)
- 语境类别: {bucket}
- 置信度: {confidence}%

请详细解释：
1. 为什么这个帖子被给予 {sentiment} 的情感分数？具体是帖子中的哪些词语、短语或表达体现了这种情感？
2. 为什么这个帖子被归类到 "{bucket}" 类别？帖子的哪些内容符合这个类别的特征？
3. 这个分类是否合理？如果不合理，应该如何调整？

请用清晰、详细的中文回答，引用帖子中的具体内容来支持你的解释。

回答："""

def get_bucket_chinese(bucket_english):
    """Convert English bucket name back to Chinese for the prompt"""
    bucket_mapping = {
        'Daily Practice': '日常实践',
        'Obligation': '责任义务',
        'Family Conflict': '家庭冲突',
        'Theory/Critique': '理论探讨',
        'Marriage/Dating': '婚恋择偶',
    }
    return bucket_mapping.get(bucket_english, bucket_english)

def extract_reasoning_from_llm(text, sentiment, bucket, confidence, model='qwen2.5:7b', timeout=60):
    """
    Query Ollama LLM to extract reasoning for the classification
    
    Args:
        text: The post text
        sentiment: The sentiment score (-2 to +2)
        bucket: The bucket category (in English)
        confidence: The confidence score
        model: Ollama model to use
        timeout: Timeout in seconds
    
    Returns:
        tuple: (reasoning_text, error_message)
    """
    
    # Convert bucket to Chinese for the prompt
    bucket_chinese = get_bucket_chinese(bucket)
    
    # Format prompt
    prompt = REASONING_PROMPT_TEMPLATE.format(
        text=text,
        sentiment=sentiment,
        bucket=bucket_chinese,
        confidence=confidence
    )
    
    try:
        # Call Ollama
        result = subprocess.run(
            ['ollama', 'run', model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            reasoning = result.stdout.strip()
            return reasoning, None
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return None, f"Ollama error: {error_msg}"
            
    except subprocess.TimeoutExpired:
        return None, f"Timeout after {timeout}s"
    except Exception as e:
        return None, f"Error: {str(e)}"

def extract_reasoning_batch(input_file='data/targeted_sample_for_reasoning.csv',
                           output_file='data/targeted_sample_with_reasoning.csv',
                           progress_file='results/reasoning_extraction_progress.json',
                           model='qwen2.5:7b',
                           timeout=60):
    """
    Extract reasoning for all posts in the targeted sample
    
    Args:
        input_file: CSV file with targeted sample
        output_file: CSV file to save results with reasoning
        progress_file: JSON file to track progress
        model: Ollama model to use
        timeout: Timeout per post in seconds
    """
    
    print("="*80)
    print("REASONING EXTRACTION FOR TARGETED SAMPLE")
    print("="*80)
    
    # Load targeted sample
    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        return None
    
    print(f"\nLoading targeted sample from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Total posts to process: {len(df):,}")
    
    # Load progress if exists
    processed_ids = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            processed_ids = set(progress.get('processed_ids', []))
        print(f"Found existing progress: {len(processed_ids):,} posts already processed")
    
    # Add reasoning column if doesn't exist
    if 'llm_reasoning' not in df.columns:
        df['llm_reasoning'] = ''
    if 'reasoning_error' not in df.columns:
        df['reasoning_error'] = ''
    if 'reasoning_extracted_at' not in df.columns:
        df['reasoning_extracted_at'] = ''
    
    # Process each post
    print(f"\nExtracting reasoning using model: {model}")
    print(f"Timeout per post: {timeout}s")
    print(f"Starting extraction...\n")
    
    start_time = time.time()
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing posts"):
        post_id = row['post_id']
        
        # Skip if already processed
        if post_id in processed_ids:
            skipped_count += 1
            continue
        
        text = row['text']
        sentiment = row['qwen_sentiment']
        bucket = row['qwen_bucket']
        confidence = row['qwen_confidence']
        
        # Extract reasoning
        reasoning, error = extract_reasoning_from_llm(
            text=text,
            sentiment=sentiment,
            bucket=bucket,
            confidence=confidence,
            model=model,
            timeout=timeout
        )
        
        # Update dataframe
        if reasoning:
            df.at[idx, 'llm_reasoning'] = reasoning
            df.at[idx, 'reasoning_extracted_at'] = datetime.now().isoformat()
            success_count += 1
        else:
            df.at[idx, 'reasoning_error'] = error or "Unknown error"
            error_count += 1
        
        # Mark as processed
        processed_ids.add(post_id)
        
        # Save progress every 10 posts
        if len(processed_ids) % 10 == 0:
            # Save dataframe
            df.to_csv(output_file, index=False)
            
            # Save progress tracking
            progress = {
                'processed_ids': list(processed_ids),
                'last_updated': datetime.now().isoformat(),
                'success_count': success_count,
                'error_count': error_count,
            }
            os.makedirs(os.path.dirname(progress_file), exist_ok=True)
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
    
    # Final save
    df.to_csv(output_file, index=False)
    
    # Save final progress
    progress = {
        'processed_ids': list(processed_ids),
        'last_updated': datetime.now().isoformat(),
        'success_count': success_count,
        'error_count': error_count,
        'completed': True
    }
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)
    
    # Summary
    elapsed_time = time.time() - start_time
    avg_time_per_post = elapsed_time / max(success_count + error_count, 1)
    
    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Total posts: {len(df):,}")
    print(f"Successfully extracted: {success_count:,}")
    print(f"Errors: {error_count:,}")
    print(f"Skipped (already processed): {skipped_count:,}")
    print(f"Success rate: {success_count/(success_count+error_count)*100:.1f}%")
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"Average time per post: {avg_time_per_post:.1f}s")
    print(f"\nOutput saved to: {output_file}")
    print(f"{'='*80}")
    
    return df

def main():
    """Main function with interactive menu"""
    
    print("REASONING EXTRACTION TOOL")
    print("="*80)
    print("This tool extracts LLM reasoning for why posts were classified with")
    print("specific sentiment scores and bucket categories.")
    print("="*80)
    
    print("\nOptions:")
    print("1. Extract reasoning for all posts in targeted sample")
    print("2. Continue from previous progress")
    print("3. Reset progress and start fresh")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1' or choice == '2':
        # Run extraction
        df = extract_reasoning_batch(
            input_file='data/targeted_sample_for_reasoning.csv',
            output_file='data/targeted_sample_with_reasoning.csv',
            progress_file='results/reasoning_extraction_progress.json',
            model='qwen2.5:7b',
            timeout=60
        )
        
        if df is not None:
            print("\n✓ Reasoning extraction completed!")
            
            # Show sample
            print("\nSample of extracted reasoning:")
            with_reasoning = df[df['llm_reasoning'] != '']
            if len(with_reasoning) > 0:
                sample = with_reasoning.iloc[0]
                print(f"\nPost: {sample['text'][:100]}...")
                print(f"Sentiment: {sample['qwen_sentiment']}")
                print(f"Bucket: {sample['qwen_bucket']}")
                print(f"Reasoning:\n{sample['llm_reasoning'][:300]}...")
    
    elif choice == '3':
        # Reset progress
        progress_file = 'results/reasoning_extraction_progress.json'
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print("✓ Progress reset. Run option 1 to start fresh.")
        else:
            print("No progress file found.")
    
    elif choice == '4':
        print("Exiting...")
    
    else:
        print("Invalid choice!")

if __name__ == '__main__':
    main()
