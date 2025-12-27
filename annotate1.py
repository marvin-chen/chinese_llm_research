"""
LLM Re-annotation Script for Manually Annotated Weibo Data

This script takes a CSV file that already contains manual annotations
and adds three new LLM annotation columns (llm_relevant1, llm_sentiment1, 
llm_reasoning1) while preserving all existing data.

Designed specifically for the "WJ edit" CSV file format which includes:
- Existing columns: weibo_id, time, full_text, llm_relevant, llm_sentiment,
  llm_reasoning, text_length, compounds_found, manual_relevant, 
  manual_sentiment, notes
- New columns to add: llm_relevant1, llm_sentiment1, llm_reasoning1

Uses Ollama (local) for annotation by default.
"""

import pandas as pd
import json
import subprocess
import time


def annotate_with_ollama(text, model="qwen2.5:7b"):
    """
    Annotate a single post using Ollama.

    Args:
        text (str): The Weibo post text
        model (str): Ollama model to use

    Returns:
        dict: Annotation results with keys 'relevant', 'sentiment', 'reasoning'
    """
    prompt = f"""你是一位研究中国传统文化的学者。请分析以下微博内容中"孝"的含义和情感倾向。

微博内容：
{text}

请按以下格式回答：

1. 相关性：这条微博是否讨论儒家"孝"的概念？
   - 相关：讨论父母子女关系、孝顺等传统价值观
   - 不相关：人名、地名、娱乐内容等

2. 情感分类（如果相关）：
   2: 强烈正面（赞扬、鼓励孝道）
   1: 温和正面（肯定孝的重要性）
   0: 中性（事实陈述、描述性）
   -1: 温和负面（质疑、批评）
   -2: 强烈负面（拒绝、讽刺孝道）

3. 简要理由（1-2句话）

请严格按照以下JSON格式回答：
{{
    "relevant": true/false,
    "sentiment": 2/1/0/-1/-2,
    "reasoning": "你的理由"
}}"""

    try:
        # Call the Ollama CLI. It reads the prompt from stdin and returns
        # a textual response on stdout. We set a timeout to avoid hangs.
        result = subprocess.run(
            ['ollama', 'run', model],
            input=prompt,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=60
        )

        response = result.stdout.strip()

        # Print raw LLM output for debugging — helps when JSON extraction fails
        print("\n--- RAW LLM OUTPUT ---\n", response, "\n----------------------\n")

        # Try to parse JSON from response. Sometimes LLMs add explanatory text
        # before/after the JSON block, so we extract the first {...} block.
        json_start = response.find('{')
        json_end = response.rfind('}') + 1

        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            annotation = json.loads(json_str)
            return annotation
        else:
            # If no JSON found, return error
            return {
                "relevant": None,
                "sentiment": None,
                "reasoning": "Failed to parse response",
                "raw_response": response
            }

    except subprocess.TimeoutExpired:
        return {
            "relevant": None,
            "sentiment": None,
            "reasoning": "Timeout",
            "raw_response": ""
        }
    except Exception as e:
        return {
            "relevant": None,
            "sentiment": None,
            "reasoning": f"Error: {str(e)}",
            "raw_response": ""
        }


def batch_annotate_existing_file(input_file, output_file, model="qwen2.5:7b", 
                                  start_index=0, end_index=None):
    """
    Re-annotate posts from an already annotated CSV file.
    
    This function:
    1. Loads the CSV file (skipping the first row if it's a header label)
    2. Preserves ALL existing columns
    3. Adds three new columns: llm_relevant1, llm_sentiment1, llm_reasoning1
    4. Allows processing a subset of rows via start_index and end_index
    
    Args:
        input_file (str): Path to the CSV file with existing annotations
        output_file (str): Path to save the new annotated results
        model (str): Ollama model name to use
        start_index (int): Starting row index (0-based, after header)
        end_index (int): Ending row index (exclusive). None means process all remaining rows.
    """
    print(f"Loading data from {input_file}...")
    
    # Read the CSV. The file has a label row at index 0, so we skip it.
    # The actual header is at row 1 (0-indexed row 1)
    df = pd.read_csv(input_file, encoding='utf-8', skiprows=[0])
    
    print(f"Loaded {len(df)} total posts")
    
    # Determine which rows to process
    if end_index is None:
        end_index = len(df)
    
    # Ensure indices are within bounds
    start_index = max(0, start_index)
    end_index = min(len(df), end_index)
    
    print(f"\nProcessing rows {start_index} to {end_index-1} ({end_index - start_index} posts)")
    print(f"Using Ollama model: {model}")
    print("="*80)
    
    # Initialize new columns if they don't exist
    if 'llm_relevant1' not in df.columns:
        df['llm_relevant1'] = None
    if 'llm_sentiment1' not in df.columns:
        df['llm_sentiment1'] = None
    if 'llm_reasoning1' not in df.columns:
        df['llm_reasoning1'] = None
    
    # Process the specified range of rows
    for idx in range(start_index, end_index):
        row = df.iloc[idx]
        
        print(f"\n[{idx+1}/{len(df)}] Processing post (index {idx})")
        print(f"Text: {str(row['full_text'])[:100]}...")
        
        # Annotate using Ollama
        annotation = annotate_with_ollama(str(row['full_text']), model)
        
        # Update the dataframe with new annotations
        df.at[idx, 'llm_relevant1'] = annotation.get('relevant')
        df.at[idx, 'llm_sentiment1'] = annotation.get('sentiment')
        df.at[idx, 'llm_reasoning1'] = annotation.get('reasoning')
        
        print(f"Relevant: {annotation.get('relevant')}")
        print(f"Sentiment: {annotation.get('sentiment')}")
        print(f"Reasoning: {annotation.get('reasoning')}")
        
        # Small delay to avoid overwhelming the LLM service
        time.sleep(0.5)
    
    # Save the updated dataframe with all original columns plus new annotations
    print(f"\nSaving results to {output_file}...")
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print("\n" + "="*80)
    print("ANNOTATION COMPLETE!")
    print("="*80)
    print(f"Results saved to: {output_file}")
    print(f"Total rows in file: {len(df)}")
    print(f"Rows annotated in this run: {end_index - start_index}")
    
    # Print summary statistics for the newly annotated rows
    annotated_subset = df.iloc[start_index:end_index]
    if annotated_subset['llm_relevant1'].notna().any():
        relevant_count = annotated_subset['llm_relevant1'].sum()
        print(f"\nRelevant posts (new annotations): {relevant_count}/{len(annotated_subset)}")
        
        if relevant_count > 0:
            relevant_posts = annotated_subset[annotated_subset['llm_relevant1'] == True]
            if len(relevant_posts) > 0 and relevant_posts['llm_sentiment1'].notna().any():
                sentiment_counts = relevant_posts['llm_sentiment1'].value_counts().sort_index()
                print("\nSentiment distribution (relevant posts only):")
                for sentiment, count in sentiment_counts.items():
                    label = {2: "Strongly Positive", 1: "Mildly Positive", 
                            0: "Neutral", -1: "Mildly Negative", 
                            -2: "Strongly Negative"}.get(sentiment, "Unknown")
                    print(f"  {int(sentiment):+2d} ({label}): {count}")


if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "llm_annotations_sample_validation WJ edit.csv"
    OUTPUT_FILE = "llm_annotations_sample_validation WJ edit_with_llm1.csv"
    
    # Ollama model configuration
    OLLAMA_MODEL = "qwen2.5:7b"  # Best for Chinese
    
    # Row range to process (useful for processing in batches)
    # Set to None to process all rows
    START_INDEX = 0      # Start from first post (0-based index)
    END_INDEX = None     # Process all remaining posts (set to a number to limit, e.g., 10)
    
    # Run annotation
    batch_annotate_existing_file(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        model=OLLAMA_MODEL,
        start_index=START_INDEX,
        end_index=END_INDEX
    )
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print(f"1. Open {OUTPUT_FILE} to review the new LLM annotations")
    print("2. Compare llm_relevant1/llm_sentiment1/llm_reasoning1 with existing annotations")
    print("3. Analyze agreement between original LLM, manual, and new LLM annotations")
    print("4. Use this for inter-annotator reliability analysis")
