"""
LLM Annotation Script for Weibo Data
Supports both Ollama (local) and GPT-4 (API)
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
        dict: Annotation results
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

# note that json can't read + sign 

    try:
        # Call Ollama via subprocess
        result = subprocess.run(
            ['ollama', 'run', model],
            input=prompt,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=60
        )

        response = result.stdout.strip()
        
         # ALWAYS print raw LLM output for debugging!
        print("\n--- RAW LLM OUTPUT ---\n", response, "\n----------------------\n")

        # Try to parse JSON from response
        # Sometimes LLM adds extra text, so we need to extract JSON
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


def annotate_with_gpt4(text, api_key):
    """
    Annotate using GPT-4 API.

    Args:
        text (str): The Weibo post text
        api_key (str): OpenAI API key

    Returns:
        dict: Annotation results
    """
    try:
        import openai
        openai.api_key = api_key

        prompt = f"""你是一位研究中国传统文化的学者。请分析以下微博内容中"孝"的含义和情感倾向。

微博内容：
{text}

请按以下格式回答：

1. 相关性：这条微博是否讨论儒家"孝"的概念？
   - 相关：讨论父母子女关系、孝顺等传统价值观
   - 不相关：人名、地名、娱乐内容等

2. 情感分类（如果相关）：
   +2: 强烈正面（赞扬、鼓励孝道）
   +1: 温和正面（肯定孝的重要性）
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

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "你是一位专业的中国文化研究学者。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        content = response.choices[0].message.content

        # Parse JSON
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        json_str = content[json_start:json_end]
        annotation = json.loads(json_str)

        return annotation

    except Exception as e:
        return {
            "relevant": None,
            "sentiment": None,
            "reasoning": f"Error: {str(e)}",
            "raw_response": ""
        }


def batch_annotate(input_file, output_file, num_posts=10, use_ollama=True, 
                   model="qwen2.5:7b", api_key=None):
    """
    Batch annotate posts from preprocessed data.

    Args:
        input_file (str): Path to preprocessed CSV
        output_file (str): Path to save annotated results
        num_posts (int): Number of posts to annotate
        use_ollama (bool): Use Ollama if True, GPT-4 if False
        model (str): Model name for Ollama
        api_key (str): OpenAI API key if using GPT-4
    """
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, encoding='utf-8')

    # Filter for posts with keyword that aren't false positives
    if 'contains_keyword' in df.columns and 'is_false_positive' in df.columns:
        df_to_annotate = df[
            df['contains_keyword'] & ~df['is_false_positive']
        ].head(num_posts).copy()
    else:
        df_to_annotate = df.head(num_posts).copy()

    print(f"\nAnnotating {len(df_to_annotate)} posts...")
    print(f"Using: {'Ollama (' + model + ')' if use_ollama else 'GPT-4'}")
    print("="*80)

    annotations = []

    for idx, row in df_to_annotate.iterrows():
        print(f"\nPost {idx + 1}/{len(df_to_annotate)}")
        print(f"Text: {row['full_text'][:100]}...")

        # Annotate
        if use_ollama:
            annotation = annotate_with_ollama(row['full_text'], model)
        else:
            annotation = annotate_with_gpt4(row['full_text'], api_key)

        # Add to results
        result = {
            'weibo_id': row['weibo_id'],
            'time': row['time'],
            'full_text': row['full_text'],
            'llm_relevant': annotation.get('relevant'),
            'llm_sentiment': annotation.get('sentiment'),
            'llm_reasoning': annotation.get('reasoning'),
            'text_length': row.get('text_length', len(row['full_text']))
        }

        if 'keyword_compounds' in row:
            result['compounds_found'] = row['keyword_compounds']

        annotations.append(result)

        print(f"Relevant: {annotation.get('relevant')}")
        print(f"Sentiment: {annotation.get('sentiment')}")
        print(f"Reasoning: {annotation.get('reasoning')}")

        # Small delay to avoid overwhelming system
        time.sleep(0.5)

    # Save results
    df_results = pd.DataFrame(annotations)
    df_results.to_csv(output_file, index=False, encoding='utf-8')

    print("\n" + "="*80)
    print("ANNOTATION COMPLETE!")
    print("="*80)
    print(f"Results saved to: {output_file}")
    print(f"Total annotated: {len(df_results)}")

    # Print summary statistics
    if df_results['llm_relevant'].notna().any():
        relevant_count = df_results['llm_relevant'].sum()
        print(f"\nRelevant posts: {relevant_count}/{len(df_results)}")

        if relevant_count > 0:
            relevant_posts = df_results[df_results['llm_relevant'] == True]
            sentiment_counts = relevant_posts['llm_sentiment'].value_counts().sort_index()
            print("\nSentiment distribution (relevant posts only):")
            for sentiment, count in sentiment_counts.items():
                label = {2: "Strongly Positive", 1: "Mildly Positive", 
                        0: "Neutral", -1: "Mildly Negative", 
                        -2: "Strongly Negative"}.get(sentiment, "Unknown")
                print(f"  {int(sentiment):+2d} ({label}): {count}")

    # print("\n" + "="*80)
    # print("NEXT STEPS:")
    # print("="*80)
    # print(f"1. Open {output_file} in Excel/spreadsheet")
    # print("2. Review the annotations manually")
    # print("3. Add a column 'manual_relevant' and 'manual_sentiment'")
    # print("4. Fill in your own judgments")
    # print("5. Compare with LLM annotations to calculate accuracy")
    # print("\nSee validation_template.csv for example format")


def create_validation_template(annotations_file):
    """
    Create a template CSV for manual validation.
    """
    df = pd.read_csv(annotations_file, encoding='utf-8')

    # Add manual annotation columns
    df['manual_relevant'] = ''
    df['manual_sentiment'] = ''
    df['notes'] = ''

    output_file = annotations_file.replace('.csv', '_validation.csv')
    df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"Validation template created: {output_file}")
    print("Fill in 'manual_relevant' and 'manual_sentiment' columns")


if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "weibo_for_annotation_孝_2016-01-01.csv"  # From preprocessing
    OUTPUT_FILE = "llm_annotations_sample.csv"
    NUM_POSTS = 10  # Start with 10 posts

    # Choose LLM
    USE_OLLAMA = True  # Set to False to use GPT-4
    OLLAMA_MODEL = "qwen2.5:7b"  # Best for Chinese
    GPT4_API_KEY = None  # Add your OpenAI API key here if using GPT-4

    # Run annotation
    batch_annotate(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        num_posts=NUM_POSTS,
        use_ollama=USE_OLLAMA,
        model=OLLAMA_MODEL,
        api_key=GPT4_API_KEY
    )

    # Create validation template
    create_validation_template(OUTPUT_FILE)