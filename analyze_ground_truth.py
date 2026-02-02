"""
Analyze Ground Truth with LLM 
Uses exact rubric wording + real examples from ground truth data
"""

import pandas as pd
import json
import subprocess
import time
import re
import sys

# Uses exact wording from the provided PDF rubric + real ground truth examples
SYSTEM_PROMPT = """你是中国文化专家，评估微博对"孝"的态度。严格按照评分标准分类。

评分标准：
• 0 (无关): "孝"是地名/人名/书名，与内容无关
• 0 (中性): 娱乐/玩笑语境，无道德判断
• -1 (轻度负面): 表达压力/矛盾，但未否定孝道价值
• -2 (强烈负面): 强烈批评孝道限制自由，视为负担/债务
• +1 (轻度正面): 一般性赞扬孝道，无具体细节（如"孝顺很重要"、征婚提及孝顺）
• +2 (强烈正面): 详细描述具体孝行，或严厉谴责不孝

关键区别：
+1 vs +2: 一般性言论 vs 具体详细例子
-1 vs -2: 矛盾压力 vs 强烈批评

上下文分类：Reciprocity(情感互惠) | Obligation(责任义务) | Care(赡养照护) | Conflict(家庭冲突) | Critique/Abstract(理论探讨) | None(无关)

示例：
1. "忠孝东路108号" → {"relevant": false, "sentiment": 0, "bucket": "None", "reasoning": "地名"}
2. "母慈子孝的一家公司！" (讽刺) → {"relevant": true, "sentiment": -1, "bucket": "Critique/Abstract", "reasoning": "讽刺但未否定"}
3. "25年养育不应该用50年来还" → {"relevant": true, "sentiment": -2, "bucket": "Conflict", "reasoning": "视孝为债务，强烈批评"}
4. 征婚："孝顺，善良" → {"relevant": true, "sentiment": 1, "bucket": "Care", "reasoning": "一般性认可，无具体例子"}
5. "公司每月发孝道金" → {"relevant": true, "sentiment": 2, "bucket": "Care", "reasoning": "具体制度化措施，详细描述"}

仅返回JSON格式：{"relevant": boolean, "sentiment": int, "bucket": "string", "reasoning": "string"}
"""

def clean_json_response(response_text):
    """Extract and clean JSON from response."""
    if not response_text or len(response_text.strip()) == 0:
        return None
    
    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        
        if start == -1 or end <= start:
            return None
        
        json_str = response_text[start:end]
        json_str = re.sub(r':\s*\+([0-9])', r': \1', json_str)
        
        data = json.loads(json_str)
        
        if 'relevant' not in data or 'sentiment' not in data:
            return None
        
        return data
    
    except:
        return None


def analyze_post_ollama(text, model="qwen2.5:7b", max_retries=3):
    """Analyze a post with Ollama, with retry logic."""
    full_prompt = f"{SYSTEM_PROMPT}\n\nPost: \"{text}\"\nResult:"
    
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ['ollama', 'run', model],
                input=full_prompt,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=45
            )
            
            response = result.stdout.strip()
            data = clean_json_response(response)
            
            if data:
                return data
            
            if attempt < max_retries - 1:
                time.sleep(0.5)
                continue
            else:
                return {
                    "relevant": None,
                    "sentiment": None,
                    "bucket": "Error",
                    "reasoning": f"Failed JSON parsing after {max_retries} attempts"
                }
        
        except subprocess.TimeoutExpired:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            else:
                return {
                    "relevant": None,
                    "sentiment": None,
                    "bucket": "Error",
                    "reasoning": "Timeout after retries"
                }
        
        except Exception as e:
            return {
                "relevant": None,
                "sentiment": None,
                "bucket": "Error",
                "reasoning": f"Exception: {str(e)[:50]}"
            }
    
    return {
        "relevant": None,
        "sentiment": None,
        "bucket": "Error",
        "reasoning": "Unknown error"
    }


def analyze_and_compare_ground_truth(input_file, output_file, model="qwen2.5:7b"):
    """Load ground truth, analyze with LLM, compare results."""
    print(f"Loading preprocessed ground truth from {input_file}...")
    df = pd.read_csv(input_file, encoding='utf-8')
    
    print(f"Total posts in file: {len(df)}")
    print(f"\nAnalyzing with {model}...")
    print(f"Using exact rubric definitions and real ground truth examples\n")
    start_time = time.time()
    
    results = []
    
    for idx, row in df.iterrows():
        text = str(row.get('cleaned_text', ''))
        
        if len(text) < 5:
            continue
        
        if (idx + 1) % 5 == 0:
            print(f"  Processed {idx + 1}/{len(df)} posts...")
        
        analysis = analyze_post_ollama(text, model)
        
        result_row = row.to_dict()
        result_row['llm_relevant'] = analysis.get('relevant')
        result_row['llm_sentiment'] = analysis.get('sentiment')
        result_row['llm_bucket'] = analysis.get('bucket')
        result_row['llm_reasoning'] = analysis.get('reasoning')
        
        manual_relevant = row.get('manual_relevant')
        llm_relevant = analysis.get('relevant')
        result_row['relevant_match'] = (
            (manual_relevant == llm_relevant) 
            if (pd.notna(manual_relevant) and llm_relevant is not None)
            else False
        )
        
        manual_sentiment = row.get('manual_sentiment')
        llm_sentiment = analysis.get('sentiment')
        result_row['sentiment_match'] = (
            (manual_sentiment == llm_sentiment)
            if (pd.notna(manual_sentiment) and llm_sentiment is not None)
            else False
        )
        
        results.append(result_row)
    
    final_df = pd.DataFrame(results)
    final_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_file}")
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f}s ({elapsed/len(final_df):.2f}s per post)\n")
    
    total_posts = len(final_df)
    relevant_matches = final_df['relevant_match'].sum()
    sentiment_matches = final_df['sentiment_match'].sum()
    relevant_accuracy = (relevant_matches / total_posts * 100) if total_posts > 0 else 0
    sentiment_accuracy = (sentiment_matches / total_posts * 100) if total_posts > 0 else 0
    
    print(f" OVERALL ACCURACY:")
    print(f"  Total posts analyzed: {total_posts}")
    print(f"  Relevance Agreement: {relevant_accuracy:.1f}% ({relevant_matches}/{total_posts})")
    print(f"  Sentiment Agreement: {sentiment_accuracy:.1f}% ({sentiment_matches}/{total_posts})\n")
    
    print(f" SENTIMENT BREAKDOWN:")
    for sentiment_val in [-2, -1, 0, 1, 2]:
        subset = final_df[final_df['manual_sentiment'] == sentiment_val]
        if len(subset) > 0:
            matches = subset['sentiment_match'].sum()
            pct = (matches / len(subset) * 100)
            label = {-2: "Strong Negative", -1: "Mild Negative", 
                     0: "Neutral/Irrelevant", 1: "Mild Positive", 
                     2: "Strong Positive"}.get(sentiment_val)
            print(f"  {int(sentiment_val):+2d} ({label:20s}): {matches}/{len(subset)} ({pct:5.1f}%)")
    
    errors = final_df[final_df['llm_bucket'] == 'Error']
    if len(errors) > 0:
        print(f"\n ERROR CASES: {len(errors)}")
    
    disagreements = final_df[~final_df['sentiment_match']]
    print(f"\n DISAGREEMENTS: {len(disagreements)} posts")
    
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    if sentiment_accuracy >= 85:
        print(f"HIGH ACCURACY ({sentiment_accuracy:.1f}%) - Safe to proceed with full dataset")
    elif sentiment_accuracy >= 75:
        print(f" MODERATE ACCURACY ({sentiment_accuracy:.1f}%) - Consider refining")
    else:
        print(f"LOW ACCURACY ({sentiment_accuracy:.1f}%) - Recommend refining prompt")
    print("="*80)
    
    return final_df


if __name__ == "__main__":
    INPUT_FILE = "ground_truth_preprocessed.csv"
    OUTPUT_FILE = "ground_truth_llm_comparison_final.csv"
    MODEL = "qwen2.5:7b"
    
    df_results = analyze_and_compare_ground_truth(INPUT_FILE, OUTPUT_FILE, MODEL)
    
    print(f"\n NEXT STEPS:")
    print(f"1. Review {OUTPUT_FILE} for detailed results")
    print(f"2. If accuracy >80%, proceed to full dataset annotation")
