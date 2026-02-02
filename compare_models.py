"""
Quick Model Comparison Test - GPT-5 Mini Version
Tests Qwen vs GPT-5 Mini on a small sample to diagnose the issue
No emojis, text-only output
Fixed timeout and retry logic
"""

import pandas as pd
import json
import subprocess
import time
import re
import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


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

# ============================================================================
# FUNCTION 1: Extract JSON from response
# ============================================================================
def clean_json_response(response_text):
    """
    Extract and clean JSON from response text.
    
    Args:
        response_text: Raw response from model
        
    Returns:
        dict: Parsed JSON object or None if parsing fails
    """
    if not response_text or len(response_text.strip()) == 0:
        return None
    
    try:
        # Find JSON block in response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        
        if start == -1 or end <= start:
            return None
        
        json_str = response_text[start:end]
        
        # Fix common formatting issue: +1 becomes 1
        json_str = re.sub(r':\s*\+([0-9])', r': \1', json_str)
        
        data = json.loads(json_str)
        return data
    
    except:
        return None


# ============================================================================
# FUNCTION 2: Test with Qwen 2.5 (Local, Free)
# ============================================================================
def test_ollama(text):
    """
    Send text to Qwen 2.5 model running locally via Ollama
    
    Args:
        text: The Weibo post text to analyze
        
    Returns:
        dict: JSON response from Qwen with sentiment and reasoning
    """
    full_prompt = f"{SYSTEM_PROMPT}\n\nPost: \"{text}\"\n\nJSON:"
    
    try:
        result = subprocess.run(
            ['ollama', 'run', 'qwen2.5:7b'],
            input=full_prompt,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=30
        )
        
        response = result.stdout.strip()
        data = clean_json_response(response)
        
        if data:
            return data
        else:
            return {"error": "No JSON found"}
    
    except subprocess.TimeoutExpired:
        return {"error": "Ollama timeout (30s)"}
    except Exception as e:
        return {"error": str(e)[:50]}


# ============================================================================
# FUNCTION 3: Test with GPT-5 Mini (API, Paid)
# ============================================================================
def test_gpt5_mini(text, api_key, max_retries=3):
    """
    Send text to GPT-5 Mini via OpenAI API with retry logic for timeouts
    
    Args:
        text: The Weibo post text to analyze
        api_key: Your OpenAI API key from https://platform.openai.com/api-keys
        max_retries: Number of retry attempts for timeouts
        
    Returns:
        dict: JSON response from GPT-5 Mini with sentiment and reasoning
    """
    for attempt in range(max_retries):
        try:
            # Create OpenAI client with API key
            client = openai.OpenAI(api_key=api_key)
            
            full_prompt = f"{SYSTEM_PROMPT}\n\nPost: \"{text}\"\n\nJSON:"
            
            # Call GPT-4o Mini API (latest OpenAI model)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Chinese cultural expert analyzing Weibo sentiment. Follow the rubric EXACTLY. Return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                max_completion_tokens=200,
                timeout=30
            )
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Debug: Check if response is None or empty
            if not response_text:
                return {"error": "Empty response from API", "finish_reason": response.choices[0].finish_reason}
            
            data = clean_json_response(response_text)
            
            if data:
                return data
            else:
                # Debug: return raw response to see what GPT-5 Mini returned
                return {"error": "Failed to parse JSON", "raw_response": response_text[:200]}
        
        except openai.APITimeoutError:
            # Timeout error - retry with exponential backoff
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            else:
                return {"error": f"GPT-5 Mini timeout after {max_retries} attempts"}
        
        except openai.RateLimitError:
            # Rate limit error - retry with exponential backoff
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            else:
                return {"error": "Rate limit exceeded"}
        
        except openai.APIConnectionError as e:
            # Connection error - do not retry
            return {"error": f"Connection error: {str(e)[:40]}"}
        
        except openai.APIStatusError as e:
            # API status error - do not retry
            return {"error": f"API error: {str(e)}"}
        
        except Exception as e:
            # Unknown error
            return {"error": str(e)[:50]}
    
    return {"error": "Unknown error"}


# ============================================================================
# FUNCTION 4: Main comparison function
# ============================================================================
def compare_models(csv_file, api_key=None):
    """
    Load ground truth examples and test both models
    
    Args:
        csv_file: Path to ground_truth_preprocessed.csv
        api_key: Optional OpenAI API key to test GPT-5 Mini
    """
    print("=" * 80)
    print("MODEL COMPARISON TEST - GPT-5 MINI VERSION")
    print("=" * 80)
    print(f"Loading examples from: {csv_file}\n")
    
    # Load CSV file
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except Exception as e:
        print(f"ERROR: Cannot load CSV: {str(e)}")
        return
    
    # Get +1 examples (Mild Positive) - first 3
    plus1_examples = df[df['manual_sentiment'] == 1].head(3)
    
    # Get +2 examples (Strong Positive) - first 3
    plus2_examples = df[df['manual_sentiment'] == 2].head(3)
    
    # Create test cases
    test_cases = []
    
    for idx, row in plus1_examples.iterrows():
        test_cases.append({
            'text': row['cleaned_text'][:150],
            'manual': 1,
            'label': '+1 (Mild Positive)'
        })
    
    for idx, row in plus2_examples.iterrows():
        test_cases.append({
            'text': row['cleaned_text'][:150],
            'manual': 2,
            'label': '+2 (Strong Positive)'
        })
    
    if not test_cases:
        print("ERROR: No +1 or +2 examples found in CSV")
        return
    
    print(f"Testing {len(test_cases)} examples\n")
    
    results = []
    
    # Test each example
    for idx, test in enumerate(test_cases, 1):
        print(f"[{idx}/{len(test_cases)}] Manual: {test['label']}")
        print(f"  Text: {test['text']}...\n")
        
        # TEST QWEN 2.5
        print(f"  Testing Qwen 2.5 (7B)...", end=" ", flush=True)
        qwen_result = test_ollama(test['text'])
        qwen_sentiment = qwen_result.get('sentiment')
        
        if qwen_sentiment == test['manual']:
            qwen_match = "[PASS]"
        else:
            qwen_match = "[FAIL]"
        
        print(f"{qwen_match} Predicted: {qwen_sentiment}")
        if 'reasoning' in qwen_result:
            print(f"           Reasoning: {qwen_result['reasoning']}")
        if 'error' in qwen_result:
            print(f"           Error: {qwen_result['error']}")
        
        # TEST GPT-5 MINI (if API key provided)
        gpt_sentiment = None
        if api_key:
            print(f"  Testing GPT-5 Mini...", end=" ", flush=True)
            gpt_result = test_gpt5_mini(test['text'], api_key)
            gpt_sentiment = gpt_result.get('sentiment')
            
            if gpt_sentiment == test['manual']:
                gpt_match = "[PASS]"
            else:
                gpt_match = "[FAIL]"
            
            print(f"{gpt_match} Predicted: {gpt_sentiment}")
            if 'reasoning' in gpt_result:
                print(f"           Reasoning: {gpt_result['reasoning']}")
            if 'error' in gpt_result:
                print(f"           Error: {gpt_result['error']}")
            if 'raw_response' in gpt_result:
                print(f"           Raw: {gpt_result['raw_response']}")
        
        # Store result
        results.append({
            'manual': test['manual'],
            'qwen_prediction': qwen_sentiment,
            'qwen_correct': qwen_sentiment == test['manual'],
            'gpt_prediction': gpt_sentiment,
            'gpt_correct': gpt_sentiment == test['manual'] if gpt_sentiment else None
        })
        
        print()
    
    # Print summary
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    qwen_correct = sum(1 for r in results if r['qwen_correct'])
    qwen_accuracy = qwen_correct / len(results) * 100 if results else 0
    print(f"Qwen 2.5 (7B):  {qwen_correct}/{len(results)} correct ({qwen_accuracy:.1f}%)")
    
    if api_key:
        gpt_correct = sum(1 for r in results if r['gpt_correct'])
        gpt_accuracy = gpt_correct / len(results) * 100 if results else 0
        print(f"GPT-5 Mini:     {gpt_correct}/{len(results)} correct ({gpt_accuracy:.1f}%)")
        
        print()
        if gpt_correct > qwen_correct:
            improvement = gpt_correct - qwen_correct
            print(f"VERDICT: GPT-5 Mini is better (improved by {improvement} more correct answers)")
            print(f"Cost: USD 0.80 per 1,000 posts (vs free with Qwen)")
        elif gpt_correct < qwen_correct:
            print(f"VERDICT: Qwen is better or comparable")
        else:
            print(f"VERDICT: Both models are equal on this sample")
    else:
        print(f"\nTo test GPT-5 Mini, provide your OpenAI API key")
    
    print("=" * 80)
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    CSV_FILE = "ground_truth_preprocessed.csv"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
    
    # Debug: Check if API key is loaded
    if OPENAI_API_KEY:
        print(f"API Key loaded: {OPENAI_API_KEY[:10]}...{OPENAI_API_KEY[-4:]}")
    else:
        print("WARNING: No API key found. Only Qwen will be tested.")
        print("To test GPT-5 Mini, add OPENAI_API_KEY to your .env file\n")
    
    compare_models(CSV_FILE, OPENAI_API_KEY)
    
    print("\nTO TEST WITH GPT-5 Mini:")
    print("1. Get your API key from https://platform.openai.com/api-keys")
    print("2. Set it in the script: OPENAI_API_KEY = 'sk-...'")
    print("3. Run again to compare")
    print("\nCOST ESTIMATE:")
    print("- GPT-5 Mini: approximately USD 0.80 per 1,000 posts")
    print("- For 30,000 posts: approximately USD 24 total")
    print("\nPRICING: https://openai.com/api/pricing/")
