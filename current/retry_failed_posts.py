"""
Retry Failed Extractions and Update Results
Re-processes posts that had extraction failures with a more lenient approach
"""

import pandas as pd
import json
import subprocess
import time
import os
from datetime import datetime
from tqdm import tqdm

# More explicit prompt for retry with stricter bucket requirements
RETRY_PROMPT = """你是中国文化专家，评估微博对"孝"的态度。

重要规则：
1. sentiment必须是整数: -2, -1, 0, 1, 或 2
2. 如果sentiment=0，bucket必须是空字符串 ""
3. 如果sentiment≠0，bucket必须从以下5个选择1个：
   - 日常实践
   - 责任义务
   - 家庭冲突
   - 理论探讨
   - 婚恋择偶

评分标准：
• -2 (强烈负面): 强烈批评孝道限制自由，视为负担/债务
• -1 (轻度负面): 表达压力/矛盾，但未否定孝道价值
• 0 (中性): 娱乐/玩笑语境，无道德判断
• +1 (轻度正面): 一般性赞扬孝道，无具体细节
• +2 (强烈正面): 详细描述具体孝行，或严厉谴责不孝

confidence评分（0-100整数）：
• 90-100%: 非常确信
• 70-89%: 比较确信
• 50-69%: 一般确信
• 30-49%: 不太确信

严格返回JSON格式（不要markdown代码块）：
{"sentiment": -2或-1或0或1或2, "bucket": "选项或空字符串", "confidence": 整数}

微博文本："""


class FailedPostRetrier:
    def __init__(self, results_file="results/qwen_analysis_results.csv"):
        self.results_file = results_file
        
        print(f"Loading results: {results_file}")
        self.df = pd.read_csv(results_file)
        print(f"Total posts: {len(self.df)}")
        
        # Find failed posts
        self.failed_mask = self.df['qwen_error'].notna() & (self.df['qwen_error'] != '')
        self.failed_df = self.df[self.failed_mask].copy()
        print(f"Failed posts: {len(self.failed_df)}")
        
        if len(self.failed_df) > 0:
            error_counts = self.failed_df['qwen_error'].value_counts()
            print("\nError breakdown:")
            for error, count in error_counts.items():
                print(f"  {error}: {count}")
    
    def extract_json_strict(self, response_text):
        """Strict JSON extraction with validation"""
        if not response_text:
            return None
        
        import re
        
        # Remove markdown code blocks
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*', '', response_text)
        
        # Find JSON object
        json_match = re.search(r'\{[^{}]*\}', response_text)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                
                # Validate sentiment
                if 'sentiment' not in data:
                    return None
                
                sentiment = data['sentiment']
                if isinstance(sentiment, str):
                    sentiment = int(sentiment.replace('+', ''))
                else:
                    sentiment = int(sentiment)
                
                if sentiment not in [-2, -1, 0, 1, 2]:
                    return None
                
                # Validate bucket
                bucket = data.get('bucket', '')
                if sentiment == 0:
                    bucket = ''
                else:
                    valid_buckets = ['日常实践', '责任义务', '家庭冲突', '理论探讨', '婚恋择偶']
                    if bucket not in valid_buckets:
                        # Try to infer from text if possible
                        return None
                
                # Validate confidence
                confidence = int(data.get('confidence', 50))
                if confidence < 0 or confidence > 100:
                    confidence = 50
                
                return {
                    'sentiment': sentiment,
                    'bucket': bucket,
                    'confidence': confidence
                }
            except:
                pass
        
        return None
    
    def retry_single_post(self, text):
        """Retry processing a single post with improved prompt"""
        if not text or len(str(text).strip()) == 0:
            return {"error": "empty_text"}
        
        clean_text = str(text).replace('"', '\\"')[:500]
        prompt = f"{RETRY_PROMPT}{clean_text}\n\nJSON:"
        
        # Try up to 2 times
        for attempt in range(2):
            try:
                result = subprocess.run(
                    ['ollama', 'run', 'qwen2.5:7b'],
                    input=prompt,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    timeout=40
                )
                
                if result.returncode != 0:
                    time.sleep(2)
                    continue
                
                response = result.stdout.strip()
                data = self.extract_json_strict(response)
                
                if data:
                    return data
                
                time.sleep(2)
            except subprocess.TimeoutExpired:
                time.sleep(2)
                continue
            except Exception as e:
                time.sleep(2)
                continue
        
        return {"error": "retry_failed"}
    
    def retry_failed_posts(self):
        """Retry all failed posts and update the dataframe"""
        if len(self.failed_df) == 0:
            print("No failed posts to retry!")
            return
        
        print(f"\n🔄 Retrying {len(self.failed_df)} failed posts...")
        print("Using improved prompt with stricter validation\n")
        
        success_count = 0
        still_failed = 0
        
        for idx in tqdm(self.failed_df.index, desc="Retrying"):
            text = self.df.loc[idx, 'text']
            
            result = self.retry_single_post(text)
            
            if 'error' not in result:
                # Success! Update the main dataframe
                self.df.loc[idx, 'qwen_sentiment'] = result['sentiment']
                self.df.loc[idx, 'qwen_bucket'] = result['bucket']
                self.df.loc[idx, 'qwen_confidence'] = result['confidence']
                self.df.loc[idx, 'qwen_error'] = None  # Clear error
                self.df.loc[idx, 'qwen_processed_at'] = datetime.now().isoformat()
                success_count += 1
            else:
                still_failed += 1
        
        print(f"\n✅ Retry complete!")
        print(f"   Successful: {success_count}")
        print(f"   Still failed: {still_failed}")
        print(f"   Success rate: {100 * success_count / len(self.failed_df):.1f}%")
        
        return success_count
    
    def save_results(self):
        """Save updated results back to CSV"""
        self.df.to_csv(self.results_file, index=False)
        print(f"\n💾 Saved updated results to: {self.results_file}")
        
        # Show final statistics
        total = len(self.df)
        successful = self.df['qwen_sentiment'].notna().sum()
        errors = self.df['qwen_error'].notna().sum()
        
        print(f"\n📊 Final Statistics:")
        print(f"   Total posts: {total:,}")
        print(f"   Successfully analyzed: {successful:,} ({100*successful/total:.1f}%)")
        print(f"   Errors: {errors:,} ({100*errors/total:.1f}%)")


def main():
    print("="*60)
    print("RETRY FAILED EXTRACTIONS")
    print("="*60)
    
    retrier = FailedPostRetrier("results/qwen_analysis_results.csv")
    
    if len(retrier.failed_df) == 0:
        print("\n✅ No failed posts found! All posts successfully analyzed.")
        return
    
    confirm = input(f"\nRetry {len(retrier.failed_df)} failed posts? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    success_count = retrier.retry_failed_posts()
    
    if success_count > 0:
        save = input("\nSave updated results to CSV? (y/n): ").strip().lower()
        if save == 'y':
            retrier.save_results()
        else:
            print("Results not saved.")
    else:
        print("\nNo successful retries to save.")


if __name__ == "__main__":
    main()
