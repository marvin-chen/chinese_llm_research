"""
Qwen Analysis for Time-Stratified Weibo Sample
Processes weibo_xiao_sample_equal_per_month.csv with filial piety sentiment analysis
"""

import pandas as pd
import json
import subprocess
import time
import os
from datetime import datetime
from tqdm import tqdm

# System prompt from user request
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

class WeiboSampleAnalyzer:
    def __init__(self, input_file, output_prefix="qwen_analysis"):
        """
        Initialize analyzer for sampled Weibo dataset
        
        Args:
            input_file: Path to weibo_xiao_sample_equal_per_month.csv
            output_prefix: Prefix for output files
        """
        self.input_file = input_file
        self.output_prefix = output_prefix
        self.progress_file = f"{output_prefix}_progress.json"
        self.results_file = f"{output_prefix}_results.csv"
        
        # Load data
        print("Loading sampled dataset...")
        self.df = pd.read_csv(input_file)
        print(f"Loaded {len(self.df)} posts from {input_file}")
        
        # Initialize results columns
        self.df['qwen_relevant'] = None
        self.df['qwen_sentiment'] = None
        self.df['qwen_bucket'] = None
        self.df['qwen_reasoning'] = None
        self.df['qwen_error'] = None
        self.df['qwen_processed_at'] = None
        
        # Load existing progress
        self.progress = self.load_progress()
    
    def load_progress(self):
        """Load processing progress"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "last_processed_idx": -1,
            "total_processed": 0,
            "start_time": None,
            "successful": 0,
            "errors": 0
        }
    
    def save_progress(self, idx):
        """Save current progress"""
        self.progress.update({
            "last_processed_idx": idx,
            "last_update": datetime.now().isoformat()
        })
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def extract_json(self, response_text):
        """Extract JSON from Qwen response"""
        if not response_text or len(response_text.strip()) == 0:
            return None
        
        try:
            # Find JSON block in response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            
            if start == -1 or end <= start:
                return None
            
            json_str = response_text[start:end]
            data = json.loads(json_str)
            return data
        
        except Exception as e:
            return None
    
    def process_single_post(self, text, max_retries=2):
        """Process a single post with Qwen"""
        if not text or len(str(text).strip()) == 0:
            return {"error": "empty_text"}
        
        full_prompt = f"{SYSTEM_PROMPT}\\n\\nPost: \\\"{text}\\\"\\n\\nJSON:"
        
        for attempt in range(max_retries):
            try:
                result = subprocess.run(
                    ['ollama', 'run', 'qwen2.5:7b'],
                    input=full_prompt,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    timeout=30  # 30 second timeout
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    response_text = result.stdout.strip()
                    json_data = self.extract_json(response_text)
                    if json_data:
                        return json_data
                
                # If first attempt failed, wait before retry
                if attempt < max_retries - 1:
                    time.sleep(1)
                    
            except subprocess.TimeoutExpired:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    return {"error": "timeout"}
            except Exception as e:
                return {"error": str(e)[:100]}
        
        return {"error": "max_retries_exceeded"}
    
    def analyze_posts(self, start_idx=None, save_interval=50):
        """Analyze all posts in the dataset"""
        
        if start_idx is None:
            start_idx = self.progress["last_processed_idx"] + 1
        
        if self.progress["start_time"] is None:
            self.progress["start_time"] = datetime.now().isoformat()
        
        total_posts = len(self.df)
        
        print("\\n" + "="*80)
        print("QWEN ANALYSIS - WEIBO FILIAL PIETY SENTIMENT")
        print("="*80)
        print(f"Total posts to analyze: {total_posts}")
        print(f"Starting from index: {start_idx}")
        print(f"Posts remaining: {total_posts - start_idx}")
        
        # Estimate time
        if start_idx < total_posts:
            remaining_posts = total_posts - start_idx
            estimated_hours = (remaining_posts * 1.5) / 3600  # 1.5 seconds per post
            print(f"Estimated time: {estimated_hours:.1f} hours")
        
        print(f"Results will be saved to: {self.results_file}")
        print("\\nStarting analysis...")
        
        batch_start_time = time.time()
        last_save_time = time.time()
        
        # Process each post
        for i in tqdm(range(start_idx, total_posts), desc="Analyzing posts"):
            row = self.df.iloc[i]
            
            # Get text to analyze
            text = row.get('text', '') or row.get('cleaned_text', '')
            
            # Process with Qwen
            result = self.process_single_post(text)
            
            # Update dataframe with results
            if 'error' in result:
                self.df.at[i, 'qwen_error'] = result['error']
                self.progress["errors"] += 1
            else:
                self.df.at[i, 'qwen_relevant'] = result.get('relevant', None)
                self.df.at[i, 'qwen_sentiment'] = result.get('sentiment', None)
                self.df.at[i, 'qwen_bucket'] = result.get('bucket', None)
                self.df.at[i, 'qwen_reasoning'] = result.get('reasoning', None)
                self.progress["successful"] += 1
            
            self.df.at[i, 'qwen_processed_at'] = datetime.now().isoformat()
            self.progress["total_processed"] += 1
            
            # Save progress periodically
            current_time = time.time()
            if i % save_interval == 0 or current_time - last_save_time > 300:  # Every 50 posts or 5 minutes
                self.save_progress(i)
                self.save_results()
                last_save_time = current_time
                
                # Print progress statistics
                elapsed = current_time - batch_start_time
                posts_processed = i - start_idx + 1
                rate = posts_processed / elapsed if elapsed > 0 else 0
                
                print(f"\\nProgress Update (Index {i}):")
                print(f"  Posts processed: {posts_processed}/{total_posts - start_idx}")
                print(f"  Rate: {rate:.2f} posts/second")
                print(f"  Successful: {self.progress['successful']}")
                print(f"  Errors: {self.progress['errors']}")
                
                if rate > 0:
                    remaining = total_posts - i - 1
                    eta_seconds = remaining / rate
                    eta_hours = eta_seconds / 3600
                    print(f"  ETA: {eta_hours:.1f} hours")
        
        # Final save
        self.save_progress(total_posts - 1)
        self.save_results()
        
        print("\\n" + "="*80)
        print("ANALYSIS COMPLETED!")
        print("="*80)
        
        total_time = time.time() - batch_start_time
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Average rate: {total_posts/total_time:.2f} posts/second")
        print(f"Successful analyses: {self.progress['successful']}")
        print(f"Errors: {self.progress['errors']}")
        print(f"Success rate: {100 * self.progress['successful'] / total_posts:.1f}%")
        
        return self.df
    
    def save_results(self):
        """Save current results to CSV"""
        # Select columns for final output
        output_columns = [
            'post_id', 'time', 'year', 'month', 'text', 'text_length',
            'qwen_relevant', 'qwen_sentiment', 'qwen_bucket', 'qwen_reasoning',
            'qwen_error', 'qwen_processed_at'
        ]
        
        # Only include columns that exist in the dataframe
        available_columns = [col for col in output_columns if col in self.df.columns]
        
        # Save to CSV
        self.df[available_columns].to_csv(self.results_file, index=False, encoding='utf-8')
    
    def generate_summary_report(self):
        """Generate summary statistics of the analysis"""
        print("\\n" + "="*80)
        print("ANALYSIS SUMMARY REPORT")
        print("="*80)
        
        total_posts = len(self.df)
        processed_posts = self.df['qwen_processed_at'].notna().sum()
        successful_posts = self.df['qwen_relevant'].notna().sum()
        error_posts = self.df['qwen_error'].notna().sum()
        
        print(f"Total posts: {total_posts}")
        print(f"Processed posts: {processed_posts}")
        print(f"Successful analyses: {successful_posts}")
        print(f"Error posts: {error_posts}")
        print(f"Success rate: {100 * successful_posts / processed_posts:.1f}%")
        
        if successful_posts > 0:
            # Sentiment distribution
            print("\\nSentiment Distribution:")
            sentiment_counts = self.df['qwen_sentiment'].value_counts().sort_index()
            for sentiment, count in sentiment_counts.items():
                print(f"  {sentiment:2d}: {count:4d} ({100*count/successful_posts:.1f}%)")
            
            # Bucket distribution
            print("\\nContext Bucket Distribution:")
            bucket_counts = self.df['qwen_bucket'].value_counts()
            for bucket, count in bucket_counts.head(10).items():
                print(f"  {bucket:15s}: {count:4d} ({100*count/successful_posts:.1f}%)")
            
            # Relevance
            relevant_count = self.df['qwen_relevant'].sum()
            print(f"\\nRelevant posts: {relevant_count} ({100*relevant_count/successful_posts:.1f}%)")
            
            # Year trends
            print("\\nSentiment by Year:")
            yearly_sentiment = self.df.groupby('year')['qwen_sentiment'].agg(['mean', 'count'])
            for year, stats in yearly_sentiment.iterrows():
                if pd.notna(stats['mean']):
                    print(f"  {year}: {stats['mean']:5.2f} (n={stats['count']:3.0f})")


def main():
    """Main execution function"""
    
    # Configuration
    input_file = "weibo_xiao_sample_equal_per_month.csv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print("Please make sure weibo_xiao_sample_equal_per_month.csv exists.")
        return
    
    # Initialize analyzer
    analyzer = WeiboSampleAnalyzer(input_file)
    
    # Ask user for confirmation
    total_posts = len(analyzer.df)
    estimated_hours = (total_posts * 1.5) / 3600
    
    print(f"\\nReady to analyze {total_posts} posts")
    print(f"Estimated time: {estimated_hours:.1f} hours")
    print(f"This will save results to: {analyzer.results_file}")
    
    choice = input("\\nContinue with analysis? (y/n): ").strip().lower()
    
    if choice in ['y', 'yes']:
        # Run analysis
        analyzer.analyze_posts()
        
        # Generate summary report
        analyzer.generate_summary_report()
        
        print(f"\nAnalysis completed!")
        print(f"Results saved to: {analyzer.results_file}")
        print(f"Progress saved to: {analyzer.progress_file}")
        
    else:
        print("Analysis cancelled.")


if __name__ == "__main__":
    main()