"""
Efficient Qwen2.5 Batch Processing for Large Weibo Dataset
Processes posts in chunks with progress saving and error handling
"""

import pandas as pd
import json
import time
import os
from datetime import datetime
import subprocess
import requests
from tqdm import tqdm

# Shortened system prompt (from compare_models.py)
SYSTEM_PROMPT = """你是中国文化专家，评估微博对"孝"的态度。严格按照评分标准分类。

评分标准：
• 0 (无关): "孝"是地名/人名/书名，与内容无关
• 0 (中性): 娱乐/玩笑语境，无道德判断
• -1 (轻度负面): 表达压力/矛盾，但未否定孝道价值
• -2 (强烈负面): 强烈批评孝道限制自由，视为负担/债务
• +1 (轻度正面): 一般性赞扬孝道，无具体细节（如"孝顺很重要"、征婚提及孝顺）
• +2 (强烈正面): 详细描述具体孝行，或严厉谴责不孝

上下文分类：Reciprocity(情感互惠) | Obligation(责任义务) | Care(赡养照护) | Conflict(家庭冲突) | Critique/Abstract(理论探讨) | None(无关)

仅返回JSON格式：{"relevant": boolean, "sentiment": int, "bucket": "string", "reasoning": "string"}
"""


class QwenBatchProcessor:
    def __init__(self, data_file, output_dir="qwen_results", chunk_size=100):
        """
        Initialize batch processor
        
        Args:
            data_file: Path to weibo_xiao_cleaned.csv
            output_dir: Directory to save results
            chunk_size: Number of posts to process per batch
        """
        self.data_file = data_file
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.progress_file = os.path.join(output_dir, "progress.json")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        print("Loading data...")
        self.df = pd.read_csv(data_file)
        
        # Filter to only posts containing '孝'
        self.df_xiao = self.df[self.df['contains_xiao'] == True].copy()
        print(f"Loaded {len(self.df_xiao)} posts containing '孝'")
        
        # Load existing progress
        self.progress = self.load_progress()
    
    def load_progress(self):
        """Load processing progress from file"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {"last_processed_idx": -1, "total_processed": 0, "start_time": None}
    
    def save_progress(self, idx, total_processed):
        """Save current progress"""
        self.progress.update({
            "last_processed_idx": idx,
            "total_processed": total_processed,
            "last_update": datetime.now().isoformat()
        })
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def process_single_post(self, text, max_retries=2):
        """Process a single post with Qwen2.5"""
        if not text or len(text.strip()) == 0:
            return {"error": "empty_text"}
        
        full_prompt = f"{SYSTEM_PROMPT}\n\nPost: \"{text}\"\n\nJSON:"
        
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
                    # Try to parse JSON from response
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
    
    def extract_json(self, text):
        """Extract JSON from Qwen response"""
        try:
            # Find JSON in response
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = text[start:end]
                return json.loads(json_str)
        except:
            pass
        return None
    
    def process_batch(self, start_idx=None):
        """Process posts in batches"""
        if start_idx is None:
            start_idx = self.progress["last_processed_idx"] + 1
        
        if self.progress["start_time"] is None:
            self.progress["start_time"] = datetime.now().isoformat()
        
        total_posts = len(self.df_xiao)
        print(f"Starting batch processing from index {start_idx}")
        print(f"Total posts to process: {total_posts - start_idx}")
        
        results = []
        batch_start_time = time.time()
        
        # Process in chunks
        for i in tqdm(range(start_idx, total_posts, self.chunk_size), desc="Processing batches"):
            chunk_end = min(i + self.chunk_size, total_posts)
            chunk = self.df_xiao.iloc[i:chunk_end]
            
            chunk_results = []
            chunk_start = time.time()
            
            # Process each post in the chunk
            for idx, row in chunk.iterrows():
                post_result = {
                    'original_idx': idx,
                    'weibo_id': row['weibo_id'],
                    'time': row['time'],
                    'year': row['year'],
                    'month': row['month'],
                    'text_length': row['text_length'],
                    'cleaned_text': row['cleaned_text'][:200] + "..." if len(row['cleaned_text']) > 200 else row['cleaned_text']
                }
                
                # Process with Qwen
                qwen_result = self.process_single_post(row['cleaned_text'])
                post_result.update(qwen_result)
                
                chunk_results.append(post_result)
            
            # Save chunk results
            chunk_file = os.path.join(self.output_dir, f"chunk_{i:06d}_{chunk_end:06d}.json")
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_results, f, ensure_ascii=False, indent=2)
            
            # Update progress
            self.save_progress(chunk_end - 1, self.progress["total_processed"] + len(chunk))
            
            # Print statistics
            chunk_time = time.time() - chunk_start
            posts_per_second = len(chunk) / chunk_time if chunk_time > 0 else 0
            
            print(f"\\nChunk {i}-{chunk_end} completed:")
            print(f"  Time: {chunk_time:.1f}s ({posts_per_second:.2f} posts/sec)")
            print(f"  Progress: {chunk_end}/{total_posts} ({100*chunk_end/total_posts:.1f}%)")
            
            # Estimate time remaining
            if posts_per_second > 0:
                remaining_posts = total_posts - chunk_end
                estimated_remaining = remaining_posts / posts_per_second / 3600  # hours
                print(f"  Estimated time remaining: {estimated_remaining:.1f} hours")
        
        print("\\nBatch processing completed!")
        return True
    
    def consolidate_results(self):
        """Combine all chunk files into final results"""
        print("Consolidating results...")
        
        chunk_files = [f for f in os.listdir(self.output_dir) if f.startswith('chunk_') and f.endswith('.json')]
        chunk_files.sort()
        
        all_results = []
        for chunk_file in chunk_files:
            with open(os.path.join(self.output_dir, chunk_file), 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                all_results.extend(chunk_data)
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(all_results)
        output_file = os.path.join(self.output_dir, "qwen_analysis_results.csv")
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"Results consolidated to: {output_file}")
        print(f"Total results: {len(results_df)}")
        
        # Generate summary statistics
        self.generate_summary(results_df)
        
        return results_df
    
    def generate_summary(self, results_df):
        """Generate summary statistics"""
        print("\\n" + "="*60)
        print("QWEN ANALYSIS SUMMARY")
        print("="*60)
        
        # Error analysis
        error_mask = results_df.get('error', pd.Series([None]*len(results_df))).notna()
        successful = len(results_df) - error_mask.sum()
        
        print(f"Total posts processed: {len(results_df)}")
        print(f"Successful analyses: {successful}")
        print(f"Errors: {error_mask.sum()}")
        
        if successful > 0:
            # Sentiment distribution
            sentiment_counts = results_df['sentiment'].value_counts().sort_index()
            print(f"\\nSentiment distribution:")
            for sentiment, count in sentiment_counts.items():
                print(f"  {sentiment}: {count} ({100*count/successful:.1f}%)")
            
            # Bucket distribution  
            bucket_counts = results_df['bucket'].value_counts()
            print(f"\\nContext bucket distribution:")
            for bucket, count in bucket_counts.head(10).items():
                print(f"  {bucket}: {count}")
            
            # Year-wise sentiment trends
            year_sentiment = results_df.groupby('year')['sentiment'].mean()
            print(f"\\nAverage sentiment by year:")
            for year, avg_sentiment in year_sentiment.items():
                print(f"  {year}: {avg_sentiment:.2f}")


def main():
    """Main execution function"""
    print("Qwen2.5 Batch Processing for Weibo Dataset")
    print("="*50)
    
    # Configuration
    data_file = "weibo_xiao_cleaned.csv"
    chunk_size = 50  # Start small for testing
    
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!")
        return
    
    # Initialize processor
    processor = QwenBatchProcessor(data_file, chunk_size=chunk_size)
    
    # Ask user for processing strategy
    print(f"Found {len(processor.df_xiao)} posts containing '孝'")
    print("\\nProcessing options:")
    print("1. Full processing (WARNING: Will take days)")
    print("2. Sample processing (1000 random posts)")
    print("3. Year range processing (specify years)")
    print("4. Resume from last checkpoint")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        # Full processing
        confirmation = input("This will process 1M+ posts. Continue? (yes/no): ")
        if confirmation.lower() == 'yes':
            processor.process_batch()
    
    elif choice == "2":
        # Sample processing
        sample_size = int(input("Enter sample size (default 1000): ") or "1000")
        sample_df = processor.df_xiao.sample(n=min(sample_size, len(processor.df_xiao)))
        processor.df_xiao = sample_df
        processor.process_batch(start_idx=0)
    
    elif choice == "3":
        # Year range processing
        start_year = int(input("Start year (2016-2023): "))
        end_year = int(input("End year (2016-2023): "))
        year_mask = (processor.df_xiao['year'] >= start_year) & (processor.df_xiao['year'] <= end_year)
        processor.df_xiao = processor.df_xiao[year_mask]
        print(f"Processing {len(processor.df_xiao)} posts from {start_year}-{end_year}")
        processor.process_batch(start_idx=0)
    
    elif choice == "4":
        # Resume processing
        processor.process_batch()
    
    # Consolidate results
    processor.consolidate_results()


if __name__ == "__main__":
    main()