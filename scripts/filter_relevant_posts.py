"""
Quick Relevance Filter - Determine which posts are actually about filial piety
Uses Qwen to classify relevant=True/False only, no sentiment analysis
Batches 5 posts per LLM call for optimal speed (1.13s per post)
Sequential processing - parallel doesn't help due to Ollama bottleneck
"""

import pandas as pd
import json
import subprocess
import time
import os
from datetime import datetime
from tqdm import tqdm

# Batch processing prompt - handles multiple posts at once
RELEVANCE_PROMPT = """你是中国文化专家。判断以下每条微博是否真正讨论"孝"（孝顺、孝道、孝敬父母等）。

相关 (true): 微博实际讨论孝道、孝顺父母、赡养老人等话题
无关 (false): "孝"仅是人名/地名/书名，或与孝道无关

返回JSON数组，每个微博一个结果：[{"id": 1, "relevant": true}, {"id": 2, "relevant": false}, ...]"""


class RelevanceFilter:
    def __init__(self, input_file="data/weibo_xiao_sample_equal_per_month.csv",
                 output_file="data/relevant_only/weibo_xiao_relevant_only.csv",
                 batch_size=15,
                 char_limit=150,
                 model="qwen2.5:3b"):
        self.input_file = input_file
        self.output_file = output_file
        self.progress_file = "results/relevance_filter_progress.json"
        self.batch_size = batch_size  # Number of posts per LLM call
        self.char_limit = char_limit  # Max chars per post
        self.model = model  # Ollama model to use
        
        print(f"Loading dataset: {input_file}")
        self.df = pd.read_csv(input_file)
        print(f"Loaded {len(self.df)} posts")
        print(f"Model: {model}")
        print(f"Batch size: {batch_size} posts per LLM call")
        print(f"Char limit: {char_limit} chars per post")
        
        # Add relevance column
        if 'is_relevant' not in self.df.columns:
            self.df['is_relevant'] = None
        if 'processed_at' not in self.df.columns:
            self.df['processed_at'] = None
        
        self.progress = self.load_progress()
    
    def load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"last_idx": -1, "processed": 0, "relevant_count": 0}
    
    def save_progress(self, idx):
        self.progress["last_idx"] = int(idx)
        self.progress["last_update"] = datetime.now().isoformat()
        os.makedirs("results", exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def check_batch_relevance(self, batch_texts):
        """Check relevance for a batch of posts - returns dict {index: True/False/None}"""
        if not batch_texts:
            return {}
        
        # Build prompt with numbered posts
        posts_text = ""
        for i, text in enumerate(batch_texts, 1):
            if not text or len(str(text).strip()) == 0:
                continue
            clean_text = str(text).replace('"', '\\"')[:self.char_limit]
            posts_text += f"\n{i}. {clean_text}"
        
        prompt = f"{RELEVANCE_PROMPT}\n{posts_text}\n\nJSON:"
        
        try:
            result = subprocess.run(
                ['ollama', 'run', self.model],
                input=prompt,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=30  # Longer timeout for batch processing
            )
            
            if result.returncode != 0:
                return {i: None for i in range(len(batch_texts))}
            
            response = result.stdout.strip()
            
            # Try to extract JSON array
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    results = {}
                    for item in data:
                        if isinstance(item, dict) and 'id' in item and 'relevant' in item:
                            idx = item['id'] - 1  # Convert 1-based to 0-based
                            if 0 <= idx < len(batch_texts):
                                results[idx] = bool(item['relevant'])
                    return results
                except:
                    pass
            
            # Fallback: return None for all
            return {i: None for i in range(len(batch_texts))}
            
        except subprocess.TimeoutExpired:
            return {i: None for i in range(len(batch_texts))}
        except Exception as e:
            print(f"Error in batch processing: {e}")
            return {i: None for i in range(len(batch_texts))}
    
    def filter_dataset(self):
        """Process all posts in batches sequentially"""
        start_idx = self.progress["last_idx"] + 1
        
        if start_idx >= len(self.df):
            print("All posts already processed!")
            return
        
        print(f"\nStarting from index {start_idx}")
        print(f"Processing {len(self.df) - start_idx} remaining posts in batches of {self.batch_size}")
        
        # Calculate total batches
        total_batches = (len(self.df) - start_idx + self.batch_size - 1) // self.batch_size
        print(f"Total batches: {total_batches}")
        
        pbar = tqdm(total=len(self.df) - start_idx, desc="Filtering posts")
        
        batch_start = start_idx
        while batch_start < len(self.df):
            batch_end = min(batch_start + self.batch_size, len(self.df))
            batch_indices = list(range(batch_start, batch_end))
            
            # Get texts for this batch
            batch_texts = [self.df.loc[idx, 'text'] for idx in batch_indices]
            
            # Check relevance for entire batch
            results = self.check_batch_relevance(batch_texts)
            
            # Update dataframe
            timestamp = datetime.now().isoformat()
            for batch_idx, df_idx in enumerate(batch_indices):
                relevance = results.get(batch_idx)
                self.df.loc[df_idx, 'is_relevant'] = relevance
                self.df.loc[df_idx, 'processed_at'] = timestamp
                
                if relevance is True:
                    self.progress["relevant_count"] += 1
                
                self.progress["processed"] += 1
            
            # Save progress after each batch
            self.save_progress(batch_end - 1)
            
            # Save results periodically
            if batch_start % (self.batch_size * 10) == 0:  # Every 10 batches
                self.save_results()
            
            pbar.update(len(batch_indices))
            batch_start = batch_end
        
        pbar.close()
        print(f"\n✅ Completed! Processed {self.progress['processed']} posts")
        print(f"Found {self.progress['relevant_count']} relevant posts")
    
    def save_results(self):
        """Save filtered relevant posts to output file"""
        relevant_df = self.df[self.df['is_relevant'] == True].copy()
        relevant_df.to_csv(self.output_file, index=False)
        print(f"Saved {len(relevant_df)} relevant posts to {self.output_file}")


def main():
    filter_obj = RelevanceFilter(
        input_file="data/weibo_xiao_sample_equal_per_month.csv",
        output_file="data/relevant_only/weibo_xiao_relevant_only.csv",
        batch_size=15,  # 15 posts per batch
        char_limit=150,  # 150 chars per post
        model="qwen2.5:7b"  # 7b is faster than 3b for batch JSON (~47 hours)
    )
    
    print("\n" + "="*60)
    print("RELEVANCE FILTERING - Optimized Batch Processing")
    print("="*60)
    print(f"Input: {filter_obj.input_file}")
    print(f"Output: {filter_obj.output_file}")
    print(f"Model: {filter_obj.model}")
    print(f"Batch size: {filter_obj.batch_size} posts per call")
    print(f"Char limit: {filter_obj.char_limit} chars per post")
    print(f"Total posts: {len(filter_obj.df)}")
    
    # Estimate based on measured 0.91s per post with 7b model
    total_batches = (len(filter_obj.df) + filter_obj.batch_size - 1) // filter_obj.batch_size
    print(f"Estimated batches: {total_batches}")
    print(f"Estimated time: ~47 hours (0.91s per post, ~2 days)")
    print("="*60 + "\n")
    
    start_time = time.time()
    filter_obj.filter_dataset()
    elapsed = time.time() - start_time
    
    print(f"\n✅ Total time: {elapsed/3600:.2f} hours")
    print(f"Average per batch: {elapsed/total_batches:.2f}s")
    print(f"Average per post: {elapsed/len(filter_obj.df):.2f}s")
    
    # Final save
    filter_obj.save_results()


if __name__ == "__main__":
    main()
