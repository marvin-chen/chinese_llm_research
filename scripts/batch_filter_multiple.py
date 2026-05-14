"""
Batch Relevance Filter - Process multiple datasets with different keywords
Automatically handles zhong, nvzhunei, nanzhuwai, yugongyishan with custom prompts
Sequential processing (parallel won't help - Ollama is bottleneck)
"""

import pandas as pd
import json
import subprocess
import time
import os
from datetime import datetime
from tqdm import tqdm


class MultiFileRelevanceFilter:
    """Process multiple files with different relevance prompts"""
    
    def __init__(self, batch_size=15, char_limit=150, model="qwen2.5:7b"):
        self.batch_size = batch_size
        self.char_limit = char_limit
        self.model = model
        
        # Configuration: (input_file, output_file, keyword, prompt)
        self.configs = [
            {
                "name": "zhong",
                "input_file": "../data/忠_sample_equal_per_month.csv",
                "output_file": "../data/relevant_only/忠_relevant_only.csv",
                "keyword": "忠",
                "prompt": """你是中国文化专家。判断以下每条微博是否真正讨论"忠"（忠诚、忠心、忠于等）。

相关 (true): 微博实际讨论忠诚、忠心、忠于某人某事等话题
无关 (false): "忠"仅是人名/地名/书名，或与忠的品质无关

返回JSON数组，每个微博一个结果：[{"id": 1, "relevant": true}, {"id": 2, "relevant": false}, ...]"""
            },
            {
                "name": "nvzhunei",
                "input_file": "../data/女主内_cleaned.csv",
                "output_file": "../data/relevant_only/女主内_relevant_only.csv",
                "keyword": "女主内",
                "prompt": """你是中国文化专家。判断以下每条微博是否真正讨论"女主内"（女性主要在家庭中的角色、传统性别分工等）。

相关 (true): 微博实际讨论女性在家庭中的角色、传统性别分工、家务管理等话题
无关 (false): "女主内"仅是人名/地名/书名，或与家庭性别角色无关

返回JSON数组，每个微博一个结果：[{"id": 1, "relevant": true}, {"id": 2, "relevant": false}, ...]"""
            },
            {
                "name": "nanzhuwai",
                "input_file": "../data/男主外_cleaned.csv",
                "output_file": "../data/relevant_only/男主外_relevant_only.csv",
                "keyword": "男主外",
                "prompt": """你是中国文化专家。判断以下每条微博是否真正讨论"男主外"（男性主要在家庭外工作、传统性别分工等）。

相关 (true): 微博实际讨论男性的经济角色、传统性别分工、工作职责等话题
无关 (false): "男主外"仅是人名/地名/书名，或与家庭性别角色无关

返回JSON数组，每个微博一个结果：[{"id": 1, "relevant": true}, {"id": 2, "relevant": false}, ...]"""
            },
            {
                "name": "yugongyishan",
                "input_file": "../data/愚公移山_cleaned.csv",
                "output_file": "../data/relevant_only/愚公移山_relevant_only.csv",
                "keyword": "愚公移山",
                "prompt": """你是中国文化专家。判断以下每条微博是否真正讨论"愚公移山"（持之以恒、坚持不懈等）。

相关 (true): 微博实际讨论愚公移山故事、坚持不懈、持之以恒等话题
无关 (false): "愚公移山"仅是人名/地名/书名，或与坚持精神无关

返回JSON数组，每个微博一个结果：[{"id": 1, "relevant": true}, {"id": 2, "relevant": false}, ...]"""
            }
        ]
    
    def check_batch_relevance(self, batch_texts, prompt):
        """Check relevance for a batch of posts with given prompt"""
        if not batch_texts:
            return {}
        
        # Build prompt with numbered posts
        posts_text = ""
        for i, text in enumerate(batch_texts, 1):
            if not text or len(str(text).strip()) == 0:
                continue
            clean_text = str(text).replace('"', '\\"')[:self.char_limit]
            posts_text += f"\n{i}. {clean_text}"
        
        full_prompt = f"{prompt}\n{posts_text}\n\nJSON:"
        
        try:
            result = subprocess.run(
                ['ollama', 'run', self.model],
                input=full_prompt,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=30
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
                            idx = item['id'] - 1
                            if 0 <= idx < len(batch_texts):
                                results[idx] = bool(item['relevant'])
                    return results
                except:
                    pass
            
            return {i: None for i in range(len(batch_texts))}
            
        except subprocess.TimeoutExpired:
            return {i: None for i in range(len(batch_texts))}
        except Exception as e:
            print(f"Error in batch processing: {e}")
            return {i: None for i in range(len(batch_texts))}
    
    def process_file(self, config):
        """Process a single file with its custom prompt"""
        print(f"\n{'='*70}")
        print(f"Processing: {config['name'].upper()}")
        print(f"Input: {config['input_file']}")
        print(f"Output: {config['output_file']}")
        print(f"{'='*70}")
        
        # Load data
        try:
            df = pd.read_csv(config['input_file'])
            print(f"Loaded {len(df)} posts")
        except FileNotFoundError:
            print(f"❌ File not found: {config['input_file']}")
            return False
        
        # Add relevance columns if needed
        if 'is_relevant' not in df.columns:
            df['is_relevant'] = None
        if 'processed_at' not in df.columns:
            df['processed_at'] = None
        
        # Process in batches
        print(f"Processing in batches of {self.batch_size}...")
        total_batches = (len(df) + self.batch_size - 1) // self.batch_size
        relevant_count = 0
        
        pbar = tqdm(total=len(df), desc=f"Filtering {config['name']}")
        
        for batch_start in range(0, len(df), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(df))
            batch_indices = list(range(batch_start, batch_end))
            
            # Get texts for this batch
            batch_texts = [df.loc[idx, 'text'] for idx in batch_indices]
            
            # Check relevance with custom prompt
            results = self.check_batch_relevance(batch_texts, config['prompt'])
            
            # Update dataframe
            timestamp = datetime.now().isoformat()
            for batch_idx, df_idx in enumerate(batch_indices):
                relevance = results.get(batch_idx)
                df.loc[df_idx, 'is_relevant'] = relevance
                df.loc[df_idx, 'processed_at'] = timestamp
                
                if relevance is True:
                    relevant_count += 1
            
            pbar.update(len(batch_indices))
        
        pbar.close()
        
        # Save results
        relevant_df = df[df['is_relevant'] == True].copy()
        relevant_df.to_csv(config['output_file'], index=False)
        
        print(f"\n✅ {config['name'].upper()} complete!")
        print(f"   Processed: {len(df)} posts")
        print(f"   Found relevant: {relevant_count} posts")
        print(f"   Saved to: {config['output_file']}")
        
        return True
    
    def process_all(self):
        """Process all configured files sequentially"""
        start_time = time.time()
        
        print("\n" + "="*70)
        print("MULTI-FILE BATCH RELEVANCE FILTERING")
        print("="*70)
        print(f"Model: {self.model}")
        print(f"Batch size: {self.batch_size} posts per call")
        print(f"Files to process: {len(self.configs)}")
        print(f"Sequential processing (Ollama bottleneck)\n")
        
        successful = 0
        for config in self.configs:
            if self.process_file(config):
                successful += 1
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*70)
        print(f"ALL PROCESSING COMPLETE!")
        print(f"Processed {successful}/{len(self.configs)} files successfully")
        print(f"Total time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
        print("="*70)


def main():
    # Can customize batch_size, char_limit, and model here
    processor = MultiFileRelevanceFilter(
        batch_size=15,
        char_limit=150,
        model="qwen2.5:7b"
    )
    
    processor.process_all()


if __name__ == "__main__":
    main()
