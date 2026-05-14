"""
Cascade Classification Filter - Strict Post Classification
Takes weibo_xiao_relevant_only.csv and applies strict filtering:
1. Must contain '孝' character
2. Confidence-based classification into Keep/Maybe/Drop
"""

import pandas as pd
import json
import subprocess
import time
import os
from datetime import datetime
from tqdm import tqdm

# Strict relevance prompt with confidence scoring
STRICT_RELEVANCE_PROMPT = """你是中国文化专家。严格判断这条微博是否真正讨论"孝道"相关话题。

判断标准：
- 必须包含'孝'字
- 必须实际讨论孝顺、孝道、赡养父母等话题
- 不能只是人名、地名、书名中的'孝'字

返回JSON格式：
{
  "relevant": true/false,
  "confidence": 0-100,
  "reason": "简短理由（中文，20字以内）"
}

confidence评分标准：
- 90-100: 明确讨论孝道主题
- 70-89: 可能相关，但不太确定
- 50-69: 模糊，难以判断
- 0-49: 明显不相关

微博文本："""


class CascadeFilter:
    def __init__(self, 
                 input_file="data/relevant_only/weibo_xiao_relevant_only.csv",
                 output_keep="data/weibo_xiao_strict_keep.csv",
                 output_maybe="data/weibo_xiao_strict_maybe.csv", 
                 output_drop="data/weibo_xiao_strict_drop.csv",
                 model="qwen2.5:7b"):
        self.input_file = input_file
        self.output_keep = output_keep
        self.output_maybe = output_maybe
        self.output_drop = output_drop
        self.progress_file = "results/cascade_filter_progress.json"
        self.model = model
        
        print(f"Loading dataset: {input_file}")
        self.df = pd.read_csv(input_file)
        print(f"Loaded {len(self.df)} posts")
        
        # Add cascade filter columns
        if 'cascade_relevant' not in self.df.columns:
            self.df['cascade_relevant'] = None
        if 'cascade_confidence' not in self.df.columns:
            self.df['cascade_confidence'] = None
        if 'cascade_reason' not in self.df.columns:
            self.df['cascade_reason'] = None
        if 'cascade_category' not in self.df.columns:
            self.df['cascade_category'] = None  # 'keep', 'maybe', 'drop'
        if 'cascade_processed_at' not in self.df.columns:
            self.df['cascade_processed_at'] = None
        
        self.progress = self.load_progress()
    
    def load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "last_idx": -1, 
            "processed": 0, 
            "keep_count": 0,
            "maybe_count": 0,
            "drop_count": 0
        }
    
    def save_progress(self, idx):
        self.progress["last_idx"] = int(idx)
        self.progress["last_update"] = datetime.now().isoformat()
        os.makedirs("results", exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def has_xiao_character(self, text):
        """Check if text contains the character '孝'"""
        if not isinstance(text, str):
            return False
        return '孝' in text
    
    def check_relevance_strict(self, text):
        """Check relevance with strict criteria and confidence score"""
        if not text or len(str(text).strip()) == 0:
            return {"relevant": False, "confidence": 0, "reason": "空文本"}
        
        # First check: must contain '孝' character
        if not self.has_xiao_character(text):
            return {"relevant": False, "confidence": 0, "reason": "不含'孝'字"}
        
        # Clean and limit text
        clean_text = str(text).replace('"', '\\"')[:500]
        prompt = f"{STRICT_RELEVANCE_PROMPT}{clean_text}\n\nJSON:"
        
        try:
            result = subprocess.run(
                ['ollama', 'run', self.model],
                input=prompt,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=30
            )
            
            if result.returncode != 0:
                return {"relevant": None, "confidence": None, "reason": "LLM调用失败"}
            
            response = result.stdout.strip()
            
            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    return {
                        "relevant": bool(data.get("relevant", False)),
                        "confidence": int(data.get("confidence", 50)),
                        "reason": str(data.get("reason", ""))[:100]
                    }
                except:
                    pass
            
            # Fallback: parse manually
            relevant = "true" in response.lower()
            conf_match = re.search(r'"?confidence"?\s*[:：]\s*(\d+)', response)
            confidence = int(conf_match.group(1)) if conf_match else 50
            
            reason_match = re.search(r'"?reason"?\s*[:：]\s*"?([^"}\n]+)"?', response)
            reason = reason_match.group(1).strip() if reason_match else ""
            
            return {
                "relevant": relevant,
                "confidence": confidence,
                "reason": reason[:100]
            }
            
        except subprocess.TimeoutExpired:
            return {"relevant": None, "confidence": None, "reason": "超时"}
        except Exception as e:
            return {"relevant": None, "confidence": None, "reason": f"错误: {str(e)[:50]}"}
    
    def classify_post(self, relevant, confidence):
        """
        Classify posts into keep/maybe/drop based on relevance and confidence
        
        Keep: relevant=True AND confidence >= 80
        Maybe: relevant=True AND 50 <= confidence < 80, OR relevant=None
        Drop: relevant=False OR (relevant=True AND confidence < 50)
        """
        if relevant is None:
            return "maybe"
        
        if relevant is True:
            if confidence >= 80:
                return "keep"
            elif confidence >= 50:
                return "maybe"
            else:
                return "drop"
        else:  # relevant is False
            return "drop"
    
    def filter_dataset(self):
        """Process all posts with cascade filtering"""
        start_idx = self.progress["last_idx"] + 1
        
        if start_idx >= len(self.df):
            print("All posts already processed!")
            return
        
        # Show existing progress if resuming
        if start_idx > 0:
            print(f"\n RESUMING FROM PREVIOUS SESSION")
            print(f"   Already processed: {self.progress['processed']} posts")
            print(f"   Keep: {self.progress['keep_count']}")
            print(f"   Maybe: {self.progress['maybe_count']}")
            print(f"   Drop: {self.progress['drop_count']}")
            print(f"   Last update: {self.progress.get('last_update', 'N/A')}")
        
        print(f"\nStarting from index {start_idx}")
        print(f"Processing {len(self.df) - start_idx} remaining posts")
        print(f"Model: {self.model}")
        print("Classification criteria:")
        print("  Keep: relevant=True AND confidence >= 80")
        print("  Maybe: relevant=True AND 50 <= confidence < 80, OR uncertain")
        print("  Drop: relevant=False OR confidence < 50")
        print("\n Press Ctrl+C anytime to stop - progress will be saved!")
        
        pbar = tqdm(total=len(self.df) - start_idx, desc="Cascade filtering")
        
        try:
            for idx in range(start_idx, len(self.df)):
                text = self.df.loc[idx, 'text']
                
                # Check relevance with strict criteria
                result = self.check_relevance_strict(text)
                
                # Classify into keep/maybe/drop
                category = self.classify_post(result["relevant"], result["confidence"])
                
                # Update dataframe
                timestamp = datetime.now().isoformat()
                self.df.loc[idx, 'cascade_relevant'] = result["relevant"]
                self.df.loc[idx, 'cascade_confidence'] = result["confidence"]
                self.df.loc[idx, 'cascade_reason'] = result["reason"]
                self.df.loc[idx, 'cascade_category'] = category
                self.df.loc[idx, 'cascade_processed_at'] = timestamp
                
                # Update counts
                self.progress["processed"] += 1
                if category == "keep":
                    self.progress["keep_count"] += 1
                elif category == "maybe":
                    self.progress["maybe_count"] += 1
                elif category == "drop":
                    self.progress["drop_count"] += 1
                
                # Save progress every 10 posts
                if (idx + 1) % 10 == 0:
                    self.save_progress(idx)
                
                # Save results every 100 posts
                if (idx + 1) % 100 == 0:
                    self.save_results()
                
                pbar.update(1)
        
        except KeyboardInterrupt:
            pbar.close()
            print(f"\n\n⚠️  Interrupted by user (Ctrl+C)")
            print(f"Saving progress at index {idx}...")
            self.save_progress(idx)
            self.save_results()
            print(f"✅ Progress saved successfully!")
            print(f"\n📊 Current status:")
            print(f"   Processed: {self.progress['processed']} posts")
            print(f"   Keep: {self.progress['keep_count']} posts")
            print(f"   Maybe: {self.progress['maybe_count']} posts")
            print(f"   Drop: {self.progress['drop_count']} posts")
            print(f"\n💡 Run the script again to resume from where you left off.")
            return
        
        pbar.close()
        print(f"\n✅ Completed! Processed {self.progress['processed']} posts")
        print(f"   Keep: {self.progress['keep_count']} posts")
        print(f"   Maybe: {self.progress['maybe_count']} posts")
        print(f"   Drop: {self.progress['drop_count']} posts")
    
    def save_results(self):
        """Save filtered posts into separate CSV files"""
        # Keep: high confidence relevant posts
        keep_df = self.df[self.df['cascade_category'] == 'keep'].copy()
        keep_df.to_csv(self.output_keep, index=False)
        
        # Maybe: medium confidence or uncertain
        maybe_df = self.df[self.df['cascade_category'] == 'maybe'].copy()
        maybe_df.to_csv(self.output_maybe, index=False)
        
        # Drop: low confidence or irrelevant
        drop_df = self.df[self.df['cascade_category'] == 'drop'].copy()
        drop_df.to_csv(self.output_drop, index=False)
        
        print(f"\n📁 Saved results:")
        print(f"   Keep ({len(keep_df)}): {self.output_keep}")
        print(f"   Maybe ({len(maybe_df)}): {self.output_maybe}")
        print(f"   Drop ({len(drop_df)}): {self.output_drop}")
    
    def reset_progress(self):
        """Reset progress to start from scratch"""
        print("\n⚠️  RESET PROGRESS")
        print("This will delete progress tracking (but NOT output files).")
        confirm = input("Are you sure? Type 'yes' to confirm: ").strip().lower()
        
        if confirm != 'yes':
            print("Reset cancelled.")
            return False
        
        if os.path.exists(self.progress_file):
            try:
                os.remove(self.progress_file)
                print(f"✓ Removed: {self.progress_file}")
            except Exception as e:
                print(f"✗ Failed to remove {self.progress_file}: {e}")
                return False
        else:
            print(f"• Progress file not found: {self.progress_file}")
        
        print(f"\n✅ Reset complete!")
        print("Run the script again to start fresh classification.")
        return True


def main():
    filter_obj = CascadeFilter(
        input_file="data/relevant_only/weibo_xiao_relevant_only.csv",
        output_keep="data/weibo_xiao_strict_keep.csv",
        output_maybe="data/weibo_xiao_strict_maybe.csv",
        output_drop="data/weibo_xiao_strict_drop.csv",
        model="qwen2.5:7b"
    )
    
    print("\n" + "="*60)
    print("CASCADE CLASSIFICATION FILTER")
    print("="*60)
    print(f"Input: {filter_obj.input_file}")
    print(f"Total posts: {len(filter_obj.df)}")
    print("\nOutput files:")
    print(f"  Keep: {filter_obj.output_keep}")
    print(f"  Maybe: {filter_obj.output_maybe}")
    print(f"  Drop: {filter_obj.output_drop}")
    print("\nCriteria:")
    print("  1. Must contain '孝' character")
    print("  2. Keep: confidence >= 80")
    print("  3. Maybe: 50 <= confidence < 80 OR uncertain")
    print("  4. Drop: confidence < 50 OR not relevant")
    print("="*60 + "\n")
    
    # Check if there's existing progress
    if filter_obj.progress["last_idx"] >= 0:
        print("⚠️  Found existing progress!")
        choice = input("Continue from where you left off? (y/n/reset): ").strip().lower()
        if choice == 'reset':
            if filter_obj.reset_progress():
                print("Please run the script again to start fresh.")
                return
        elif choice == 'n':
            print("Exiting. Run with different settings if needed.")
            return
        # Otherwise continue (choice == 'y' or default)
    
    start_time = time.time()
    filter_obj.filter_dataset()
    elapsed = time.time() - start_time
    
    print(f"\n✅ Total time: {elapsed/3600:.2f} hours")
    print(f"   Average per post: {elapsed/len(filter_obj.df):.2f}s")
    
    # Final save
    filter_obj.save_results()
    
    # Show summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    processed = filter_obj.df[filter_obj.df['cascade_processed_at'].notna()]
    if len(processed) > 0:
        keep_pct = 100 * filter_obj.progress['keep_count'] / len(processed)
        maybe_pct = 100 * filter_obj.progress['maybe_count'] / len(processed)
        drop_pct = 100 * filter_obj.progress['drop_count'] / len(processed)
        
        print(f"Processed: {len(processed):,} posts")
        print(f"\nBreakdown:")
        print(f"  Keep:  {filter_obj.progress['keep_count']:,} ({keep_pct:.1f}%)")
        print(f"  Maybe: {filter_obj.progress['maybe_count']:,} ({maybe_pct:.1f}%)")
        print(f"  Drop:  {filter_obj.progress['drop_count']:,} ({drop_pct:.1f}%)")
        
        # Confidence distribution for keep category
        keep_posts = processed[processed['cascade_category'] == 'keep']
        if len(keep_posts) > 0:
            avg_conf = keep_posts['cascade_confidence'].mean()
            print(f"\nKeep posts average confidence: {avg_conf:.1f}")


if __name__ == "__main__":
    main()
