"""
Batch Qwen Analysis with Resume Capability - Improved Version
Processes weibo_xiao_sample_equal_per_month.csv with smart batching and resume
"""

import pandas as pd
import json
import subprocess
import time
import os
from datetime import datetime
from tqdm import tqdm

# Enhanced system prompt for sentiment analysis (posts already filtered for relevance)
SYSTEM_PROMPT = """你是中国文化专家，评估微博对"孝"的态度。所有微博都已确认与孝道相关。

评分标准：
• -2 (强烈负面): 强烈批评孝道限制自由，视为负担/债务
• -1 (轻度负面): 表达压力/矛盾，但未否定孝道价值
• 0 (中性): 娱乐/玩笑语境，无道德判断
• +1 (轻度正面): 一般性赞扬孝道，无具体细节（如"孝顺很重要"、征婚提及孝顺）
• +2 (强烈正面): 详细描述具体孝行，或严厉谴责不孝

关键区别：
+1 vs +2: 一般性言论 vs 具体详细例子
-1 vs -2: 矛盾压力 vs 强烈批评

上下文分类（必须从以下5个选项中选1个，如果sentiment=0则bucket为空字符串）：
• 日常实践 (日常生活中对父母的关爱照顾，包括情感交流和物质照料)
• 责任义务 (强调孝道作为道德责任、社会规范或应尽义务)
• 家庭冲突 (因孝道引发的家庭矛盾、代际冲突)
• 理论探讨 (对孝道概念的抽象讨论、批判分析或文化评论)
• 婚恋择偶 (择偶婚恋中对孝顺的要求或讨论)

置信度评分（百分比0-100%）：
• 90-100%: 非常确信，文本明确表达态度
• 70-89%: 比较确信，有充分证据支持判断  
• 50-69%: 一般确信，基于常见模式判断
• 30-49%: 不太确信，可能有歧义
• 0-29%: 很不确信，难以判断

仅返回JSON格式：{"sentiment": int, "bucket": "日常实践或责任义务或家庭冲突或理论探讨或婚恋择偶或空字符串", "confidence": int}
"""

class BatchWeiboAnalyzer:
    def __init__(self, input_file, output_prefix="qwen_analysis"):
        self.input_file = input_file
        self.output_prefix = output_prefix
        self.progress_file = f"results/{output_prefix}_progress.json"
        self.results_file = f"results/{output_prefix}_results.csv"
        
        # Load data (assumed to be already filtered for relevance)
        print(f"Loading dataset: {input_file}")
        self.df = pd.read_csv(input_file)
        print(f"Loaded {len(self.df)} posts (pre-filtered for relevance)")
        
        # Initialize Qwen result columns with proper dtypes (if not exist)
        qwen_column_types = {
            'qwen_sentiment': 'Int64',  # Nullable integer for -2 to +2
            'qwen_bucket': 'object',     # String for bucket names
            'qwen_confidence': 'Int64',  # Nullable integer for 0-100 
            'qwen_reasoning': 'object',  # String for reasoning text
            'qwen_error': 'object',      # String for error messages
            'qwen_processed_at': 'object'  # String for timestamp
        }
        for col, dtype in qwen_column_types.items():
            if col not in self.df.columns:
                self.df[col] = pd.Series(dtype=dtype)
        
        # Load existing progress and results
        self.progress = self.load_progress()
        self.load_existing_results()
    
    def load_progress(self):
        """Load processing progress from file"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"WARNING: Progress file corrupted or unreadable: {e}")
                print("Starting fresh...")
                # Remove corrupted file
                try:
                    os.remove(self.progress_file)
                except:
                    pass
        
        return {
            "last_processed_idx": -1,
            "total_processed": 0,
            "successful": 0,
            "errors": 0,
            "start_time": None,
            "sessions": []
        }
    
    def save_progress(self, idx):
        """Save current progress"""
        self.progress.update({
            "last_processed_idx": int(idx),  # Convert to regular Python int
            "last_update": datetime.now().isoformat()
        })
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)
    
    def load_existing_results(self):
        """Load and merge existing results from previous runs"""
        if os.path.exists(self.results_file):
            try:
                print(f"Found existing results file: {self.results_file}")
                existing_df = pd.read_csv(self.results_file)
                
                # Count processed posts
                processed_count = existing_df['qwen_processed_at'].notna().sum()
                successful_count = existing_df['qwen_relevant'].notna().sum()
                error_count = existing_df['qwen_error'].notna().sum()
                
                print(f"   Previously processed: {processed_count} posts")
                print(f"   Successful: {successful_count}, Errors: {error_count}")
                
                # Merge results back to main df
                if 'post_id' in existing_df.columns and 'post_id' in self.df.columns:
                    merge_cols = ['post_id'] + [col for col in existing_df.columns if col.startswith('qwen_')]
                    self.df = self.df.drop(columns=[col for col in merge_cols[1:] if col in self.df.columns])
                    self.df = self.df.merge(existing_df[merge_cols], on='post_id', how='left')
                
                # Update progress counters (convert pandas int64 to regular Python ints)
                self.progress.update({
                    "total_processed": int(processed_count),
                    "successful": int(successful_count),
                    "errors": int(error_count)
                })
                
                # Find last processed index
                if processed_count > 0:
                    last_processed_mask = self.df['qwen_processed_at'].notna()
                    if last_processed_mask.any():
                        self.progress["last_processed_idx"] = int(self.df[last_processed_mask].index.max())  # Convert to regular Python int
                
                print(f"Merged existing results")
                
                # Show confidence distribution if available
                if 'qwen_confidence' in existing_df.columns:
                    conf_counts = existing_df['qwen_confidence'].value_counts().sort_index()
                    if len(conf_counts) > 0:
                        print(f"   Confidence distribution: {dict(conf_counts)}")
                
            except Exception as e:
                print(f"WARNING: Could not load existing results: {str(e)}")
    
    def extract_json_robust(self, text):
        """Extract JSON with multiple fallback strategies"""
        if not text:
            return None
        
        # Strategy 1: Find JSON block
        import re
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',  # Markdown json block
            r'```\s*(\{.*?\})\s*```',      # Markdown block without json label  
            r'(\{[^{}]*"relevant"[^{}]*\})', # Simple JSON with "relevant"
            r'(\{.*?\})',                   # Any JSON-like structure
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match.strip())
                except:
                    continue
        
        # Strategy 2: Parse field by field
        try:
            result = {}
            if 'true' in text.lower() or 'false' in text.lower():
                result['relevant'] = 'true' in text.lower()
            
            # Look for sentiment numbers - try multiple patterns
            sentiment_match = re.search(r'"sentiment":\s*([+-]?\d+)', text)
            if sentiment_match:
                result['sentiment'] = int(sentiment_match.group(1))
            else:
                # Fallback: look for standalone sentiment values
                for sentiment in ['-2', '-1', '+1', '+2', '0']:
                    if sentiment in text:
                        result['sentiment'] = int(sentiment.replace('+', ''))
                        break
            
            # Look for confidence - try multiple patterns
            conf_match = re.search(r'"confidence":\s*(\d+)', text)
            if conf_match:
                result['confidence'] = int(conf_match.group(1))
            else:
                # Try without quotes
                conf_match = re.search(r'confidence[：:]\s*(\d+)', text)
                if conf_match:
                    result['confidence'] = int(conf_match.group(1))
            
            # Look for bucket - try multiple patterns (Chinese only)
            bucket_match = re.search(r'"bucket":\s*"([^"]+)"', text)
            if bucket_match:
                result['bucket'] = bucket_match.group(1)
            else:
                # Fallback: look for Chinese bucket keywords anywhere in text
                bucket_patterns = ['日常实践', '责任义务', '家庭冲突', '理论探讨', '婚恋择偶']
                for bucket in bucket_patterns:
                    if bucket in text:
                        result['bucket'] = bucket
                        break
            
            # Look for reasoning - try various patterns
            reasoning_patterns = [
                r'"reasoning":\s*"([^"]+)"',  # Standard JSON format
                r'"reasoning":\s*"([^"]*)',   # Incomplete JSON 
                r'reasoning[：:]\s*([^，。！？\n]+)',  # Chinese colon format
                r'分析[：:]?\s*([^，。！？\n]+)',  # Analysis prefix
                r'因为([^，。！？\n]+)',  # Because format
                r'该微博([^，。！？\n]+)',  # This weibo format
            ]
            
            for pattern in reasoning_patterns:
                match = re.search(pattern, text)
                if match:
                    result['reasoning'] = match.group(1).strip()
                    break
            
            # Return if we found sentiment or bucket (for pre-filtered relevant posts)
            if 'sentiment' in result or 'bucket' in result or 'relevant' in result:
                return result
        except:
            pass
        
        return None

    def extract_json(self, response_text):
        """Extract JSON from Qwen response using robust method with validation"""
        result = self.extract_json_robust(response_text)
        
        # Convert "none", "None", "null" to empty string for bucket
        if result and 'bucket' in result:
            if result['bucket'] and str(result['bucket']).lower() in ['none', 'null']:
                result['bucket'] = ''
        
        # If sentiment is 0, set bucket to empty string and return
        if result and 'sentiment' in result and result['sentiment'] == 0:
            result['bucket'] = ''
            return result
        
        # For non-zero sentiments, validate bucket is one of allowed values
        if result and 'sentiment' in result and result['sentiment'] != 0:
            if 'bucket' not in result or not result['bucket']:
                # Non-zero sentiment needs a valid bucket - reject
                return None
            
            valid_buckets = ['日常实践', '责任义务', '家庭冲突', '理论探讨', '婚恋择偶']
            if result['bucket'] not in valid_buckets:
                # Invalid bucket - reject this result
                return None
        
        return result
    
    def process_single_post(self, text, retry=False):
        """Process a single post with Qwen - with optional retry for 100% success rate"""
        if not text or len(str(text).strip()) == 0:
            return {"error": "empty_text"}
        
        # Clean text and limit length for better processing
        clean_text = str(text).replace('"', '\\"')[:500]
        full_prompt = f"{SYSTEM_PROMPT}\n\n微博文本：{clean_text}\n\nJSON:"
        
        # Use faster 3b model for 2x speedup; longer timeout on retry
        timeout_duration = 60 if retry else 30
        
        try:
            result = subprocess.run(
                ['ollama', 'run', 'qwen2.5:7b'],  
                input=full_prompt,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=timeout_duration  # 30s normal, 60s on retry
            )
            
            if result.returncode != 0:
                # Retry once if ollama failed and not already retrying
                if not retry:
                    time.sleep(2)
                    return self.process_single_post(text, retry=True)
                return {"error": "ollama_failed"}
                
            if not result.stdout.strip():
                if not retry:
                    time.sleep(2)
                    return self.process_single_post(text, retry=True)
                return {"error": "empty_response"}
            
            response_text = result.stdout.strip()
            json_data = self.extract_json(response_text)
            if json_data:
                return json_data
            else:
                # Retry once for JSON extraction failures
                if not retry:
                    time.sleep(2)
                    return self.process_single_post(text, retry=True)
                # Save failed response for debugging
                with open("results/failed_extraction_debug.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n{'='*60}\nFailed extraction:\n")
                    f.write(f"Text: {clean_text[:100]}...\n")
                    f.write(f"Qwen output: {response_text}\n")
                return {"error": "json_extraction_failed"}
                
        except subprocess.TimeoutExpired:
            # Retry once with longer timeout
            if not retry:
                print(f"    Timeout, retrying with 60s timeout...")
                time.sleep(2)
                return self.process_single_post(text, retry=True)
            return {"error": "timeout"}
        except Exception as e:
            if not retry:
                time.sleep(2)
                return self.process_single_post(text, retry=True)
            return {"error": f"exception: {str(e)[:50]}"}
    
    def show_status(self):
        """Show current progress status"""
        total_posts = len(self.df)
        processed = self.progress["total_processed"]
        remaining = total_posts - processed
        
        print("\\n" + "="*60)
        print("CURRENT ANALYSIS STATUS")
        print("="*60)
        print(f"Total posts in sample: {total_posts:,}")
        print(f"Posts processed: {processed:,} ({100*processed/total_posts:.1f}%)")
        print(f"Successful analyses: {self.progress['successful']:,}")
        print(f"Errors: {self.progress['errors']:,}")
        print(f"Remaining: {remaining:,}")
        
        if processed > 0:
            success_rate = 100 * self.progress['successful'] / processed
            print(f"Success rate: {success_rate:.1f}%")
        
        if remaining > 0:
            est_minutes = (remaining * 1.5) / 60
            print(f"Estimated time remaining: {est_minutes:.0f} minutes ({est_minutes/60:.1f} hours)")
        
        # Show next batch info
        next_idx = self.progress["last_processed_idx"] + 1
        if next_idx < total_posts:
            print(f"\nNext to process: Index {next_idx}")
            print(f"Preview: {str(self.df.iloc[next_idx].get('text', ''))[:80]}...")
        else:
            print(f"\nAnalysis complete!")
        
        return processed, remaining
    
    def run_batch(self, batch_size=50, max_batches=None):
        """Run batch analysis with frequent saves and resume capability"""
        
        # Show initial status
        processed, remaining = self.show_status()
        
        if remaining == 0:
            print("\nAnalysis already complete!")
            return self.df
        
        # Start analysis
        if self.progress["start_time"] is None:
            self.progress["start_time"] = datetime.now().isoformat()
        
        session_start = datetime.now()
        start_idx = self.progress["last_processed_idx"] + 1
        total_posts = len(self.df)
        
        print(f"\nStarting batch analysis...")
        print(f"   Batch size: {batch_size}")
        print(f"   Starting from index: {start_idx}")
        
        if max_batches:
            print(f"   Max batches: {max_batches}")
        
        print(f"   You can stop anytime with Ctrl+C!")
        
        session_stats = {"batches": 0, "processed": 0, "successful": 0, "errors": 0}
        
        try:
            batch_num = 0
            current_idx = start_idx
            
            while current_idx < total_posts:
                # Check batch limit
                if max_batches and batch_num >= max_batches:
                    print(f"\nReached max batches limit ({max_batches})")
                    break
                
                batch_end = min(current_idx + batch_size, total_posts)
                batch_num += 1
                
                print(f"\\n{'='*50}")
                print(f"BATCH {batch_num}: Posts {current_idx}-{batch_end-1}")
                print(f"{'='*50}")
                
                batch_start = time.time()
                batch_successful = 0
                batch_errors = 0
                
                # Process posts in this batch
                for i in tqdm(range(current_idx, batch_end), desc=f"Batch {batch_num}"):
                    # Skip already processed
                    if pd.notna(self.df.at[i, 'qwen_processed_at']):
                        continue
                    
                    row = self.df.iloc[i]
                    text = row.get('text', '') or row.get('cleaned_text', '')
                    
                    # Analyze with Qwen
                    result = self.process_single_post(text)
                    
                    # Store results
                    if 'error' in result:
                        self.df.at[i, 'qwen_error'] = result['error']
                        self.progress["errors"] = int(self.progress["errors"] + 1)
                        session_stats["errors"] += 1
                        batch_errors += 1
                    else:
                        # All posts are already relevant, just store sentiment analysis
                        self.df.at[i, 'qwen_sentiment'] = result.get('sentiment', 0)
                        self.df.at[i, 'qwen_bucket'] = result.get('bucket', 'None')
                        self.df.at[i, 'qwen_confidence'] = result.get('confidence', 50)
                        self.df.at[i, 'qwen_reasoning'] = ''  # No reasoning for speed
                        
                        self.progress["successful"] = int(self.progress["successful"] + 1)
                        session_stats["successful"] += 1
                        batch_successful += 1
                    
                    self.df.at[i, 'qwen_processed_at'] = datetime.now().isoformat()
                    self.progress["total_processed"] = int(self.progress["total_processed"] + 1)
                    session_stats["processed"] += 1
                
                # Save after each batch
                batch_time = time.time() - batch_start
                self.save_progress(int(batch_end - 1))  # Convert to regular Python int
                self.save_results()
                
                # Batch summary
                batch_rate = batch_size / batch_time if batch_time > 0 else 0
                session_time = (datetime.now() - session_start).total_seconds()
                
                print(f"\nBatch {batch_num} completed in {batch_time:.0f}s")
                print(f"   Rate: {batch_rate:.2f} posts/sec")
                print(f"   Success: {batch_successful}/{batch_size}")
                print(f"   Errors: {batch_errors}")
                
                # Overall progress
                overall_progress = 100 * self.progress["total_processed"] / total_posts
                print(f"\nOverall: {self.progress['total_processed']}/{total_posts} ({overall_progress:.1f}%)")
                print(f"   Session time: {session_time/60:.0f} minutes")
                print(f"   Success rate: {100 * self.progress['successful'] / max(self.progress['total_processed'], 1):.1f}%")
                
                # ETA
                remaining_posts = total_posts - batch_end
                if batch_rate > 0 and remaining_posts > 0:
                    eta_minutes = remaining_posts / batch_rate / 60
                    print(f"   ETA: {eta_minutes:.0f} minutes")
                
                session_stats["batches"] += 1
                current_idx = batch_end
        
        except KeyboardInterrupt:
            print(f"\nWARNING: Analysis interrupted by user!")
            self.save_progress(int(current_idx - 1))  # Convert to regular Python int
            self.save_results()
            print(f"Progress saved. Resume by running the script again.")
        
        # Session summary
        session_time = (datetime.now() - session_start).total_seconds()
        self.progress["sessions"].append({
            "start_time": session_start.isoformat(),
            "duration_minutes": session_time / 60,
            "batches": int(session_stats["batches"]),
            "processed": int(session_stats["processed"]),
            "successful": int(session_stats["successful"]),
            "errors": int(session_stats["errors"])
        })
        
        print(f"\nSession completed!")
        print(f"   Duration: {session_time/60:.0f} minutes")
        print(f"   Batches: {session_stats['batches']}")
        print(f"   Posts processed: {session_stats['processed']}")
        print(f"   Success rate: {100 * session_stats['successful'] / max(session_stats['processed'], 1):.1f}%")
        
        return self.df
    
    def save_results(self):
        """Save current results to CSV"""
        output_columns = [
            'post_id', 'time', 'year', 'month', 'text', 'text_length',
            'qwen_sentiment', 'qwen_bucket', 'qwen_confidence', 'qwen_reasoning',
            'qwen_error', 'qwen_processed_at'
        ]
        
        available_columns = [col for col in output_columns if col in self.df.columns]
        os.makedirs("results", exist_ok=True)
        self.df[available_columns].to_csv(self.results_file, index=False, encoding='utf-8')
    
    def reset_analysis(self):
        """Reset all analysis progress and results - start from scratch"""
        print("\n RESET ANALYSIS")
        print("This will delete all progress and results, starting from scratch.")
        confirm = input("Are you sure? Type 'yes' to confirm: ").strip().lower()
        
        if confirm != 'yes':
            print("Reset cancelled.")
            return False
        
        files_to_remove = [
            self.progress_file,
            self.results_file,
            "results/failed_extraction_debug.txt"
        ]
        
        removed_count = 0
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"✓ Removed: {file_path}")
                    removed_count += 1
                except Exception as e:
                    print(f"✗ Failed to remove {file_path}: {e}")
            else:
                print(f"• Not found: {file_path}")
        
        print(f"\n✅ Reset complete! Removed {removed_count} file(s).")
        print("Please restart the script to begin fresh analysis.")
        return True
    
    def create_truly_relevant_dataset(self):
        """Show summary stats - all posts are already relevant"""
        analyzed = self.df[self.df['qwen_processed_at'].notna()]
        
        print(f"Analysis Summary:")
        print(f"   Total posts: {len(self.df):,}")
        print(f"   Analyzed: {len(analyzed):,}")
        print(f"   (All posts are pre-filtered as relevant)")
        
        if len(analyzed) == 0:
            print("No posts have been analyzed yet!")
            return
        
        # Show sentiment distribution
        print(f"\n💯 Sentiment Distribution:")
        sentiment_counts = analyzed['qwen_sentiment'].value_counts().sort_index()
        for sentiment, count in sentiment_counts.items():
            print(f"   {sentiment:+2d}: {count:,} posts")
        
        # Show bucket distribution
        if 'qwen_bucket' in analyzed.columns:
            bucket_counts = analyzed['qwen_bucket'].value_counts()
            print(f"\n📦 Context Buckets:")
            for bucket, count in bucket_counts.items():
                print(f"   {bucket}: {count:,} posts")


def main():
    """Interactive main function"""
    
    input_file = "data/weibo_xiao_relevant_only.csv"
    
    if not os.path.exists(input_file):
        print(f"ERROR: Input file '{input_file}' not found!")
        print("TIP: Run 'python filter_relevant_posts.py' first to filter relevant posts")
        return
    
    print("ANALYSIS SETUP:")
    print("Working with pre-filtered relevant posts only")
    print(f"Input file: {input_file}")
    
    # Initialize analyzer (data already filtered for relevance)
    analyzer = BatchWeiboAnalyzer(input_file)
    
    while True:
        processed, remaining = analyzer.show_status()
        
        if remaining == 0:
            print("\nAnalysis complete! Exiting.")
            break
        
        print(f"\nBATCH ANALYSIS OPTIONS:")
        print(f"1. Test run (1 batch of 10 posts)")
        print(f"2. Quick run (5 batches of 50 posts)")
        print(f"3. Medium run (20 batches of 50 posts)")
        print(f"4. Custom run (specify batches & size)")
        print(f"5. Full run (process all remaining)")
        print(f"6. Create truly relevant dataset (filter out irrelevant)")
        print(f"7. Just show status")
        print(f"8. Reset analysis (delete all progress & results)")
        print(f"0. Exit")
        
        choice = input(f"\nSelect option (0-8): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            analyzer.run_batch(batch_size=10, max_batches=1)
        elif choice == "2":
            analyzer.run_batch(batch_size=50, max_batches=5)
        elif choice == "3":
            analyzer.run_batch(batch_size=50, max_batches=20)
        elif choice == "4":
            try:
                batch_size = int(input("Batch size (default 50): ") or "50")
                max_batches = input("Max batches (Enter for unlimited): ").strip()
                max_batches = int(max_batches) if max_batches else None
                analyzer.run_batch(batch_size=batch_size, max_batches=max_batches)
            except ValueError:
                print("ERROR: Invalid input!")
        elif choice == "5":
            analyzer.run_batch(batch_size=100, max_batches=None)
        elif choice == "6":
            analyzer.create_truly_relevant_dataset()
        elif choice == "7":
            continue  # Status already shown
        elif choice == "8":
            if analyzer.reset_analysis():
                print("\nExiting... Please restart to begin fresh analysis.")
                break
        else:
            print("ERROR: Invalid option!")
        
        # Ask if user wants to continue
        if remaining > 0:
            cont = input(f"\\nContinue with more batches? (y/n): ").strip().lower()
            if cont not in ['y', 'yes']:
                break


if __name__ == "__main__":
    main()