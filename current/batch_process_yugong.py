"""
Batch Buckets + Sentiment Analyzer (愚公移山)

Matches the structure and resume-capable flow used in `batch_analyze.py`.
Processes a pre-filtered relevant CSV and produces results with resume, progress
file, and a results CSV in `results/`.
"""

import pandas as pd
import json
import subprocess
import time
import os
from datetime import datetime
from tqdm import tqdm


SYSTEM_PROMPT = """你是中文文本分类与情感评估专家。输入为已筛选出与“愚公移山”相关的微博文本。

请返回JSON仅含以下字段：{"sentiment": int, "bucket": str, "confidence": int}

情感分数定义：-2 强烈负面, -1 负面, 0 中性, 1 积极, 2 强烈积极

上下文类别（若 sentiment==0 则 bucket 设为空字符串）：
• 政治语境
• 学习成长
• 批评讽刺
• 专名指称
• 一般使用
• 建设实践

返回示例：{"sentiment": 1, "bucket": "学习成长", "confidence": 85}
"""


class BatchBucketSentimentAnalyzer:
    def __init__(self, input_file, output_prefix="愚公_buckets"):
        self.input_file = input_file
        self.output_prefix = output_prefix
        self.progress_file = f"results/{output_prefix}_progress.json"
        self.results_file = f"results/{output_prefix}_results.csv"

        print(f"Loading dataset: {input_file}")
        self.df = pd.read_csv(input_file)
        print(f"Loaded {len(self.df)} posts (pre-filtered for relevance)")

        # Initialize result columns
        result_column_types = {
            'qwen_sentiment': 'Int64',
            'qwen_bucket': 'object',
            'qwen_confidence': 'Int64',
            'qwen_error': 'object',
            'qwen_processed_at': 'object'
        }
        for col, dtype in result_column_types.items():
            if col not in self.df.columns:
                self.df[col] = pd.Series(dtype=dtype)

        self.progress = self.load_progress()
        self.load_existing_results()

    def load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                print("WARNING: corrupted progress file — starting fresh")
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
        self.progress.update({
            "last_processed_idx": int(idx),
            "last_update": datetime.now().isoformat()
        })
        os.makedirs("results", exist_ok=True)
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)

    def load_existing_results(self):
        if os.path.exists(self.results_file):
            try:
                print(f"Found existing results file: {self.results_file}")
                existing_df = pd.read_csv(self.results_file)
                processed_count = existing_df['qwen_processed_at'].notna().sum()
                successful_count = existing_df['qwen_sentiment'].notna().sum()
                error_count = existing_df['qwen_error'].notna().sum()

                self.progress.update({
                    "total_processed": int(processed_count),
                    "successful": int(successful_count),
                    "errors": int(error_count)
                })

                if 'post_id' in existing_df.columns and 'post_id' in self.df.columns:
                    merge_cols = ['post_id'] + [c for c in existing_df.columns if c.startswith('qwen_')]
                    self.df = self.df.drop(columns=[c for c in merge_cols[1:] if c in self.df.columns])
                    self.df = self.df.merge(existing_df[merge_cols], on='post_id', how='left')

                if processed_count > 0:
                    mask = self.df['qwen_processed_at'].notna()
                    if mask.any():
                        self.progress['last_processed_idx'] = int(self.df[mask].index.max())

                print(f"   Previously processed: {processed_count}, successful: {successful_count}, errors: {error_count}")
            except Exception as e:
                print(f"WARNING: Could not load existing results: {e}")

    def extract_json_robust(self, text):
        if not text:
            return None
        import re
        # Try to find JSON object
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except:
                pass

        # Fallback patterns
        try:
            res = {}
            # sentiment
            s = re.search(r'"sentiment"\s*[:：]\s*([+-]?\d+)', text)
            if s:
                res['sentiment'] = int(s.group(1))
            # bucket
            b = None
            for bucket in ["政治语境","学习成长","批评讽刺","专名指称","一般使用","建设实践"]:
                if bucket in text:
                    b = bucket
                    break
            if b:
                res['bucket'] = b
            # confidence
            c = re.search(r'"confidence"\s*[:：]\s*(\d+)', text)
            if c:
                res['confidence'] = int(c.group(1))

            if res:
                return res
        except:
            pass

        return None

    def extract_json(self, response_text):
        result = self.extract_json_robust(response_text)
        if not result:
            return None

        # Normalize bucket for sentiment==0
        if 'sentiment' in result and result['sentiment'] == 0:
            result['bucket'] = ''

        # Validate sentiment range
        if 'sentiment' in result:
            try:
                s = int(result['sentiment'])
                if s < -2 or s > 2:
                    return None
            except:
                return None

        # If sentiment !=0 ensure bucket present
        if 'sentiment' in result and result['sentiment'] != 0:
            if 'bucket' not in result or not result['bucket']:
                return None

        return result

    def process_single_post(self, text, retry=False):
        if not text or len(str(text).strip()) == 0:
            return {'error': 'empty_text'}

        clean_text = str(text).replace('"', '\\"')[:500]
        full_prompt = SYSTEM_PROMPT + f"\n\n微博文本：{clean_text}\n\nJSON:"

        timeout_duration = 60 if retry else 30
        try:
            result = subprocess.run(
                ['ollama', 'run', 'qwen2.5:7b'],
                input=full_prompt,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=timeout_duration
            )

            if result.returncode != 0:
                if not retry:
                    time.sleep(2)
                    return self.process_single_post(text, retry=True)
                return {'error': 'ollama_failed'}

            if not result.stdout.strip():
                if not retry:
                    time.sleep(2)
                    return self.process_single_post(text, retry=True)
                return {'error': 'empty_response'}

            response_text = result.stdout.strip()
            json_data = self.extract_json(response_text)
            if json_data:
                return json_data
            else:
                if not retry:
                    time.sleep(2)
                    return self.process_single_post(text, retry=True)
                with open('results/failed_extraction_debug.txt','a',encoding='utf-8') as f:
                    f.write(f"\n{'='*60}\nText: {clean_text[:100]}...\nResponse: {response_text}\n")
                return {'error': 'json_extraction_failed'}

        except subprocess.TimeoutExpired:
            if not retry:
                time.sleep(2)
                return self.process_single_post(text, retry=True)
            return {'error': 'timeout'}
        except Exception as e:
            if not retry:
                time.sleep(2)
                return self.process_single_post(text, retry=True)
            return {'error': f'exception: {str(e)[:80]}'}

    def show_status(self):
        total_posts = len(self.df)
        processed = self.progress['total_processed']
        remaining = total_posts - processed

        print('\n' + '='*60)
        print('CURRENT BUCKETS+SENTIMENT STATUS')
        print('='*60)
        print(f'Total posts: {total_posts:,}')
        print(f'Processed: {processed:,} ({100*processed/total_posts:.1f}%)')
        print(f"Successful: {self.progress['successful']:,}")
        print(f"Errors: {self.progress['errors']:,}")

        next_idx = self.progress['last_processed_idx'] + 1
        if next_idx < total_posts:
            print(f"\nNext to process: Index {next_idx}")
            print(f"Preview: {str(self.df.iloc[next_idx].get('text',''))[:80]}...")
        else:
            print('\nAll posts processed')

        return processed, remaining

    def run_batch(self, batch_size=50, max_batches=None):
        processed, remaining = self.show_status()
        if remaining == 0:
            print('\nAlready complete')
            return self.df

        if self.progress['start_time'] is None:
            self.progress['start_time'] = datetime.now().isoformat()

        session_start = datetime.now()
        start_idx = self.progress['last_processed_idx'] + 1
        total_posts = len(self.df)

        print(f"Starting batch run from {start_idx}, batch_size={batch_size}")

        session_stats = {'batches':0,'processed':0,'successful':0,'errors':0}

        try:
            batch_num = 0
            current_idx = start_idx
            while current_idx < total_posts:
                if max_batches and batch_num >= max_batches:
                    print(f"Reached max_batches={max_batches}")
                    break

                batch_end = min(current_idx + batch_size, total_posts)
                batch_num += 1

                print(f"\n{'='*50}\nBATCH {batch_num}: {current_idx}-{batch_end-1}\n{'='*50}")
                batch_start = time.time()
                batch_success = 0
                batch_errors = 0

                for i in tqdm(range(current_idx, batch_end), desc=f"Batch {batch_num}"):
                    if pd.notna(self.df.at[i,'qwen_processed_at']):
                        continue

                    row = self.df.iloc[i]
                    text = row.get('text','') or row.get('cleaned_text','')
                    res = self.process_single_post(text)

                    if 'error' in res:
                        self.df.at[i,'qwen_error'] = res['error']
                        self.progress['errors'] = int(self.progress['errors'] + 1)
                        session_stats['errors'] += 1
                        batch_errors += 1
                    else:
                        self.df.at[i,'qwen_sentiment'] = res.get('sentiment',0)
                        self.df.at[i,'qwen_bucket'] = res.get('bucket','')
                        self.df.at[i,'qwen_confidence'] = res.get('confidence',50)
                        self.progress['successful'] = int(self.progress['successful'] + 1)
                        session_stats['successful'] += 1
                        batch_success += 1

                    self.df.at[i,'qwen_processed_at'] = datetime.now().isoformat()
                    self.progress['total_processed'] = int(self.progress['total_processed'] + 1)
                    session_stats['processed'] += 1

                batch_time = time.time() - batch_start
                self.save_progress(int(batch_end-1))
                self.save_results()

                print(f"\nBatch {batch_num} completed in {batch_time:.0f}s")
                print(f"  Success: {batch_success}/{batch_size}, Errors: {batch_errors}")

                current_idx = batch_end
                session_stats['batches'] += 1

        except KeyboardInterrupt:
            print('\nInterrupted by user — saving progress')
            self.save_progress(int(current_idx-1))
            self.save_results()
            return self.df

        session_time = (datetime.now() - session_start).total_seconds()
        self.progress['sessions'].append({
            'start_time': session_start.isoformat(),
            'duration_minutes': session_time/60,
            'batches': int(session_stats['batches']),
            'processed': int(session_stats['processed']),
            'successful': int(session_stats['successful']),
            'errors': int(session_stats['errors'])
        })

        print(f"\nSession completed: processed {session_stats['processed']} posts in {session_time/60:.1f} minutes")
        return self.df

    def save_results(self):
        output_columns = [
            'post_id','time','year','month','text','text_length',
            'qwen_sentiment','qwen_bucket','qwen_confidence','qwen_error','qwen_processed_at'
        ]
        available = [c for c in output_columns if c in self.df.columns]
        os.makedirs('results', exist_ok=True)
        self.df[available].to_csv(self.results_file, index=False, encoding='utf-8')

    def reset_analysis(self):
        print('\nRESET analysis (will delete progress & results)')
        confirm = input("Type 'yes' to confirm: ").strip().lower()
        if confirm != 'yes':
            print('Cancelled')
            return False
        for p in [self.progress_file, self.results_file, 'results/failed_extraction_debug.txt']:
            if os.path.exists(p):
                try:
                    os.remove(p)
                    print(f"Removed: {p}")
                except Exception as e:
                    print(f"Failed to remove {p}: {e}")
            else:
                print(f"Not found: {p}")
        return True

    def create_truly_relevant_dataset(self):
        analyzed = self.df[self.df['qwen_processed_at'].notna()]
        print(f"Analyzed: {len(analyzed)} / {len(self.df)}")
        if len(analyzed) == 0:
            print('No analyzed posts yet')
            return
        print('\nSentiment distribution:')
        print(analyzed['qwen_sentiment'].value_counts().sort_index())
        print('\nBucket distribution:')
        if 'qwen_bucket' in analyzed.columns:
            print(analyzed['qwen_bucket'].value_counts())


def main():
    input_file = 'data/愚公移山_relevant_only.csv'
    if not os.path.exists(input_file):
        print(f"ERROR: Input '{input_file}' not found")
        return

    analyzer = BatchBucketSentimentAnalyzer(input_file)

    while True:
        processed, remaining = analyzer.show_status()
        if remaining == 0:
            print('\nAll done')
            break

        print('\nOptions:')
        print('1. Test run (1 batch of 10)')
        print('2. Quick run (5 batches of 50)')
        print('3. Full run')
        print('4. Reset analysis')
        print('0. Exit')
        choice = input('Select: ').strip()
        if choice == '0':
            break
        elif choice == '1':
            analyzer.run_batch(batch_size=10, max_batches=1)
        elif choice == '2':
            analyzer.run_batch(batch_size=50, max_batches=5)
        elif choice == '3':
            analyzer.run_batch(batch_size=100, max_batches=None)
        elif choice == '4':
            if analyzer.reset_analysis():
                break
        else:
            print('Invalid option')


if __name__ == '__main__':
    main()
