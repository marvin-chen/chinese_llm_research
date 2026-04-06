"""
Batch Analysis for 忠 / 不忠
Processes data/relevant_only/忠_relevant_only.csv

Workflow:
1. Identify whether the post is mainly about 忠 or 不忠 with a simple first-pass matcher.
2. Apply the sentiment rubric to the detected target in a smaller second prompt.
3. Multiply the raw sentiment by -1 when the target is 不忠.
4. Classify topic bucket and part of speech.

Reasoning is deferred to a later pass.
"""

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


SYSTEM_PROMPT = """你是中国文化专家，评估微博对“忠”或“不忠”的态度。

评分标准：
-2（负面）
忠：批评愚忠；认为忠是不好的品质
不忠：认为不忠是品行问题；强烈批评一个人“不忠”（提及具体行为，或是语气非常负面）；批评容忍不忠的人

-1（轻度负面）
忠：作为动词时，表忠心的对象是负面的
不忠：只是简单地用不忠形容某个人/事/物，以此作为批评

0（中性）
不相信忠的存在；单纯讨论“忠”或“不忠”的概念，不给予任何评价；引用古人的话

1（轻度正面）
忠：只是简单地用忠形容某个人事物，以此作为赞扬；作为动词时，表忠心的对象是正面的（不特别强调）
不忠：指出某个人/事/物并不应该贴上“不忠”这个标签

2（正面）
忠：特别强调一个人的“忠”（提及具体行为或是情感强烈）；作为形容词时与其它高尚品德并列；作为动词时，表忠心的对象是非常正面的；表扬忠这个品德本身，认为忠是大家应该学习的品质
不忠：批评骂别人不忠的人；不认为不忠是不好的品质；提出不忠的好处

四个 bucket 只选一个：
1. 人伦关系：自己、伴侣、偶像、君主、朋友、家庭
2. 抽象概念：感情、价值观、职责、梦想
3. 群体组织：国家、政党、球队、公司
4. 其它对象：品牌、节目、原著

词性只选一个：
a = 形容词
n = 名词
v = 动词

仅返回 JSON：
{"sentiment_raw": -2到2之间的整数, "bucket": "人伦关系或抽象概念或群体组织或其它对象", "part_of_speech": "a或n或v", "confidence": 0到100之间的整数}
"""


class ZhongSplitAnalyzer:
    def __init__(self, input_file=DATA_DIR / "relevant_only/忠_relevant_only.csv", output_prefix="zhong_split_analysis",
                 model="qwen2.5:7b"):
        self.input_file = input_file
        self.output_prefix = output_prefix
        self.model = model

        self.progress_file = RESULTS_DIR / f"{output_prefix}_progress.json"
        self.results_file = RESULTS_DIR / f"{output_prefix}_results.csv"
        self.zhong_file = RESULTS_DIR / f"{output_prefix}_zhong.csv"
        self.bu_zhong_file = RESULTS_DIR / f"{output_prefix}_bu_zhong.csv"

        print(f"Loading dataset: {input_file}")
        self.df = pd.read_csv(input_file)
        print(f"Loaded {len(self.df)} posts")
        print(f"Model: {model}")

        # Initialize columns
        column_types = {
            "qwen_target": "object",
            "qwen_sentiment_raw": "Int64",
            "qwen_sentiment": "Int64",
            "qwen_bucket": "object",
            "qwen_pos": "object",
            "qwen_confidence": "Int64",
            "qwen_reasoning": "object",
            "qwen_error": "object",
            "qwen_processed_at": "object",
        }
        for col, dtype in column_types.items():
            if col not in self.df.columns:
                self.df[col] = pd.Series(dtype=dtype)

        self.progress = self.load_progress()
        self.load_existing_results()

    def load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                try:
                    os.remove(self.progress_file)
                except Exception:
                    pass
        return {
            "last_processed_idx": -1,
            "total_processed": 0,
            "successful": 0,
            "errors": 0,
            "start_time": None,
            "sessions": [],
        }

    def save_progress(self, idx):
        self.progress.update({
            "last_processed_idx": int(idx),
            "last_update": datetime.now().isoformat(),
        })
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)

    def load_existing_results(self):
        if not os.path.exists(self.results_file):
            return

        try:
            print(f"Found existing results file: {self.results_file}")
            existing_df = pd.read_csv(self.results_file)
            processed_count = existing_df["qwen_processed_at"].notna().sum() if "qwen_processed_at" in existing_df.columns else 0
            successful_count = existing_df["qwen_sentiment"].notna().sum() if "qwen_sentiment" in existing_df.columns else 0
            error_count = existing_df["qwen_error"].notna().sum() if "qwen_error" in existing_df.columns else 0

            print(f"   Previously processed: {processed_count} posts")
            print(f"   Successful: {successful_count}, Errors: {error_count}")

            if "post_id" in existing_df.columns and "post_id" in self.df.columns:
                merge_cols = ["post_id"] + [col for col in existing_df.columns if col.startswith("qwen_")]
                existing_subset = existing_df[merge_cols].copy()
                drop_cols = [col for col in merge_cols[1:] if col in self.df.columns]
                if drop_cols:
                    self.df = self.df.drop(columns=drop_cols)
                self.df = self.df.merge(existing_subset, on="post_id", how="left")

                self.progress.update({
                    "total_processed": int(processed_count),
                    "successful": int(successful_count),
                    "errors": int(error_count),
                })

                last_processed_mask = self.df["qwen_processed_at"].notna()
                if last_processed_mask.any():
                    self.progress["last_processed_idx"] = int(self.df[last_processed_mask].index.max())

                print("Merged existing results")
        except Exception as e:
            print(f"WARNING: Could not load existing results: {str(e)}")

    def normalize_target(self, value):
        if value is None:
            return None
        value = str(value).strip()
        if value in ["忠", "zhong", "Zhong"]:
            return "忠"
        if value in ["不忠", "bu忠", "bu zhong", "buzhong", "不忠于", "disloyal"]:
            return "不忠"
        return None

    def detect_target(self, text):
        """Deterministic first-pass target detection."""
        if text is None:
            return None

        text = str(text)

        bu_zhong_patterns = [r"不忠于", r"不忠", r"不忠心", r"不忠诚"]
        for pattern in bu_zhong_patterns:
            if re.search(pattern, text):
                return "不忠"

        if re.search(r"忠", text):
            return "忠"

        return None

    def normalize_bucket(self, value):
        if value is None:
            return None
        value = str(value).strip()
        bucket_map = {
            "人伦关系": "人伦关系",
            "抽象概念": "抽象概念",
            "群体组织": "群体组织",
            "其它对象": "其它对象",
            "其他对象": "其它对象",
        }
        return bucket_map.get(value, value if value in bucket_map.values() else None)

    def normalize_pos(self, value):
        if value is None:
            return None
        value = str(value).strip().lower()
        pos_map = {
            "a": "a",
            "adj": "a",
            "形容词": "a",
            "n": "n",
            "noun": "n",
            "名词": "n",
            "v": "v",
            "verb": "v",
            "动词": "v",
        }
        return pos_map.get(value, None)

    def extract_json(self, response_text):
        if not response_text:
            return None

        cleaned = response_text.strip()
        if not cleaned:
            return None

        patterns = [
            r"```json\s*(\{.*?\})\s*```",
            r"```\s*(\{.*?\})\s*```",
            r"(\{.*\})",
        ]

        for pattern in patterns:
            match = re.search(pattern, cleaned, re.DOTALL)
            if not match:
                continue
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict):
                    return data
            except Exception:
                continue

        return None

    def analyze_target(self, text):
        """First pass: detect whether the post is about 忠 or 不忠."""
        target = self.detect_target(text)
        if target:
            return target

        # Fallback for edge cases where a simple match is not enough.
        clean_text = str(text).replace('"', '\\"')[:200]
        prompt = f"你只需要判断这条微博主要是在谈『忠』还是『不忠』。\n只返回 JSON：{{\"target\": \"忠或不忠\"}}\n\n微博文本：{clean_text}\n\nJSON:"

        try:
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=20,
            )
            if result.returncode != 0:
                return None

            data = self.extract_json(result.stdout.strip())
            if not data:
                return None
            return self.normalize_target(data.get("target"))
        except Exception:
            return None

    def process_single_post(self, text, retry=False):
        if not text or len(str(text).strip()) == 0:
            return {"error": "empty_text"}

        target = self.analyze_target(text)
        if target is None:
            return {"error": "target_detection_failed"}

        clean_text = str(text).replace('"', '\\"')[:600]
        full_prompt = f"{SYSTEM_PROMPT}\n\n当前 target：{target}\n\n微博文本：{clean_text}\n\nJSON:"
        timeout_duration = 60 if retry else 35

        try:
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=full_prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout_duration,
            )

            if result.returncode != 0:
                if not retry:
                    time.sleep(1)
                    return self.process_single_post(text, retry=True)
                return {"error": "ollama_failed"}

            response_text = result.stdout.strip()
            if not response_text:
                if not retry:
                    time.sleep(1)
                    return self.process_single_post(text, retry=True)
                return {"error": "empty_response"}

            data = self.extract_json(response_text)
            if not data:
                if not retry:
                    time.sleep(1)
                    return self.process_single_post(text, retry=True)
                with open(RESULTS_DIR / "failed_extraction_debug.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n{'=' * 60}\nFailed extraction:\n")
                    f.write(f"Text: {clean_text[:150]}...\n")
                    f.write(f"Qwen output: {response_text}\n")
                return {"error": "json_extraction_failed"}

            sentiment_raw = data.get("sentiment_raw", data.get("sentiment"))
            bucket = self.normalize_bucket(data.get("bucket"))
            pos = self.normalize_pos(data.get("part_of_speech", data.get("pos")))
            confidence = data.get("confidence")

            try:
                sentiment_raw = int(sentiment_raw)
            except Exception:
                return {"error": "invalid_sentiment_raw"}

            if sentiment_raw < -2 or sentiment_raw > 2:
                return {"error": "sentiment_out_of_range"}

            if bucket is None:
                return {"error": "invalid_bucket"}

            if pos is None:
                return {"error": "invalid_pos"}

            try:
                confidence = int(confidence)
            except Exception:
                confidence = 50

            if confidence < 0:
                confidence = 0
            if confidence > 100:
                confidence = 100

            final_sentiment = sentiment_raw if target == "忠" else -1 * sentiment_raw

            return {
                "target": target,
                "sentiment_raw": sentiment_raw,
                "sentiment": final_sentiment,
                "bucket": bucket,
                "part_of_speech": pos,
                "confidence": confidence,
                "reasoning": "",
            }

        except subprocess.TimeoutExpired:
            if not retry:
                time.sleep(1)
                return self.process_single_post(text, retry=True)
            return {"error": "timeout"}
        except Exception as e:
            if not retry:
                time.sleep(1)
                return self.process_single_post(text, retry=True)
            return {"error": f"exception: {str(e)[:50]}"}

    def show_status(self):
        total_posts = len(self.df)
        processed = self.progress["total_processed"]
        remaining = total_posts - processed

        print("\n" + "=" * 60)
        print("ZHONG / BU ZHONG ANALYSIS STATUS")
        print("=" * 60)
        print(f"Total posts in sample: {total_posts:,}")
        print(f"Posts processed: {processed:,} ({100 * processed / total_posts:.1f}%)")
        print(f"Successful analyses: {self.progress['successful']:,}")
        print(f"Errors: {self.progress['errors']:,}")
        print(f"Remaining: {remaining:,}")

        if processed > 0:
            success_rate = 100 * self.progress["successful"] / processed
            print(f"Success rate: {success_rate:.1f}%")

        if remaining > 0:
            est_minutes = (remaining * 1.0) / 60
            print(f"Estimated time remaining: {est_minutes:.0f} minutes ({est_minutes / 60:.1f} hours)")

        next_idx = self.progress["last_processed_idx"] + 1
        if next_idx < total_posts:
            preview_text = str(self.df.iloc[next_idx].get("text", ""))[:80]
            print(f"\nNext to process: Index {next_idx}")
            print(f"Preview: {preview_text}...")
        else:
            print("\nAnalysis complete!")

        return processed, remaining

    def run_batch(self, batch_size=50, max_batches=None):
        processed, remaining = self.show_status()
        if remaining == 0:
            print("\nAnalysis already complete!")
            return self.df

        if self.progress["start_time"] is None:
            self.progress["start_time"] = datetime.now().isoformat()

        session_start = datetime.now()
        start_idx = self.progress["last_processed_idx"] + 1
        total_posts = len(self.df)

        print("\nStarting batch analysis...")
        print(f"   Batch size: {batch_size}")
        print(f"   Starting from index: {start_idx}")
        if max_batches:
            print(f"   Max batches: {max_batches}")
        print("   You can stop anytime with Ctrl+C!")

        session_stats = {"batches": 0, "processed": 0, "successful": 0, "errors": 0}

        try:
            batch_num = 0
            current_idx = start_idx

            while current_idx < total_posts:
                if max_batches and batch_num >= max_batches:
                    print(f"\nReached max batches limit ({max_batches})")
                    break

                batch_end = min(current_idx + batch_size, total_posts)
                batch_num += 1

                print(f"\n{'=' * 50}")
                print(f"BATCH {batch_num}: Posts {current_idx}-{batch_end - 1}")
                print(f"{'=' * 50}")

                batch_start_time = time.time()
                batch_successful = 0
                batch_errors = 0

                for i in tqdm(range(current_idx, batch_end), desc=f"Batch {batch_num}"):
                    if pd.notna(self.df.at[i, "qwen_processed_at"]):
                        continue

                    row = self.df.iloc[i]
                    text = row.get("text", "") or row.get("cleaned_text", "")
                    result = self.process_single_post(text)

                    if "error" in result:
                        self.df.at[i, "qwen_error"] = result["error"]
                        self.progress["errors"] = int(self.progress["errors"] + 1)
                        session_stats["errors"] += 1
                        batch_errors += 1
                    else:
                        self.df.at[i, "qwen_target"] = result.get("target")
                        self.df.at[i, "qwen_sentiment_raw"] = result.get("sentiment_raw")
                        self.df.at[i, "qwen_sentiment"] = result.get("sentiment")
                        self.df.at[i, "qwen_bucket"] = result.get("bucket")
                        self.df.at[i, "qwen_pos"] = result.get("part_of_speech")
                        self.df.at[i, "qwen_confidence"] = result.get("confidence", 50)
                        self.df.at[i, "qwen_reasoning"] = result.get("reasoning", "")

                        self.progress["successful"] = int(self.progress["successful"] + 1)
                        session_stats["successful"] += 1
                        batch_successful += 1

                    self.df.at[i, "qwen_processed_at"] = datetime.now().isoformat()
                    self.progress["total_processed"] = int(self.progress["total_processed"] + 1)
                    session_stats["processed"] += 1

                batch_time = time.time() - batch_start_time
                self.save_progress(int(batch_end - 1))
                self.save_results()

                batch_rate = batch_size / batch_time if batch_time > 0 else 0
                session_time = (datetime.now() - session_start).total_seconds()

                print(f"\nBatch {batch_num} completed in {batch_time:.0f}s")
                print(f"   Rate: {batch_rate:.2f} posts/sec")
                print(f"   Success: {batch_successful}/{batch_size}")
                print(f"   Errors: {batch_errors}")

                overall_progress = 100 * self.progress["total_processed"] / total_posts
                print(f"\nOverall: {self.progress['total_processed']}/{total_posts} ({overall_progress:.1f}%)")
                print(f"   Session time: {session_time / 60:.0f} minutes")
                print(f"   Success rate: {100 * self.progress['successful'] / max(self.progress['total_processed'], 1):.1f}%")

                remaining_posts = total_posts - batch_end
                if batch_rate > 0 and remaining_posts > 0:
                    eta_minutes = remaining_posts / batch_rate / 60
                    print(f"   ETA: {eta_minutes:.0f} minutes")

                session_stats["batches"] += 1
                current_idx = batch_end

        except KeyboardInterrupt:
            print("\nWARNING: Analysis interrupted by user!")
            self.save_progress(int(current_idx - 1))
            self.save_results()
            print("Progress saved. Resume by running the script again.")

        session_time = (datetime.now() - session_start).total_seconds()
        self.progress["sessions"].append({
            "start_time": session_start.isoformat(),
            "duration_minutes": session_time / 60,
            "batches": int(session_stats["batches"]),
            "processed": int(session_stats["processed"]),
            "successful": int(session_stats["successful"]),
            "errors": int(session_stats["errors"]),
        })

        print("\nSession completed!")
        print(f"   Duration: {session_time / 60:.0f} minutes")
        print(f"   Batches: {session_stats['batches']}")
        print(f"   Posts processed: {session_stats['processed']}")
        print(f"   Success rate: {100 * session_stats['successful'] / max(session_stats['processed'], 1):.1f}%")

        return self.df

    def save_results(self):
        output_columns = [
            "post_id",
            "time",
            "year",
            "month",
            "text",
            "user_id",
            "text_length",
            "is_relevant",
            "processed_at",
            "qwen_target",
            "qwen_sentiment_raw",
            "qwen_sentiment",
            "qwen_bucket",
            "qwen_pos",
            "qwen_confidence",
            "qwen_reasoning",
            "qwen_error",
            "qwen_processed_at",
        ]

        available_columns = [col for col in output_columns if col in self.df.columns]
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.df[available_columns].to_csv(self.results_file, index=False, encoding="utf-8")

        analyzed = self.df[self.df["qwen_processed_at"].notna()].copy()
        if len(analyzed) > 0:
            zhong_df = analyzed[analyzed["qwen_target"] == "忠"].copy()
            bu_zhong_df = analyzed[analyzed["qwen_target"] == "不忠"].copy()
            zhong_df.to_csv(self.zhong_file, index=False, encoding="utf-8")
            bu_zhong_df.to_csv(self.bu_zhong_file, index=False, encoding="utf-8")

    def summarize_results(self):
        analyzed = self.df[self.df["qwen_processed_at"].notna()].copy()
        if len(analyzed) == 0:
            print("No posts have been analyzed yet!")
            return

        print("\nANALYSIS SUMMARY")
        print(f"   Total posts: {len(self.df):,}")
        print(f"   Analyzed: {len(analyzed):,}")
        print(f"   忠: {(analyzed['qwen_target'] == '忠').sum():,}")
        print(f"   不忠: {(analyzed['qwen_target'] == '不忠').sum():,}")

        print("\nFINAL SENTIMENT DISTRIBUTION")
        sentiment_counts = analyzed["qwen_sentiment"].value_counts().sort_index()
        for sentiment, count in sentiment_counts.items():
            print(f"   {int(sentiment):+2d}: {count:,} posts")

        print("\nBUCKET DISTRIBUTION")
        bucket_counts = analyzed["qwen_bucket"].value_counts()
        for bucket, count in bucket_counts.items():
            print(f"   {bucket}: {count:,} posts")

        print("\nPART OF SPEECH DISTRIBUTION")
        pos_counts = analyzed["qwen_pos"].value_counts()
        for pos, count in pos_counts.items():
            label = {"a": "形容词", "n": "名词", "v": "动词"}.get(pos, pos)
            print(f"   {pos} ({label}): {count:,} posts")


def main():
    input_file = DATA_DIR / "relevant_only/忠_relevant_only.csv"

    if not os.path.exists(input_file):
        print(f"ERROR: Input file '{input_file}' not found!")
        print("TIP: Make sure the relevant-only folder exists and contains 忠_relevant_only.csv")
        return

    print("ANALYSIS SETUP")
    print("Working with 忠_relevant_only.csv")
    print(f"Input file: {input_file}")

    analyzer = ZhongSplitAnalyzer(input_file=input_file)

    while True:
        processed, remaining = analyzer.show_status()

        if remaining == 0:
            print("\nAnalysis complete! Exiting.")
            analyzer.summarize_results()
            break

        print("\nBATCH ANALYSIS OPTIONS:")
        print("1. Test run (1 batch of 10 posts)")
        print("2. Quick run (5 batches of 50 posts)")
        print("3. Medium run (20 batches of 50 posts)")
        print("4. Custom run (specify batches & size)")
        print("5. Full run (process all remaining)")
        print("6. Just show status")
        print("7. Reset analysis (delete all progress & results)")
        print("8. Show summary")
        print("0. Exit")

        choice = input("\nSelect option (0-8): ").strip()

        if choice == "0":
            print("Goodbye!")
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
            continue
        elif choice == "7":
            confirm = input("Type 'yes' to confirm reset: ").strip().lower()
            if confirm == "yes":
                files_to_remove = [
                    analyzer.progress_file,
                    analyzer.results_file,
                    analyzer.zhong_file,
                    analyzer.bu_zhong_file,
                    RESULTS_DIR / "failed_extraction_debug.txt",
                ]
                for file_path in files_to_remove:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                print("Reset complete.")
                analyzer = ZhongSplitAnalyzer(input_file=input_file)
            else:
                print("Reset cancelled.")
        elif choice == "8":
            analyzer.summarize_results()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
