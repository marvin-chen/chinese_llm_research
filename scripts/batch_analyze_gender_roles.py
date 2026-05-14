#!/usr/bin/env python3
"""
Batch Qwen Analysis for 女主内 / 男主外
Processes the two relevant-only CSVs and assigns sentiment scores plus content buckets.
"""

from __future__ import annotations

import argparse
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


SYSTEM_PROMPT = """你是中文社交媒体内容分析专家，判断微博对“男主外女主内”这类传统性别角色分工观念的态度。

评分标准：
• -2 (强烈负面): 对该观念本身进行严厉批评、否定，或将其与女性压迫、性别歧视等社会问题直接挂钩
• -1 (轻度负面): 表达不认同、无奈、疲惫、反感或保留意见，但不是最强烈的批判
• 0 (中性): 主要是客观描述、引用、玩笑、段子、文学化表达，或没有明确态度
• +1 (轻度正面): 表达一般性认同、接受、向往或支持，但不算非常强烈
• +2 (强烈正面): 明确强烈支持、赞美、维护该观念，或用具体经历/细节强化支持

关键区别：
+1 vs +2: 一般性支持 vs 强烈、具体、明确的支持
-1 vs -2: 保留/无奈/反感 vs 严厉批评、直接否定

内容桶必须从以下 6 个选项中选 1 个，所有帖子都要选，不要留空：
• 社会批判: 对“男主外女主内”观念本身进行严厉批评、否定，或将其与社会问题（如女性压迫、性别歧视）直接挂钩
• 个人立场/体验: 表达个人对该观念的认同、向往、排斥或亲身经历的感受（包括无奈、疲惫、幸福等）
• 家庭/婚姻关系: 讨论该观念在具体家庭中的应用以及带来的影响，如夫妻和谐、父亲缺位、育儿分工，或作为择偶/婚姻标准
• 文化/历史/社会分析: 将该观念作为客观存在的文化现象、历史背景或社会议题进行分析、引用或报道，不表达个人好恶
• 性别平等/角色讨论: 围绕该观念展开关于性别平等、现代性别角色分工的探讨、辩论或反思
• 其他: 幽默段子、文学创作等难以归入以上类别的特殊内容

请严格只返回 JSON，不要输出多余解释：
{"sentiment": -2到2之间的整数, "bucket": "社会批判或个人立场/体验或家庭/婚姻关系或文化/历史/社会分析或性别平等/角色讨论或其他", "confidence": 0到100之间的整数}
"""


class GenderRoleAnalyzer:
    def __init__(self, input_file: Path, output_prefix: str, model: str = "qwen2.5:7b"):
        self.input_file = Path(input_file)
        self.output_prefix = output_prefix
        self.model = model

        self.progress_file = RESULTS_DIR / f"{output_prefix}_progress.json"
        self.results_file = RESULTS_DIR / f"{output_prefix}_results.csv"

        print(f"Loading dataset: {self.input_file}")
        self.df = pd.read_csv(self.input_file)
        print(f"Loaded {len(self.df)} posts")
        print(f"Model: {model}")

        column_types = {
            "qwen_sentiment": "Int64",
            "qwen_bucket": "object",
            "qwen_confidence": "Int64",
            "qwen_reasoning": "object",
            "qwen_error": "object",
            "qwen_processed_at": "object",
        }
        for col, dtype in column_types.items():
            if col not in self.df.columns:
                if dtype == "Int64":
                    self.df[col] = pd.Series(dtype=dtype)
                else:
                    self.df[col] = ""
            else:
                # Ensure correct dtype for existing columns
                if dtype == "object" and self.df[col].dtype != "object":
                    self.df[col] = self.df[col].astype("object")

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

    def normalize_bucket(self, value):
        if value is None:
            return None
        normalized = str(value).strip()
        bucket_map = {
            "社会批判": "社会批判",
            "个人立场/体验": "个人立场/体验",
            "家庭/婚姻关系": "家庭/婚姻关系",
            "文化/历史/社会分析": "文化/历史/社会分析",
            "性别平等/角色讨论": "性别平等/角色讨论",
            "其他": "其他",
            "其它": "其他",
        }
        return bucket_map.get(normalized, normalized if normalized in bucket_map.values() else None)

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

    def validate_result(self, data):
        if not isinstance(data, dict):
            return None

        if "sentiment" not in data or "bucket" not in data:
            return None

        try:
            sentiment = data["sentiment"]
            if isinstance(sentiment, str):
                sentiment = int(sentiment.replace("+", "").strip())
            else:
                sentiment = int(sentiment)
        except Exception:
            return None

        if sentiment not in [-2, -1, 0, 1, 2]:
            return None

        bucket = self.normalize_bucket(data.get("bucket"))
        if bucket is None:
            return None

        confidence = data.get("confidence", 50)
        try:
            confidence = int(confidence)
        except Exception:
            confidence = 50
        confidence = max(0, min(100, confidence))

        result = {
            "sentiment": sentiment,
            "bucket": bucket,
            "confidence": confidence,
        }
        if "reasoning" in data and data["reasoning"] is not None:
            result["reasoning"] = str(data["reasoning"]).strip()
        return result

    def process_single_post(self, text, retry=False, idx: int = -1):
        if not text or len(str(text).strip()) == 0:
            return {"error": "empty_text"}

        clean_text = str(text).replace('"', '\\"')[:500]
        full_prompt = f"{SYSTEM_PROMPT}\n\n微博文本：{clean_text}\n\nJSON:"
        timeout_duration = 75 if retry else 45
        
        if idx >= 0:
            print(f"  [{idx}] Calling Ollama...", flush=True)

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
                    time.sleep(2)
                    return self.process_single_post(text, retry=True, idx=idx)
                return {"error": "ollama_failed"}

            if not result.stdout.strip():
                if not retry:
                    time.sleep(2)
                    return self.process_single_post(text, retry=True, idx=idx)
                return {"error": "empty_response"}

            response_text = result.stdout.strip()
            json_data = self.extract_json(response_text)
            validated = self.validate_result(json_data) if json_data else None

            if validated:
                return validated

            if not retry:
                time.sleep(2)
                return self.process_single_post(text, retry=True, idx=idx)

            with open(RESULTS_DIR / "gender_role_failed_extraction_debug.txt", "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\nFailed extraction [Post #{idx}]:\n")
                f.write(f"Text: {clean_text[:120]}...\n")
                f.write(f"Qwen output: {response_text}\n")
            return {"error": "json_extraction_failed"}

        except subprocess.TimeoutExpired:
            if not retry:
                print(f"    ⏱ Timeout, retrying with longer timeout...")
                time.sleep(2)
                return self.process_single_post(text, retry=True, idx=idx)
            return {"error": "timeout"}
        except Exception as e:
            if not retry:
                time.sleep(2)
                return self.process_single_post(text, retry=True, idx=idx)
            return {"error": f"exception: {str(e)[:50]}"}

    def get_post_text(self, row):
        for column in ["cleaned_text", "text", "full_text"]:
            value = row.get(column, "")
            if pd.notna(value) and str(value).strip():
                return str(value)
        return ""

    def show_status(self, total_posts):
        processed = self.progress["total_processed"]
        remaining = max(total_posts - processed, 0)

        print("\n" + "=" * 60)
        print("CURRENT ANALYSIS STATUS")
        print("=" * 60)
        print(f"Total posts in sample: {total_posts:,}")
        print(f"Posts processed: {processed:,} ({100 * processed / max(total_posts, 1):.1f}%)")
        print(f"Successful analyses: {self.progress['successful']:,}")
        print(f"Errors: {self.progress['errors']:,}")
        print(f"Remaining: {remaining:,}")

        if processed > 0:
            success_rate = 100 * self.progress["successful"] / processed
            print(f"Success rate: {success_rate:.1f}%")

        if remaining > 0:
            est_minutes = (remaining * 0.8) / 60
            print(f"Estimated time remaining: {est_minutes:.0f} minutes ({est_minutes/60:.1f} hours)")

        next_idx = self.progress["last_processed_idx"] + 1
        if next_idx < total_posts:
            preview_text = self.get_post_text(self.df.iloc[next_idx])
            print(f"\nNext to process: Index {next_idx}")
            print(f"Preview: {preview_text[:80]}...")
        else:
            print("\nAnalysis complete!")

        return processed, remaining

    def run_batch(self, max_posts=None, batch_size=10):
        total_posts = len(self.df)
        start_idx = self.progress["last_processed_idx"] + 1
        
        # If max_posts is set, it's the limit for THIS session, not overall
        # Only limit if we haven't finished yet
        if max_posts is not None and start_idx < total_posts:
            # Process up to max_posts MORE posts in this session
            batch_end_idx = min(start_idx + max_posts, total_posts)
        else:
            batch_end_idx = total_posts
        
        processed_so_far = self.progress["total_processed"]
        remaining = batch_end_idx - start_idx
        
        print("\n" + "=" * 60)
        print("CURRENT ANALYSIS STATUS")
        print("=" * 60)
        print(f"Total posts in file: {total_posts:,}")
        print(f"Already processed: {processed_so_far:,}")
        print(f"Successful: {self.progress['successful']:,}, Errors: {self.progress['errors']:,}")
        if remaining > 0:
            print(f"Remaining to process: {remaining:,}")
            success_rate = 100 * self.progress["successful"] / max(processed_so_far, 1) if processed_so_far > 0 else 0
            print(f"Success rate so far: {success_rate:.1f}%")
            est_minutes = (remaining * 0.8) / 60
            print(f"Estimated time: {est_minutes:.0f} min ({est_minutes/60:.1f} hrs) @ 0.8s/post")
        else:
            print(f"\n✓ All posts in file already processed!")

        if remaining == 0:
            return self.df

        if self.progress["start_time"] is None:
            self.progress["start_time"] = datetime.now().isoformat()

        print("\nStarting batch analysis...")
        print(f"   Starting from post #{start_idx + 1}")
        if max_posts is not None:
            print(f"   This session limit: {max_posts} posts")
        print(f"   Batch size: {batch_size} posts")
        print("   You can stop anytime with Ctrl+C!")

        session_stats = {"batches": 0, "processed": 0, "successful": 0, "errors": 0}
        session_start = datetime.now()
        total_posts_in_file = len(self.df)

        try:
            current_idx = start_idx
            while current_idx < batch_end_idx:
                batch_start_idx = current_idx
                batch_end_idx_current = min(batch_start_idx + batch_size, batch_end_idx)
                batch_total = batch_end_idx_current - batch_start_idx
                batch_num = session_stats["batches"] + 1
                batch_successful = 0
                batch_errors = 0

                print(f"\n{'=' * 50}")
                print(f"BATCH {batch_num}: Posts {batch_start_idx + 1}-{batch_end_idx_current}")
                print(f"{'=' * 50}")

                for post_idx in range(batch_start_idx, batch_end_idx_current):
                    row = self.df.iloc[post_idx]
                    text = self.get_post_text(row)
                    preview = text[:60].replace("\n", " ")
                    print(f"\nPost #{post_idx + 1}/{total_posts_in_file}: {preview}...")

                    result = self.process_single_post(text, idx=post_idx + 1)

                    if "error" in result:
                        self.df.at[post_idx, "qwen_error"] = result["error"]
                        self.progress["errors"] = int(self.progress["errors"] + 1)
                        session_stats["errors"] += 1
                        batch_errors += 1
                        print(f"    ERROR: {result['error']}")
                    else:
                        self.df.at[post_idx, "qwen_sentiment"] = int(result.get("sentiment", 0))
                        self.df.at[post_idx, "qwen_bucket"] = str(result.get("bucket", ""))
                        self.df.at[post_idx, "qwen_confidence"] = int(result.get("confidence", 50))
                        reasoning = result.get("reasoning", "")
                        self.df.at[post_idx, "qwen_reasoning"] = str(reasoning) if reasoning else ""
                        self.progress["successful"] = int(self.progress["successful"] + 1)
                        session_stats["successful"] += 1
                        batch_successful += 1
                        print(f"    ✓ Sentiment: {result.get('sentiment'):+d}, Bucket: {result.get('bucket')}, Conf: {result.get('confidence', 50)}%")

                    self.df.at[post_idx, "qwen_processed_at"] = datetime.now().isoformat()
                    self.progress["total_processed"] = int(self.progress["total_processed"] + 1)
                    session_stats["processed"] += 1

                    self.save_progress(post_idx)

                self.save_results()

                session_stats["batches"] += 1

                batch_rate = 100 * batch_successful / max(batch_total, 1)
                overall_rate = 100 * self.progress["successful"] / max(self.progress["total_processed"], 1)
                posts_left = batch_end_idx - batch_end_idx_current
                if posts_left > 0:
                    time_left_min = (posts_left * 0.8) / 60
                    eta_text = f"ETA: ~{time_left_min:.0f}min"
                else:
                    eta_text = "ETA: done"

                print(f"\nBatch {batch_num} completed")
                print(f"   Batch success rate: {batch_rate:.1f}% ({batch_successful}/{batch_total})")
                print(f"   Batch errors: {batch_errors}")
                print(f"   Overall processed: {self.progress['total_processed']}/{total_posts} ({100 * self.progress['total_processed'] / max(total_posts, 1):.1f}%)")
                print(f"   Overall success rate: {overall_rate:.1f}%")
                print(f"   Remaining in session: {posts_left}")
                print(f"   {eta_text}")

                current_idx = batch_end_idx_current

        except KeyboardInterrupt:
            print("\nWARNING: Analysis interrupted by user!")
            self.save_progress(int(current_idx - 1))
            self.save_results()
            print("Progress saved. Resume by running the script again.")

        session_time = (datetime.now() - session_start).total_seconds()
        self.progress["sessions"].append({
            "start_time": session_start.isoformat(),
            "duration_minutes": session_time / 60,
            "processed": int(session_stats["processed"]),
            "successful": int(session_stats["successful"]),
            "errors": int(session_stats["errors"]),
        })

        print("\nSession Summary:")
        print(f"   Duration: {session_time/60:.1f} minutes")
        print(f"   Posts processed this session: {session_stats['processed']}")
        print(f"   Successful: {session_stats['successful']}, Errors: {session_stats['errors']}")
        if session_stats['processed'] > 0:
            rate = 100 * session_stats['successful'] / session_stats['processed']
            print(f"   Session success rate: {rate:.1f}%")
        
        print(f"\n   Total progress: {self.progress['total_processed']}/{total_posts_in_file} posts")
        print(f"   Overall success rate: {100 * self.progress['successful'] / max(self.progress['total_processed'], 1):.1f}%")
        if batch_end_idx < total_posts_in_file:
            print(f"   Still {total_posts_in_file - batch_end_idx:,} posts remaining")
        else:
            print(f"   ✓ All posts in this file completed!")

        return self.df

    def save_results(self):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.results_file, index=False, encoding="utf-8")


def build_configs():
    return [
        {
            "name": "nvzhunei",
            "input_file": DATA_DIR / "女主内_relevant_only.csv",
            "output_prefix": "nvzhunei_qwen_analysis",
        },
        {
            "name": "nanzhuwai",
            "input_file": DATA_DIR / "男主外_relevant_only.csv",
            "output_prefix": "nanzhuwai_qwen_analysis",
        },
    ]


def process_all(model: str = "qwen2.5:7b", limit: int | None = None, batch_size: int = 10):
    print("=" * 70)
    print("GENDER ROLE SENTIMENT ANALYSIS")
    print("=" * 70)
    print(f"Model: {model}")
    print(f"Limit: {limit if limit is not None else 'all posts'}")
    print("Files: 女主内_relevant_only.csv, 男主外_relevant_only.csv")
    print("Sequential processing (Ollama is the bottleneck)\n")

    configs = build_configs()
    successful = 0

    for config in configs:
        print(f"\n{'=' * 70}")
        print(f"Processing: {config['name']}")
        print(f"Input: {config['input_file']}")
        print(f"Output prefix: {config['output_prefix']}")
        print(f"{'=' * 70}")

        analyzer = GenderRoleAnalyzer(
            input_file=config["input_file"],
            output_prefix=config["output_prefix"],
            model=model,
        )

        analyzer.run_batch(max_posts=limit, batch_size=batch_size)
        successful += 1

    print("\n" + "=" * 70)
    print("ALL PROCESSING COMPLETE")
    print(f"Processed {successful}/{len(configs)} files successfully")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen sentiment + bucket analysis for 女主内 / 男主外 datasets.")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama model name")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N posts per file")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of posts to treat as one reporting batch")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_all(model=args.model, limit=args.limit, batch_size=args.batch_size)
