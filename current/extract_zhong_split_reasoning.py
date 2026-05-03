#!/usr/bin/env python3
"""Extract Ollama reasoning for zhong-split analysis samples.

This script reads zhong_split_analysis_results.csv, creates two filtered
subsets, and asks Ollama to explain the sentiment and bucket assignment for
each row:
1. A random sample of 100 rows with sentiment -2.
2. All rows with sentiment -1 and label 不忠.

Each output CSV includes the original sentiment, label, bucket, and the
generated llm_reasoning column.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_FILE = PROJECT_ROOT / "results" / "zhong_split_analysis_results.csv"
DEFAULT_NEGATIVE_TWO_OUTPUT = PROJECT_ROOT / "results" / "zhong_split_reasoning_sentiment_minus2_sample100.csv"
DEFAULT_BU_ZHONG_OUTPUT = PROJECT_ROOT / "results" / "zhong_split_reasoning_sentiment_minus1_bu_zhong.csv"
DEFAULT_NEGATIVE_TWO_PROGRESS = PROJECT_ROOT / "results" / "zhong_split_reasoning_sentiment_minus2_progress.json"
DEFAULT_BU_ZHONG_PROGRESS = PROJECT_ROOT / "results" / "zhong_split_reasoning_sentiment_minus1_bu_zhong_progress.json"

REASONING_PROMPT_TEMPLATE = """你是一个中文社交媒体内容分析专家。请分析以下微博帖子关于“忠/不忠”的情感和语境分类。

帖子内容：
{text}

这个帖子之前被分类为：
- 情感分数: {sentiment} (范围从-2到+2，-2=强烈负面，-1=轻微负面，0=中立/无关，+1=轻微正面，+2=强烈正面)
- 标签: {label}
- 语境类别: {bucket}

请详细解释：
1. 为什么这个帖子被给予 {sentiment} 的情感分数？具体是帖子中的哪些词语、短语或表达体现了这种情感？
2. 为什么这个帖子被归类到 "{bucket}" 类别？帖子的哪些内容符合这个类别的特征？
3. 这个分类是否合理？如果不合理，应该如何调整？

请用清晰、详细的中文回答，引用帖子中的具体内容来支持你的解释。

回答："""

BUCKET_MAPPING = {
    "人伦关系": "人伦关系",
    "抽象概念": "抽象概念",
    "群体组织": "群体组织",
    "其它对象": "其它对象",
    "其他对象": "其它对象",
}


def get_bucket_chinese(bucket_value: str) -> str:
    return BUCKET_MAPPING.get(bucket_value, bucket_value)


def resolve_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "qwen_sentiment" in frame.columns:
        frame["qwen_sentiment"] = pd.to_numeric(frame["qwen_sentiment"], errors="coerce")
    if "qwen_target" in frame.columns:
        frame["qwen_target"] = frame["qwen_target"].fillna("").astype(str).str.strip()
    if "qwen_bucket" in frame.columns:
        frame["qwen_bucket"] = (
            frame["qwen_bucket"].fillna("").astype(str).str.strip().replace({"None": "", "nan": ""})
        )
    if "qwen_reasoning" in frame.columns:
        frame["qwen_reasoning"] = frame["qwen_reasoning"].fillna("").astype(str).str.strip()
    return frame


def build_export_frame(
    df: pd.DataFrame,
    reasoning_column: str,
    label_column: str,
    sample_type: str,
) -> pd.DataFrame:
    columns = [
        column
        for column in ["post_id", "text", reasoning_column, "qwen_sentiment", label_column, "qwen_bucket"]
        if column in df.columns
    ]
    export_df = df[columns].copy()
    export_df = export_df.rename(
        columns={
            reasoning_column: "llm_reasoning",
            "qwen_sentiment": "sentiment",
            label_column: "label",
            "qwen_bucket": "bucket",
        }
    )
    export_df.insert(0, "sample_type", sample_type)
    ordered_columns = [
        column
        for column in ["sample_type", "post_id", "text", "llm_reasoning", "sentiment", "label", "bucket"]
        if column in export_df.columns
    ]
    return export_df[ordered_columns]


def extract_reasoning_from_llm(
    text: str,
    sentiment: int | float,
    label: str,
    bucket: str,
    model: str = "qwen2.5:7b",
    timeout: int = 60,
) -> tuple[str | None, str | None]:
    prompt = REASONING_PROMPT_TEMPLATE.format(
        text=text,
        sentiment=sentiment,
        label=label,
        bucket=get_bucket_chinese(bucket),
    )

    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip(), None
        error_message = result.stderr.strip() if result.stderr else "Unknown error"
        return None, f"Ollama error: {error_message}"
    except subprocess.TimeoutExpired:
        return None, f"Timeout after {timeout}s"
    except Exception as exc:
        return None, f"Error: {exc}"


def load_progress(progress_file: Path) -> set[int]:
    if not progress_file.exists():
        return set()
    try:
        with open(progress_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return set(payload.get("processed_ids", []))
    except Exception:
        return set()


def save_progress(progress_file: Path, processed_ids: set[int], success_count: int, error_count: int) -> None:
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "processed_ids": list(processed_ids),
        "last_updated": datetime.now().isoformat(),
        "success_count": success_count,
        "error_count": error_count,
    }
    with open(progress_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def process_with_reasoning(
    df: pd.DataFrame,
    output_file: Path,
    progress_file: Path,
    model: str,
    timeout: int,
) -> pd.DataFrame:
    frame = df.copy()
    if "sentiment" not in frame.columns:
        if "qwen_sentiment" in frame.columns:
            frame["sentiment"] = pd.to_numeric(frame["qwen_sentiment"], errors="coerce")
        else:
            raise ValueError("Could not find a sentiment column in the source file.")
    else:
        frame["sentiment"] = pd.to_numeric(frame["sentiment"], errors="coerce")

    if "label" not in frame.columns:
        if "qwen_target" in frame.columns:
            frame["label"] = frame["qwen_target"].fillna("").astype(str).str.strip()
        elif "target" in frame.columns:
            frame["label"] = frame["target"].fillna("").astype(str).str.strip()
        else:
            raise ValueError("Could not find a label column in the source file.")
    else:
        frame["label"] = frame["label"].fillna("").astype(str).str.strip()

    if "bucket" not in frame.columns:
        if "qwen_bucket" in frame.columns:
            frame["bucket"] = frame["qwen_bucket"].fillna("").astype(str).str.strip().replace({"None": "", "nan": ""})
        else:
            raise ValueError("Could not find a bucket column in the source file.")
    else:
        frame["bucket"] = frame["bucket"].fillna("").astype(str).str.strip().replace({"None": "", "nan": ""})

    if "llm_reasoning" not in frame.columns:
        frame["llm_reasoning"] = ""
    if "reasoning_error" not in frame.columns:
        frame["reasoning_error"] = ""
    if "reasoning_extracted_at" not in frame.columns:
        frame["reasoning_extracted_at"] = ""

    processed_ids = load_progress(progress_file)
    success_count = 0
    error_count = 0

    id_column = "post_id" if "post_id" in frame.columns else None
    if id_column is None:
        raise ValueError("Could not find post_id in the source file.")

    print(f"\nExtracting reasoning with model: {model}")
    print(f"Timeout per post: {timeout}s")
    print(f"Already processed: {len(processed_ids):,}")

    for idx, row in tqdm(frame.iterrows(), total=len(frame), desc=f"Processing {output_file.name}"):
        post_id = row[id_column]
        if post_id in processed_ids:
            continue

        reasoning, error = extract_reasoning_from_llm(
            text=row.get("text", ""),
            sentiment=row.get("sentiment", ""),
            label=row.get("label", ""),
            bucket=row.get("bucket", ""),
            model=model,
            timeout=timeout,
        )

        if reasoning:
            frame.at[idx, "llm_reasoning"] = reasoning
            frame.at[idx, "reasoning_extracted_at"] = datetime.now().isoformat()
            success_count += 1
        else:
            frame.at[idx, "reasoning_error"] = error or "Unknown error"
            error_count += 1

        processed_ids.add(post_id)

        if len(processed_ids) % 10 == 0:
            frame.to_csv(output_file, index=False, encoding="utf-8")
            save_progress(progress_file, processed_ids, success_count, error_count)

    frame.to_csv(output_file, index=False, encoding="utf-8")
    save_progress(progress_file, processed_ids, success_count, error_count)

    print(f"Saved: {output_file}")
    print(f"Success: {success_count:,}, Errors: {error_count:,}")
    return frame


def build_filtered_samples(
    source_file: Path = DEFAULT_SOURCE_FILE,
    negative_two_output: Path = DEFAULT_NEGATIVE_TWO_OUTPUT,
    bu_zhong_output: Path = DEFAULT_BU_ZHONG_OUTPUT,
    sample_size: int = 100,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not source_file.exists():
        raise FileNotFoundError(f"Input file not found: {source_file}")

    df = normalize_frame(pd.read_csv(source_file))

    label_column = resolve_column(df, ("qwen_target", "label", "target"))

    if label_column is None:
        raise ValueError("Could not find a label column. Expected qwen_target, label, or target.")
    if "qwen_sentiment" not in df.columns:
        raise ValueError("Could not find qwen_sentiment in the source file.")

    df[label_column] = df[label_column].fillna("").astype(str).str.strip()

    negative_two = df[df["qwen_sentiment"] == -2].copy()
    if len(negative_two) > sample_size:
        negative_two = negative_two.sample(n=sample_size, random_state=random_state).sort_index()

    bu_zhong_minus_one = df[
        (df["qwen_sentiment"] == -1) & (df[label_column] == "不忠")
    ].copy()

    return negative_two, bu_zhong_minus_one


def export_reasoning_samples(
    source_file: Path = DEFAULT_SOURCE_FILE,
    negative_two_output: Path = DEFAULT_NEGATIVE_TWO_OUTPUT,
    bu_zhong_output: Path = DEFAULT_BU_ZHONG_OUTPUT,
    sample_size: int = 100,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    negative_two, bu_zhong_minus_one = build_filtered_samples(
        source_file=source_file,
        negative_two_output=negative_two_output,
        bu_zhong_output=bu_zhong_output,
        sample_size=sample_size,
        random_state=random_state,
    )

    reasoning_column = resolve_column(pd.read_csv(source_file), ("qwen_reasoning", "llm_reasoning", "reasoning"))
    label_column = resolve_column(pd.read_csv(source_file), ("qwen_target", "label", "target"))
    if reasoning_column is None or label_column is None:
        raise ValueError("Could not resolve reasoning or label columns from source file.")

    negative_two_export = build_export_frame(negative_two, reasoning_column, label_column, "sentiment_-2_sample")
    bu_zhong_export = build_export_frame(bu_zhong_minus_one, reasoning_column, label_column, "sentiment_-1_bu_zhong")
    negative_two_output.parent.mkdir(parents=True, exist_ok=True)
    bu_zhong_output.parent.mkdir(parents=True, exist_ok=True)
    negative_two_export.to_csv(negative_two_output, index=False, encoding="utf-8")
    bu_zhong_export.to_csv(bu_zhong_output, index=False, encoding="utf-8")
    return negative_two_export, bu_zhong_export


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Ollama reasoning for zhong-split analysis samples.")
    parser.add_argument(
        "--source-file",
        type=Path,
        default=DEFAULT_SOURCE_FILE,
        help="Path to zhong_split_analysis_results.csv",
    )
    parser.add_argument(
        "--negative-two-output",
        type=Path,
        default=DEFAULT_NEGATIVE_TWO_OUTPUT,
        help="Output CSV for the random -2 sample",
    )
    parser.add_argument(
        "--bu-zhong-output",
        type=Path,
        default=DEFAULT_BU_ZHONG_OUTPUT,
        help="Output CSV for the -1 and 不忠 sample",
    )
    parser.add_argument("--sample-size", type=int, default=100, help="Sample size for sentiment -2 rows")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--model", type=str, default="qwen2.5:7b", help="Ollama model to use")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout per post in seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    negative_two_subset, bu_zhong_subset = build_filtered_samples(
        source_file=args.source_file,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )

    negative_two_export = process_with_reasoning(
        negative_two_subset,
        args.negative_two_output,
        args.negative_two_output.with_name(args.negative_two_output.stem + "_progress.json"),
        args.model,
        args.timeout,
    )
    bu_zhong_export = process_with_reasoning(
        bu_zhong_subset,
        args.bu_zhong_output,
        args.bu_zhong_output.with_name(args.bu_zhong_output.stem + "_progress.json"),
        args.model,
        args.timeout,
    )

    print(f"Loaded: {args.source_file}")
    print(f"Saved -2 sample: {len(negative_two_export):,} rows -> {args.negative_two_output}")
    print(f"Saved -1 不忠 sample: {len(bu_zhong_export):,} rows -> {args.bu_zhong_output}")
    print("Columns exported: llm_reasoning, sentiment, label, bucket")


if __name__ == "__main__":
    main()
