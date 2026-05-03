#!/usr/bin/env python3
"""
Visualize zhong_split_analysis_results.csv with Chinese-renderable Plotly charts.

Outputs:
1. Sentiment -2 distribution by bucket.
2. Bucket distribution by year.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_FILE = BASE_DIR / "results" / "zhong_split_analysis_results.csv"
OUTPUT_DIR = BASE_DIR / "results"

BUCKET_ORDER = ["人伦关系", "其它对象", "抽象概念", "群体组织"]
BUCKET_COLORS = {
    "人伦关系": "#1f77b4",
    "其它对象": "#ff7f0e",
    "抽象概念": "#2ca02c",
    "群体组织": "#d62728",
}


def load_data() -> pd.DataFrame:
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Results file not found: {RESULTS_FILE}")

    df = pd.read_csv(RESULTS_FILE)
    df = df.copy()

    df["qwen_sentiment"] = pd.to_numeric(df["qwen_sentiment"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["qwen_bucket"] = df["qwen_bucket"].fillna("").astype(str).str.strip().replace({"None": "", "nan": ""})

    return df


def create_negative_two_bucket_chart(df: pd.DataFrame) -> go.Figure:
    negative_two = df[(df["qwen_sentiment"] == -2) & (df["qwen_bucket"] != "")].copy()
    bucket_counts = negative_two["qwen_bucket"].value_counts().reindex(BUCKET_ORDER, fill_value=0)
    total = int(bucket_counts.sum())
    percentages = (bucket_counts / total * 100).round(1) if total else bucket_counts * 0

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=bucket_counts.index.tolist(),
            y=bucket_counts.values.tolist(),
            marker_color=[BUCKET_COLORS[bucket] for bucket in bucket_counts.index],
            text=[f"{count}<br>({pct:.1f}%)" for count, pct in zip(bucket_counts.values, percentages.values)],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>数量: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"-2 分布 (n={total})",
        xaxis_title="类别",
        yaxis_title="文章数",
        template="plotly_white",
        height=600,
        width=1100,
        font=dict(family="PingFang SC, Heiti SC, Arial Unicode MS, sans-serif", size=14),
        margin=dict(t=80, l=60, r=30, b=60),
    )

    return fig


def create_bucket_distribution_by_year_chart(df: pd.DataFrame) -> go.Figure:
    yearly = df[(df["qwen_bucket"] != "") & df["year"].notna()].copy()
    yearly["year"] = yearly["year"].astype(int)

    yearly_counts = (
        yearly.groupby(["year", "qwen_bucket"]).size().unstack(fill_value=0).reindex(columns=BUCKET_ORDER, fill_value=0)
    )
    yearly_pct = yearly_counts.div(yearly_counts.sum(axis=1), axis=0) * 100

    fig = go.Figure()

    for bucket in BUCKET_ORDER:
        fig.add_trace(
            go.Bar(
                x=yearly_pct.index.tolist(),
                y=yearly_pct[bucket].round(1).tolist(),
                name=bucket,
                marker_color=BUCKET_COLORS[bucket],
                hovertemplate=f"<b>%{{x}}</b><br>{bucket}: %{{y:.1f}}%<extra></extra>",
            )
        )

    fig.update_layout(
        title="按年份的分布",
        xaxis_title="年份",
        yaxis_title="占比 (%)",
        barmode="stack",
        template="plotly_white",
        height=700,
        width=1200,
        font=dict(family="PingFang SC, Heiti SC, Arial Unicode MS, sans-serif", size=14),
        legend_title_text="类别",
        margin=dict(t=80, l=60, r=30, b=60),
    )

    return fig


def main() -> None:
    df = load_data()

    negative_two_fig = create_negative_two_bucket_chart(df)
    yearly_fig = create_bucket_distribution_by_year_chart(df)

    negative_two_png = OUTPUT_DIR / "zhong_split_negative_two_bucket_distribution.png"
    yearly_png = OUTPUT_DIR / "zhong_split_bucket_distribution_by_year.png"

    negative_two_fig.write_image(str(negative_two_png), width=1100, height=600, scale=2)
    yearly_fig.write_image(str(yearly_png), width=1200, height=700, scale=2)

    negative_two_counts = df[(df["qwen_sentiment"] == -2) & (df["qwen_bucket"] != "")]["qwen_bucket"].value_counts().reindex(BUCKET_ORDER, fill_value=0)
    yearly_counts = (
        df[(df["qwen_bucket"] != "") & df["year"].notna()]
        .assign(year=lambda frame: frame["year"].astype(int))
        .groupby(["year", "qwen_bucket"]).size().unstack(fill_value=0).reindex(columns=BUCKET_ORDER, fill_value=0)
    )

    print(f"Loaded {len(df):,} rows from {RESULTS_FILE}")
    print("\nSentiment -2 bucket counts:")
    print(negative_two_counts)
    print("\nBucket counts by year:")
    print(yearly_counts)
    print("\nSaved:")
    print(negative_two_png)
    print(yearly_png)


if __name__ == "__main__":
    main()