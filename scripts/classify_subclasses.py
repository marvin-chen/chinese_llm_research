#!/usr/bin/env python3
"""
Classify subclass targets for three buckets using Ollama (or heuristics).

Produces three CSVs under `results/`:
- results/ren_lun_guan_xi.csv
- results/qun_ti_zu_zhi.csv
- results/chou_xiang_gai_nian.csv

Usage (example):
    python scripts/classify_subclasses.py --input results/zhong_split_analysis_results.csv \
            --model llama2

If Ollama is not available, run with `--dry-run` to use simple heuristics.
"""
import argparse
import json
import logging
import time
from typing import Dict

import pandas as pd
import subprocess


LOG = logging.getLogger(__name__)


def call_ollama(prompt: str, model: str, timeout: int = 30) -> str:
    """Call Ollama via the local CLI: `ollama run <model> <prompt>` and return stdout.

    This matches the project's other analysis scripts which invoke Ollama through
    the `ollama` subprocess instead of the HTTP API.
    """
    try:
        result = subprocess.run(
            ['ollama', 'run', model, prompt], capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or 'ollama returned non-zero exit code')
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        raise RuntimeError(f'Ollama timeout after {timeout}s')


def parse_json_from_text(text: str) -> Dict:
    # try to find first JSON object in text
    try:
        return json.loads(text)
    except Exception:
        # fallback: attempt to locate a JSON substring
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
    raise ValueError('no JSON found in response')


def heuristic_classify(text: str, bucket: str) -> Dict:
    t = text or ""
    low = t.lower()
    if bucket == '群体组织':
        keywords_china = ['中国', '中华人民共和国', '中国共产党', '爱国', '为国', '爱党', '民族']
        for k in keywords_china:
            if k in t:
                return {'subclass': '中国', 'confidence': 0.6, 'explanation': f'keyword match: {k}'}
        return {'subclass': '其他', 'confidence': 0.5, 'explanation': 'no China keywords found'}
    if bucket == '人伦关系':
        keywords_partner = ['老婆', '老公', '男票', '女票', '伴侣', '恋爱', '结婚', '丈夫', '妻子']
        for k in keywords_partner:
            if k in t:
                return {'subclass': '伴侣', 'confidence': 0.6, 'explanation': f'keyword match: {k}'}
        keywords_idol = ['粉丝', '偶像', '明星', '歌手', '演员', 'idol', '饭', '粉']
        for k in keywords_idol:
            if k in low:
                return {'subclass': '偶像', 'confidence': 0.6, 'explanation': f'keyword match: {k}'}
        return {'subclass': '其他', 'confidence': 0.5, 'explanation': 'no partner/idol keywords'}
    if bucket == '抽象概念':
        if '自己' in t or '我' in t:
            return {'subclass': '自己', 'confidence': 0.6, 'explanation': 'mentions self'}
        return {'subclass': '其他', 'confidence': 0.5, 'explanation': 'no self mention'}
    return {'subclass': '其他', 'confidence': 0.0, 'explanation': 'unknown bucket'}


PROMPTS = {
    '群体组织': (
        "请判断以下微博文本中“忠”的对象是否为中国（包括中华人民共和国或中国共产党）或其他群体组织。"
        " 仅输出 JSON：{\"subclass\": \"中国\" 或 \"其他\", \"confidence\": 0-1, \"explanation\": \"简短说明\"}。\n\n文本：\n"),
    '人伦关系': (
        "请判断以下微博文本中“忠”的对象在“人伦关系”范畴内，属于哪一子类：\n"
        "选项：\"伴侣\"（恋爱对象/结婚对象等），\"偶像\"（明星/演员/歌手/球星等），\"其他\"。"
        " 仅输出 JSON：{\"subclass\": \"伴侣\"/\"偶像\"/\"其他\", \"confidence\": 0-1, \"explanation\": \"简短说明\"}。\n\n文本：\n"
    ),
    '抽象概念': (
        "请判断以下微博文本中“忠”的对象在“抽象概念”范畴内，属于哪一子类：\n"
        "选项：\"自己\"（忠的对象是自己），\"其他\"（忠的对象是其他抽象概念）。"
        " 仅输出 JSON：{\"subclass\": \"自己\"/\"其他\", \"confidence\": 0-1, \"explanation\": \"简短说明\"}。\n\n文本：\n"
    ),
}


def classify_df(df: pd.DataFrame, bucket: str, model: str, dry_run: bool, include_explanation: bool) -> pd.DataFrame:
    rows = []
    prompt_base = PROMPTS[bucket]
    for _, r in df.iterrows():
        text = r.get('text', '')
        result = None
        llm_resp = ''
        if not dry_run:
            prompt = prompt_base + text
            try:
                llm_resp = call_ollama(prompt, model)
                parsed = parse_json_from_text(llm_resp)
                result = parsed
            except Exception as e:
                LOG.warning('Ollama call/parse failed: %s; falling back to heuristic', e)
                result = heuristic_classify(text, bucket)
        else:
            result = heuristic_classify(text, bucket)
        out = dict(r)
        out['subclass'] = result.get('subclass')
        out['subclass_confidence'] = result.get('confidence')
        if include_explanation:
            out['subclass_explanation'] = result.get('explanation')
            out['llm_response'] = llm_resp
        rows.append(out)
        time.sleep(0.05)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--model', default='llama2')
    parser.add_argument('--dry-run', action='store_true', help='Do not call Ollama; use heuristics')
    parser.add_argument('--limit', type=int, default=0, help='(deprecated) Limit rows per bucket (0 = all)')
    parser.add_argument('--sample-frac', type=float, default=0.05, help='Fraction to sample per bucket (0=no sampling, 1=all)')
    parser.add_argument('--sample-n-max', type=int, default=5000, help='Maximum rows to sample per bucket (0=no limit)')
    parser.add_argument('--sample-random-state', dest='sample_random_state', type=int, default=42)
    parser.add_argument('--include-explanation', action='store_true', help='Include explanation and raw LLM response columns in outputs')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    df = pd.read_csv(args.input)
    buckets = {
        '人伦关系': 'results/ren_lun_guan_xi.csv',
        '群体组织': 'results/qun_ti_zu_zhi.csv',
        '抽象概念': 'results/chou_xiang_gai_nian.csv',
    }

    sample_frac = args.sample_frac
    sample_n_max = args.sample_n_max
    random_state = args.sample_random_state
    include_explanation = args.include_explanation

    for b, outpath in buckets.items():
        sub = df[df['qwen_bucket'] == b].copy()
        LOG.info('Bucket %s original rows: %d', b, len(sub))
        if len(sub) == 0:
            LOG.info('No rows for bucket %s, writing empty file %s', b, outpath.replace('.csv', '_sampled.csv'))
            pd.DataFrame().to_csv(outpath.replace('.csv', '_sampled.csv'), index=False)
            continue

        # determine sample size
        if sample_frac and sample_frac > 0:
            n = int(len(sub) * sample_frac)
        else:
            n = len(sub)
        if sample_n_max and sample_n_max > 0:
            n = min(n, sample_n_max)
        n = max(0, min(n, len(sub)))
        if n == 0:
            sampled = sub.iloc[0:0]
        else:
            sampled = sub.sample(n=n, random_state=random_state)

        LOG.info('Processing bucket %s sampled rows: %d', b, len(sampled))
        classified = classify_df(sampled, b, args.model, args.dry_run, include_explanation)
        outpath_sample = outpath.replace('.csv', '_sampled.csv')
        # drop explanation/llm response columns if not requested
        if not include_explanation:
            drop_cols = [c for c in ['subclass_explanation', 'llm_response'] if c in classified.columns]
            if drop_cols:
                classified = classified.drop(columns=drop_cols)
        classified.to_csv(outpath_sample, index=False)
        LOG.info('Wrote %s', outpath_sample)


if __name__ == '__main__':
    main()
