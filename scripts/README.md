Usage: classify subclassifications for three buckets

Run the classifier to produce three CSVs (one per bucket) with subclass labels.

Example:

```bash
# activate your venv first
python scripts/classify_subclasses.py \
  --input results/zhong_split_analysis_results.csv \
  --model llama2 \
  # Ollama is invoked via the local `ollama` CLI (e.g. `ollama run <model>`)
```

If you don't have Ollama running or want a quick test, use `--dry-run` to apply heuristics:

```bash
python scripts/classify_subclasses.py --input results/zhong_split_analysis_results.csv --dry-run
```

Outputs (in `results/`):

- `ren_lun_guan_xi.csv` — rows where `qwen_bucket` == 人伦关系 with subclass columns
- `qun_ti_zu_zhi.csv` — rows where `qwen_bucket` == 群体组织 with subclass columns
- `chou_xiang_gai_nian.csv` — rows where `qwen_bucket` == 抽象概念 with subclass columns

Notes:
- The script expects `pandas` and a working `ollama` CLI on your PATH. Install Python deps via `pip install -r requirements.txt` and Ollama separately.
- Ollama is called through the `ollama` subprocess (matching other analysis scripts); set `--model` to the desired Ollama model (default `llama2`).
