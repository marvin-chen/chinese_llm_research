#!/usr/bin/env python3
"""
Quick progress monitor for gender role analysis.
Shows real-time stats without slowing down the main analysis.
"""

import json
import time
from pathlib import Path
from datetime import datetime


def get_stats(progress_file: Path, results_file: Path):
    """Get current statistics from progress and results files."""
    stats = {
        "progress": None,
        "results": None,
        "timestamp": datetime.now().isoformat()
    }
    
    if progress_file.exists():
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                stats['progress'] = json.load(f)
        except Exception as e:
            print(f"Error reading progress: {e}")
    
    if results_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(results_file)
            processed = df['qwen_processed_at'].notna().sum()
            successful = df['qwen_sentiment'].notna().sum()
            errors = df['qwen_error'].notna().sum()
            stats['results'] = {
                'total': len(df),
                'processed': int(processed),
                'successful': int(successful),
                'errors': int(errors),
            }
        except Exception as e:
            print(f"Error reading results: {e}")
    
    return stats


def format_time(minutes):
    """Format minutes into hours:minutes."""
    if minutes < 60:
        return f"{minutes:.0f}m"
    else:
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:.0f}h {mins:.0f}m"


def monitor(interval=5):
    """Monitor progress of both files."""
    print("=" * 70)
    print("GENDER ROLE ANALYSIS PROGRESS MONITOR")
    print("=" * 70)
    print(f"Monitoring every {interval}s. Press Ctrl+C to stop.\n")
    
    results_dir = Path("results")
    files = [
        ("女主内", "nvzhunei_qwen_analysis_progress.json", "nvzhunei_qwen_analysis_results.csv"),
        ("男主外", "nanzhuwai_qwen_analysis_progress.json", "nanzhuwai_qwen_analysis_results.csv"),
    ]
    
    last_update = {}
    
    while True:
        try:
            for name, progress_file, results_file in files:
                progress_path = results_dir / progress_file
                results_path = results_dir / results_file
                
                stats = get_stats(progress_path, results_path)
                
                if stats['results'] is None:
                    print(f"[{name}] Not started yet or file not found")
                    continue
                
                r = stats['results']
                p = stats['progress'] if stats['progress'] else {}
                
                processed = r['processed']
                total = r['total']
                successful = r['successful']
                errors = r['errors']
                
                # Calculate stats
                pct = 100 * processed / max(total, 1)
                success_rate = 100 * successful / max(processed, 1) if processed > 0 else 0
                remaining = total - processed
                
                # Time estimate
                time_est = None
                if processed > 0 and remaining > 0:
                    avg_time_per = p.get('total_processed', 0) / (len(p.get('sessions', [])) * 5 + 1) if p.get('sessions') else 0.8
                    time_est = format_time(remaining * avg_time_per / 60)
                
                # Check if updated since last check
                last_processed = last_update.get(name, -1)
                new_posts = "⏳" if processed > last_processed else " "
                last_update[name] = processed
                
                # Print status
                print(f"{new_posts} [{name:6s}] {processed:4d}/{total:4d} ({pct:5.1f}%) | ✓ {successful:4d} ✗ {errors:3d} | Rate {success_rate:5.1f}%", end="")
                if time_est:
                    print(f" | ETA: {time_est}")
                else:
                    print()
            
            print("-" * 70)
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\n" + "=" * 70)
            print("Monitoring stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(interval)


if __name__ == "__main__":
    monitor(interval=5)
