#!/usr/bin/env python3
"""
Quick script to check reasoning extraction progress
"""

import json
import os
from datetime import datetime

progress_file = 'results/reasoning_extraction_progress.json'

if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    
    total = 129
    processed = len(progress.get('processed_ids', []))
    success = progress.get('success_count', 0)
    errors = progress.get('error_count', 0)
    last_updated = progress.get('last_updated', 'Unknown')
    
    print("="*60)
    print("REASONING EXTRACTION PROGRESS")
    print("="*60)
    print(f"Processed: {processed}/{total} posts ({processed/total*100:.1f}%)")
    print(f"  ✓ Successful: {success}")
    print(f"  ✗ Errors: {errors}")
    print(f"Last updated: {last_updated}")
    
    if processed < total:
        remaining = total - processed
        print(f"\nRemaining: {remaining} posts (~{remaining} minutes)")
    else:
        print("\n✓ Extraction complete!")
else:
    print("No progress file found yet. Extraction may have just started.")
