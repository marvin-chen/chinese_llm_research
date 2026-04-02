#!/usr/bin/env python3
"""
Clear all analysis progress and start fresh
"""

import os
import pandas as pd

def clear_analysis_progress():
    """Remove all analysis files to start fresh"""
    files_to_remove = [
        'qwen_analysis_progress.json',
        'qwen_analysis_results.csv'
    ]
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed {file}")
        else:
            print(f"INFO: {file} doesn't exist")
    
    print("\nAll analysis progress cleared!")
    print("TIP: You can now run 'python batch_analyze.py' to start fresh")

if __name__ == "__main__":
    clear_analysis_progress()