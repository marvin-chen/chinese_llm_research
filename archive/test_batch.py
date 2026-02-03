#!/usr/bin/env python3

import sys
sys.path.append('.')

from batch_analyze import BatchWeiboAnalyzer

def test_batch():
    try:
        print("Testing batch analysis...")
        analyzer = BatchWeiboAnalyzer("weibo_xiao_sample_equal_per_month.csv", filter_relevant=True)
        print("Analyzer created successfully")
        
        # Try to run just one batch to trigger the error
        print("Running one test batch...")
        analyzer.run_batch(batch_size=5, max_batches=1)
        print("Test completed successfully")
        
    except Exception as e:
        print(f"ERROR: Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_batch()