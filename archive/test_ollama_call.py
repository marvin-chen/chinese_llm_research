#!/usr/bin/env python3

import subprocess
import json

# Test the exact same call that batch_analyze.py uses
SYSTEM_PROMPT = """你是中国文化专家，评估微博对"孝"的态度。严格按照评分标准分类。

评分标准：
• 0 (无关): "孝"是地名/人名/书名，与内容无关
• 0 (中性): 娱乐/玩笑语境，无道德判断
• -1 (轻度负面): 表达压力/矛盾，但未否定孝道价值
• -2 (强烈负面): 强烈批评孝道限制自由，视为负担/债务
• +1 (轻度正面): 一般性赞扬孝道，无具体细节（如"孝顺很重要"、征婚提及孝顺）
• +2 (强烈正面): 详细描述具体孝行，或严厉谴责不孝

置信度评分（百分比0-100%）：
• 90-100%: 非常确信，文本明确表达态度
• 70-89%: 比较确信，有充分证据支持判断  
• 50-69%: 一般确信，基于常见模式判断
• 30-49%: 不太确信，可能有歧义
• 0-29%: 很不确信，难以判断

仅返回JSON格式：{"relevant": boolean, "sentiment": int, "bucket": "string", "confidence": int, "reasoning": "string"}"""

def test_ollama_call():
    text = "孝顺父母，努力工作"
    full_prompt = f"{SYSTEM_PROMPT}\n\nPost: \"{text}\"\n\nJSON:"
    
    print("Testing Ollama subprocess call...")
    try:
        result = subprocess.run(
            ['ollama', 'run', 'qwen2.5:7b'],
            input=full_prompt,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=60
        )
        
        print(f"Return code: {result.returncode}")
        print(f"Output: {result.stdout[:200]}...")
        if result.stderr:
            print(f"Error: {result.stderr}")
            
        # Try to extract JSON
        if result.returncode == 0 and result.stdout.strip():
            try:
                start = result.stdout.find('{')
                end = result.stdout.rfind('}') + 1
                if start != -1 and end > start:
                    json_str = result.stdout[start:end]
                    parsed = json.loads(json_str)
                    print(f"Successfully parsed JSON: {parsed}")
                else:
                    print("ERROR: No JSON found in output")
            except Exception as e:
                print(f"ERROR: JSON parsing failed: {e}")
                
    except Exception as e:
        print(f"ERROR: Subprocess call failed: {e}")

if __name__ == "__main__":
    test_ollama_call()