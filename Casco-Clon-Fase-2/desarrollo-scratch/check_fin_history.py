import json
tp = r'C:\Users\migue\.gemini\antigravity\brain\708952eb-13ab-449a-b9c1-cf788f04feb2\.system_generated\logs\transcript_full.jsonl'
with open(tp, encoding='utf-8') as f:
    for i, line in enumerate(f):
        if 2980 <= i <= 3025 or 3170 <= i <= 3210:
            d = json.loads(line)
            if d.get('thinking') or (d.get('type') == 'PLANNER_RESPONSE'):
                print(f"=== STEP {i} ===")
                print((d.get('thinking') or '')[:1500])
                print((d.get('content') or '')[:1000])
