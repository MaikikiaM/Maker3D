import json
tp = r'C:\Users\migue\.gemini\antigravity\brain\708952eb-13ab-449a-b9c1-cf788f04feb2\.system_generated\logs\transcript_full.jsonl'
out = open(r'c:\Users\migue\Proyectos IA varios\Miscelanea\scratch\steps_context.txt', 'w', encoding='utf-8')
with open(tp, encoding='utf-8') as f:
    for i, line in enumerate(f):
        if 3215 <= i <= 3240 or 3450 <= i <= 3475:
            d = json.loads(line)
            if d.get('content'):
                out.write(f"=== STEP {i} ({d.get('type')}) ===\n")
                out.write(str(d.get('content'))[:6000] + "\n\n")
out.close()
