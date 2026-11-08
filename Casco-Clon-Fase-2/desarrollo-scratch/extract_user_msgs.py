import json, re
tp = r'C:\Users\migue\.gemini\antigravity\brain\708952eb-13ab-449a-b9c1-cf788f04feb2\.system_generated\logs\transcript_full.jsonl'
out = open(r'c:\Users\migue\Proyectos IA varios\Miscelanea\scratch\user_msgs.txt', 'w', encoding='utf-8')
with open(tp, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        if d.get('type') == 'USER_INPUT':
            c = str(d.get('content', ''))
            m = re.search(r'<USER_REQUEST>(.*?)</USER_REQUEST>', c, re.S)
            txt = m.group(1).strip() if m else c[:300]
            media = [x.get('uri') for x in (d.get('media') or [])]
            out.write(f'=== step {i} media={media}\n{txt[:3000]}\n\n')
out.close()
