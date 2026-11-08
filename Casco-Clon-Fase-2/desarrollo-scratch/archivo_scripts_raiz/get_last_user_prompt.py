import json

log_path = r'C:\Users\migue\.gemini\antigravity\brain\708952eb-13ab-449a-b9c1-cf788f04feb2\.system_generated\logs\transcript_full.jsonl'
with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    user_steps = []
    for line in f:
        try:
            data = json.loads(line)
            if data.get('type') == 'USER_INPUT':
                user_steps.append(data)
        except Exception:
            pass

print(f"Total user steps: {len(user_steps)}")
if user_steps:
    last = user_steps[-1]
    print(f"Last step index: {last.get('step_index')}")
    with open('last_user_prompt.txt', 'w', encoding='utf-8') as out:
        out.write(last.get('content', ''))
    print("Written to last_user_prompt.txt")
