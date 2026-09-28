import json

p = '/home/agent_7e5e4c51-2bc5-45c0-82ce-e259ee7251e6/.local/share/kilo/tool-output/tool_0e6a19311001rFIL9RrEV8uAEF'
d = json.load(open(p))
print('TOP KEYS:', list(d.keys()))
hits = d.get('hits', [])
print('NUM HITS:', len(hits))
for i, h in enumerate(hits):
    print('=' * 80)
    print(f'[{i}] author={h.get("author")} date={h.get("created_at")} objectID={h.get("objectID")} parent_id={h.get("parent_id")} story_id={h.get("story_id")}')
    print(f'story_title={h.get("story_title")}')
    print(f'story_url={h.get("story_url")}')
    print('--- comment_text ---')
    print(h.get('comment_text'))
