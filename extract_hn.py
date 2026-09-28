import json, html, re

p = "/home/agent_7e5e4c51-2bc5-45c0-82ce-e259ee7251e6/.local/share/kilo/tool-output/tool_0e698bca7001NPLrjXWHn5cQUT"
d = json.load(open(p))
print("TOP KEYS:", list(d.keys()))
print("nbHits:", d.get("nbHits"), "page:", d.get("page"), "nbPages:", d.get("nbPages"), "hitsPerPage:", d.get("hitsPerPage"))
hits = d.get("hits", [])
print("NUM HITS:", len(hits))
print("=" * 80)

def clean(text):
    if not text:
        return ""
    t = html.unescape(text)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

for i, h in enumerate(hits):
    print(f"### HIT {i}")
    print("author:", h.get("author"))
    print("created_at:", h.get("created_at"))
    print("objectID:", h.get("objectID"))
    print("parent_id:", h.get("parent_id"))
    print("story_id:", h.get("story_id"))
    print("story_title:", h.get("story_title"))
    print("story_url:", h.get("story_url"))
    print("comment_text:", clean(h.get("comment_text")))
    print("-" * 80)
