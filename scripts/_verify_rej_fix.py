import json
errors = 0
rej_fix = []
with open("data/pairs/manual_pairs.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if r["id"].startswith("rej_fix_"):
                rej_fix.append(r)
        except Exception as e:
            print(f"Line {i} ERROR: {e}")
            errors += 1

print(f"JSON errors: {errors}")
print(f"rej_fix records: {len(rej_fix)}")
for r in rej_fix:
    ct  = r["plotspec"].get("chart_type", "?")
    df  = str(r["plotspec"].get("data_filter", ""))
    ctx = bool(r.get("data_context"))
    print(f"  {r['id'][:45]:45s}  ct={ct:8s}  filter={df[:28]:28s}  ctx={ctx}")
