import json, statistics, time, urllib.request
KEY="REDACTED_SET_SGLANG_API_KEY"
BASE="http://10.100.8.1:8940"
def gen(ids, mn):
    t0=time.perf_counter()
    req=urllib.request.Request(BASE+"/generate", data=json.dumps({"input_ids":ids,"sampling_params":{"max_new_tokens":mn,"temperature":0.0}}).encode(),
        headers={"Content-Type":"application/json","Authorization":"Bearer "+KEY})
    with urllib.request.urlopen(req, timeout=1800) as r: body=json.loads(r.read().decode())
    return time.perf_counter()-t0, body["meta_info"]
print("L  | warmup(s) | decode(ms/token) | accept | cached")
for L in (4096, 32768, 100000):
    ids=[100+(i%50) for i in range(L)]
    tw,_=gen(ids,2)  # warm cache
    rows=[]
    for _ in range(5):
        t,m=gen(ids,32); rows.append((t,m["completion_tokens"],m.get("spec_accept_length")))
    tps=[t/ct for t,ct,_ in rows]
    print(f"L={L:6d} warmup={tw:6.1f}s decode={statistics.median(tps)*1000:.2f} ms/tok accept={rows[0][2]} cached={m.get('cached_tokens')} ct={rows[0][1]}")
